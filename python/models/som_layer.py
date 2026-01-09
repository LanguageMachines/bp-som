"""
Self-Organizing Map Layer for BP-SOM

Implements the SOM component that continuously learns from hidden layer activations
and provides clustering-based error signals during backpropagation.

Based on the original C implementation in include/bpsom/som.h
Optimized with vectorized operations for GPU acceleration.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SelfOrganizingMap(nn.Module):
    """
    Self-Organizing Map for BP-SOM architecture.
    Optimized with vectorized operations for GPU acceleration.

    The SOM maintains a 2D grid of prototype vectors and provides:
    1. Continuous unsupervised learning from activations
    2. Clustering-based error signals for backpropagation
    3. Class-specific organization through partial winner tracking
    """

    def __init__(
        self,
        input_dim: int,
        grid_size: int,
        num_classes: int,
        som_lr_max: float = 0.20,
        som_lr_min: float = 0.05,
        som_context_max: int = 2,
        som_context_min: int = 0,
        reliability_threshold: float = 0.95,
        som_error_weight: float = 0.25,
    ):
        """
        Args:
            input_dim: Dimension of input activations
            grid_size: Size of SOM grid (grid_size x grid_size)
            num_classes: Number of classification classes
            som_lr_max: Maximum SOM learning rate
            som_lr_min: Minimum SOM learning rate
            som_context_max: Maximum neighborhood radius
            som_context_min: Minimum neighborhood radius
            reliability_threshold: Minimum reliability to use SOM error (0-1)
            som_error_weight: Weight of SOM error in combined gradient (α)
        """
        super().__init__()

        self.input_dim = input_dim
        self.grid_size = grid_size
        self.num_classes = num_classes

        # SOM parameters
        self.som_lr_max = som_lr_max
        self.som_lr_min = som_lr_min
        self.som_context_max = som_context_max
        self.som_context_min = som_context_min
        self.reliability_threshold = reliability_threshold
        self.som_error_weight = som_error_weight

        # SOM grid storage - mirroring C implementation
        # som_network_vector[x][y][feature_dim]
        self.register_buffer(
            'som_vectors',
            torch.rand(grid_size, grid_size, input_dim) * 0.5 + 0.5
        )

        # Class labeling: som_cell_teller[x][y][class_id]
        # Track count of each class mapped to each cell
        self.register_buffer(
            'cell_class_counts',
            torch.zeros(grid_size, grid_size, num_classes + 1, dtype=torch.long)
        )

        # Majority class label for each cell
        self.register_buffer(
            'cell_labels',
            torch.zeros(grid_size, grid_size, dtype=torch.long)
        )

        # Reliability percentage (0-100) for each cell
        self.register_buffer(
            'cell_reliability',
            torch.zeros(grid_size, grid_size, dtype=torch.float)
        )

        # Statistics
        self.total_distance = 0.0
        self.som_usage_count = 0
        self.total_examples = 0

    def get_distance_matrix(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distance matrix between batch x and vectors w.
        Vectorized computation for GPU acceleration.

        Args:
            x: (B, D) batch of activation vectors
            w: (N, D) SOM prototype vectors (flattened grid)

        Returns:
            (B, N) distance matrix
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        w_norm = (w ** 2).sum(1).view(1, -1)
        dist = x_norm + w_norm - 2.0 * torch.mm(x, w.t())
        return torch.clamp(dist, 0.0, None)

    def update_som_network_batch(
        self,
        activations: torch.Tensor,
        bmu_indices: torch.Tensor,
        som_lr: float,
        som_context: int
    ):
        """
        Update SOM prototypes using batch neighborhood learning.
        Vectorized for efficient GPU computation.

        Args:
            activations: (B, D) batch of activations
            bmu_indices: (B,) flattened BMU indices
            som_lr: Current SOM learning rate
            som_context: Current neighborhood radius
        """
        B = activations.size(0)
        G = self.grid_size

        # Convert flat indices to 2D coordinates
        bmu_x = bmu_indices // G
        bmu_y = bmu_indices % G

        # Create coordinate grid (G, G)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(G, device=activations.device),
            torch.arange(G, device=activations.device),
            indexing='ij'
        )

        # Compute Chebyshev distances for all grid cells to all BMUs
        # Expand to (B, G, G)
        dist_x = torch.abs(x_grid.unsqueeze(0) - bmu_x.view(B, 1, 1))
        dist_y = torch.abs(y_grid.unsqueeze(0) - bmu_y.view(B, 1, 1))
        dist_chebyshev = torch.max(dist_x, dist_y)

        # Compute update powers with neighborhood mask
        mask = (dist_chebyshev <= som_context).float()
        powers = som_lr / (2.0 ** dist_chebyshev.float())
        powers = powers * mask  # (B, G, G)

        # Accumulate updates: W_new = W_old + sum_k(p_k * x_k) - W_old * sum_k(p_k)
        denom = powers.sum(dim=0).unsqueeze(-1)  # (G, G, 1)

        powers_flat = powers.view(B, -1)  # (B, G*G)
        num_flat = torch.matmul(powers_flat.t(), activations)  # (G*G, D)
        numerator = num_flat.view(G, G, -1)

        self.som_vectors += (numerator - self.som_vectors * denom)

    def forward(
        self,
        activations: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epochs: int = 100,
        training: bool = True
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Process activations through SOM using vectorized operations.

        During training:
        - Updates SOM prototypes
        - Computes SOM error for backpropagation

        During evaluation:
        - Only computes statistics

        Args:
            activations: [batch_size, input_dim] hidden layer activations
            labels: [batch_size] class labels (required for training)
            epoch: Current epoch number
            max_epochs: Total number of epochs (for scheduling)
            training: Whether in training mode

        Returns:
            som_errors: [batch_size, input_dim] SOM-based errors (or None)
            stats: Dictionary of statistics
        """
        batch_size = activations.size(0)

        # Compute current SOM parameters (scheduled decay)
        progress = (max_epochs - epoch) / max_epochs if max_epochs > 0 else 0
        progress = max(0, min(1, progress))

        som_context = self.som_context_min + int(
            (progress ** 4) * (self.som_context_max - self.som_context_min)
        )
        som_lr = self.som_lr_min + (progress ** 1) * (self.som_lr_max - self.som_lr_min)

        # Flatten SOM vectors for distance computation: (G*G, D)
        flat_som = self.som_vectors.view(-1, self.input_dim)

        # 1. Compute Distances and Global BMUs (vectorized)
        dists = self.get_distance_matrix(activations, flat_som)  # (B, G*G)
        min_dists, bmu_indices = torch.min(dists, dim=1)  # (B,)

        # Statistics
        total_dist = torch.sqrt(min_dists).sum().item()
        som_used_count = 0
        som_errors = None

        if training and labels is not None:
            # 2. Compute Class-Specific BMUs and Errors (vectorized)
            # Mask distances where cell_label != label
            flat_labels = self.cell_labels.view(-1)
            # (B, G*G) mask: True where cell label doesn't match sample label
            label_mask = (flat_labels.unsqueeze(0) != labels.unsqueeze(1))

            class_dists = dists.clone()
            class_dists[label_mask] = float('inf')

            _, class_bmu_indices = torch.min(class_dists, dim=1)

            # Get prototypes and reliability for class BMUs
            class_bmu_vectors = flat_som[class_bmu_indices]  # (B, D)
            flat_reliability = self.cell_reliability.view(-1)
            reliabilities = flat_reliability[class_bmu_indices] / 100.0  # (B,)

            # Compute SOM errors
            # Valid if reliability >= threshold
            valid_mask = (reliabilities >= self.reliability_threshold).float().unsqueeze(1)
            som_errors = 0.01 * reliabilities.unsqueeze(1) * (class_bmu_vectors - activations)
            som_errors = som_errors * valid_mask

            som_used_count = (reliabilities >= self.reliability_threshold).sum().item()

            # 3. Update SOM Network (vectorized batch update)
            if min_dists.mean() > 1e-4:
                self.update_som_network_batch(activations, bmu_indices, som_lr, som_context)

        self.total_distance += total_dist
        self.som_usage_count += som_used_count
        self.total_examples += batch_size

        stats = {
            'som_lr': som_lr,
            'som_context': som_context,
            'avg_distance': total_dist / batch_size if batch_size > 0 else 0,
            'som_usage_pct': (som_used_count / batch_size * 100) if batch_size > 0 else 0,
        }

        return som_errors, stats

    def update_cell_labels(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Update SOM cell class labels based on training data using batched processing.
        Called after each epoch on training set.

        Args:
            activations: [num_examples, input_dim]
            labels: [num_examples] class labels
        """
        self.cell_class_counts.zero_()

        flat_som = self.som_vectors.view(-1, self.input_dim)
        batch_size = 1024  # Process in chunks to save memory

        with torch.no_grad():
            for i in range(0, activations.size(0), batch_size):
                batch_act = activations[i:i + batch_size]
                batch_lbl = labels[i:i + batch_size]

                dists = self.get_distance_matrix(batch_act, flat_som)
                _, bmu_indices = torch.min(dists, dim=1)

                bmu_x = bmu_indices // self.grid_size
                bmu_y = bmu_indices % self.grid_size

                # Accumulate counts
                for j, label in enumerate(batch_lbl):
                    self.cell_class_counts[bmu_x[j], bmu_y[j], label] += 1

        # Update labels and reliability
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                counts = self.cell_class_counts[x, y]
                total = counts.sum().item()

                if total > 0:
                    max_count, max_class = counts.max(0)
                    self.cell_labels[x, y] = max_class
                    self.cell_reliability[x, y] = (max_count.float() / total) * 100
                else:
                    self.cell_labels[x, y] = 0
                    self.cell_reliability[x, y] = 0

    def reset_statistics(self):
        """Reset epoch statistics."""
        self.total_distance = 0.0
        self.som_usage_count = 0
        self.total_examples = 0

    def get_statistics(self) -> dict:
        """Get accumulated statistics."""
        return {
            'avg_distance': self.total_distance / max(1, self.total_examples),
            'som_usage_pct': (self.som_usage_count / max(1, self.total_examples)) * 100,
            'total_examples': self.total_examples,
        }
