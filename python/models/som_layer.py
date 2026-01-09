"""
Self-Organizing Map Layer for BP-SOM

Implements the SOM component that continuously learns from hidden layer activations
and provides clustering-based error signals during backpropagation.

Based on the original C implementation in include/bpsom/som.h
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class SelfOrganizingMap(nn.Module):
    """
    Self-Organizing Map for BP-SOM architecture.

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

    def compute_distance(self, activation: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance between activation and prototype.
        Based on count_distance in som.h:13

        Args:
            activation: [input_dim] activation vector
            prototype: [input_dim] prototype vector

        Returns:
            Squared Euclidean distance
        """
        return torch.sum((activation - prototype) ** 2)

    def find_best_matching_unit(
        self,
        activation: torch.Tensor,
        class_label: Optional[int] = None
    ) -> Tuple[int, int, float]:
        """
        Find best matching unit (BMU) in SOM grid.
        Optionally constrain to cells with specific class label (partial winner).

        Based on process_som_vectors in som.h:24-98

        Args:
            activation: [input_dim] activation vector
            class_label: If provided, only consider cells with this label

        Returns:
            (x, y, distance): Coordinates and distance of BMU
        """
        min_dist = float('inf')
        bmu_x, bmu_y = 0, 0

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Skip if constraining by class and this cell doesn't match
                if class_label is not None and self.cell_labels[x, y] != class_label:
                    continue

                dist = self.compute_distance(activation, self.som_vectors[x, y])

                if dist < min_dist:
                    min_dist = dist
                    bmu_x, bmu_y = x, y

        # Convert to float (handles both tensor and float('inf') cases)
        if isinstance(min_dist, torch.Tensor):
            min_dist = min_dist.item()

        return bmu_x, bmu_y, min_dist

    def update_som_network(
        self,
        activation: torch.Tensor,
        bmu_x: int,
        bmu_y: int,
        som_lr: float,
        som_context: int
    ):
        """
        Update SOM prototypes using neighborhood learning.
        Based on update_som_network in som.h:147-159

        Args:
            activation: [input_dim] current activation
            bmu_x, bmu_y: Best matching unit coordinates
            som_lr: Current SOM learning rate
            som_context: Current neighborhood radius
        """
        # Update BMU and neighborhood
        for x in range(max(0, bmu_x - som_context), min(self.grid_size, bmu_x + som_context + 1)):
            for y in range(max(0, bmu_y - som_context), min(self.grid_size, bmu_y + som_context + 1)):
                # Manhattan distance for neighborhood
                dist = max(abs(x - bmu_x), abs(y - bmu_y))

                # Update power decreases with distance: lr / 2^dist
                update_power = som_lr / (2.0 ** dist)

                # Update: vector += update_power * (activation - vector)
                self.som_vectors[x, y] += update_power * (activation - self.som_vectors[x, y])

    def compute_som_error(
        self,
        activation: torch.Tensor,
        class_label: int,
        reliability: float,
        prototype: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute SOM-based error signal for backpropagation.
        Based on bp.h:111-119

        Args:
            activation: [input_dim] current activation
            class_label: True class label
            reliability: Reliability of the partial winner cell
            prototype: [input_dim] partial winner prototype

        Returns:
            SOM error vector or None if reliability too low
        """
        # Only use SOM error if reliability meets threshold
        if reliability < self.reliability_threshold:
            return None

        # som_error = 0.01 * reliability * (prototype - activation)
        # The 0.01 factor matches the C implementation
        som_error = 0.01 * reliability * (prototype - activation)

        return som_error

    def forward(
        self,
        activations: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epochs: int = 100,
        training: bool = True
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Process activations through SOM.

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
        # Based on update_som_context_and_lr in bpsom.cc:362
        progress = (max_epochs - epoch) / max_epochs if max_epochs > 0 else 0
        progress = max(0, min(1, progress))  # Clamp to [0, 1]

        som_context = self.som_context_min + int(
            (progress ** 4) * (self.som_context_max - self.som_context_min)
        )
        som_lr = self.som_lr_min + (progress ** 1) * (self.som_lr_max - self.som_lr_min)

        # Initialize output
        som_errors = [] if training and labels is not None else None

        # Statistics
        total_dist = 0.0
        som_used_count = 0

        # Process each example in batch
        for i in range(batch_size):
            activation = activations[i]  # [input_dim]
            label = labels[i].item() if labels is not None else None

            # Find best matching unit (overall winner)
            bmu_x, bmu_y, min_dist = self.find_best_matching_unit(activation)
            total_dist += np.sqrt(min_dist)

            if training and labels is not None:
                # Find partial winner (best matching unit with same class)
                part_x, part_y, part_dist = self.find_best_matching_unit(activation, class_label=label)

                # Get reliability of partial winner
                reliability = self.cell_reliability[part_x, part_y].item() / 100.0  # Convert from percentage

                # Compute SOM error
                som_error = self.compute_som_error(
                    activation,
                    label,
                    reliability,
                    self.som_vectors[part_x, part_y]
                )

                if som_error is not None:
                    som_errors.append(som_error)
                    som_used_count += 1
                else:
                    som_errors.append(torch.zeros_like(activation))

                # Update SOM network
                if min_dist > 1e-4:  # DONOTHING threshold
                    self.update_som_network(activation, bmu_x, bmu_y, som_lr, som_context)

        # Convert errors to tensor
        if som_errors is not None:
            som_errors = torch.stack(som_errors)  # [batch_size, input_dim]

        # Update statistics
        self.total_distance += total_dist
        self.som_usage_count += som_used_count
        self.total_examples += batch_size

        # Compile stats
        stats = {
            'som_lr': som_lr,
            'som_context': som_context,
            'avg_distance': total_dist / batch_size if batch_size > 0 else 0,
            'som_usage_pct': (som_used_count / batch_size * 100) if batch_size > 0 else 0,
        }

        return som_errors, stats

    def update_cell_labels(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Update SOM cell class labels based on training data.
        Called after each epoch on training set.
        Based on count_som_cell_winners in som.h:197-230

        Args:
            activations: [num_examples, input_dim]
            labels: [num_examples] class labels
        """
        # Reset counters
        self.cell_class_counts.zero_()

        # Count class mappings for each cell
        with torch.no_grad():
            for i in range(activations.size(0)):
                activation = activations[i]
                label = labels[i].item()

                bmu_x, bmu_y, _ = self.find_best_matching_unit(activation)
                self.cell_class_counts[bmu_x, bmu_y, label] += 1

        # Determine majority label and reliability for each cell
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                counts = self.cell_class_counts[x, y]  # [num_classes + 1]
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
