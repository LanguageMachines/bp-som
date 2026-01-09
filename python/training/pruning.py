"""
Unit Pruning for BP-SOM

Implements dynamic pruning of inactive hidden units based on activation statistics.
Based on prune_if_possible and prune_weights in bp.h:304-352
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class UnitPruner:
    """
    Handles pruning of inactive hidden units in BP-SOM layer.

    Units with low activation variance (std < threshold) are considered
    inactive and can be pruned to reduce model complexity.
    """

    def __init__(
        self,
        prune_threshold: float = 0.02,
        min_units: int = 1,
        enabled: bool = True,
    ):
        """
        Args:
            prune_threshold: Std threshold below which units are pruned
            min_units: Minimum number of units to keep
            enabled: Whether pruning is enabled
        """
        self.prune_threshold = prune_threshold
        self.min_units = min_units
        self.enabled = enabled

        # Track pruned units per epoch
        self.pruning_history = []

    def identify_prunable_units(
        self,
        mean_activations: torch.Tensor,
        std_activations: torch.Tensor
    ) -> List[int]:
        """
        Identify units that should be pruned.

        Based on prune_if_possible in bp.h:304-319

        Args:
            mean_activations: [hidden_dim] mean activation per unit
            std_activations: [hidden_dim] std activation per unit

        Returns:
            List of unit indices to prune
        """
        if not self.enabled:
            return []

        hidden_dim = std_activations.size(0)

        # Find units with std below threshold
        prunable_mask = std_activations <= self.prune_threshold

        # Get indices
        prunable_indices = torch.where(prunable_mask)[0].tolist()

        # Ensure we don't prune below minimum
        current_units = hidden_dim
        units_to_remove = len(prunable_indices)

        if current_units - units_to_remove < self.min_units:
            # Can't prune this many, limit pruning
            max_to_prune = current_units - self.min_units
            if max_to_prune > 0:
                # Keep units with lowest std
                sorted_indices = torch.argsort(std_activations)
                prunable_indices = sorted_indices[:max_to_prune].tolist()
            else:
                prunable_indices = []

        return prunable_indices

    def prune_units(
        self,
        model,
        units_to_prune: List[int],
        mean_activations: torch.Tensor
    ) -> bool:
        """
        Prune specified units from BP-SOM hidden layer.

        Based on prune_weights in bp.h:321-352

        This is complex because we need to:
        1. Remove units from hidden layer weights
        2. Absorb pruned units' contribution into bias
        3. Update SOM grid prototypes
        4. Update classifier weights

        Args:
            model: BPSOMBertForSequenceClassification model
            units_to_prune: List of unit indices to prune
            mean_activations: Mean activations for bias absorption

        Returns:
            True if pruning was performed
        """
        if not units_to_prune:
            return False

        print(f"\nPruning {len(units_to_prune)} units: {units_to_prune}")

        hidden_layer = model.bpsom_hidden
        classifier = model.classifier

        # Get current dimensions
        input_dim = hidden_layer.input_dim
        hidden_dim = hidden_layer.hidden_dim
        output_dim = classifier.out_features

        # Create mask for keeping units
        keep_mask = torch.ones(hidden_dim, dtype=torch.bool)
        keep_mask[units_to_prune] = False
        keep_indices = torch.where(keep_mask)[0]

        new_hidden_dim = len(keep_indices)

        # 1. Update hidden layer weights (input -> hidden)
        # Keep only non-pruned units
        new_weight = hidden_layer.weight.data[keep_indices, :]  # [new_hidden_dim, input_dim]
        new_bias = hidden_layer.bias.data[keep_indices]  # [new_hidden_dim]

        # 2. Update classifier weights (hidden -> output)
        # Absorb pruned units into bias: bias += mean_activation * weight
        old_classifier_weight = classifier.weight.data.clone()  # [output_dim, hidden_dim]
        old_classifier_bias = classifier.bias.data.clone()  # [output_dim]

        # Absorb pruned units
        for unit_idx in units_to_prune:
            contribution = mean_activations[unit_idx] * old_classifier_weight[:, unit_idx]
            old_classifier_bias += contribution

        # Keep only non-pruned weights
        new_classifier_weight = old_classifier_weight[:, keep_indices]  # [output_dim, new_hidden_dim]

        # 3. Update SOM prototypes
        # Remove pruned units from all SOM prototypes
        som = hidden_layer.som
        old_som_vectors = som.som_vectors.data.clone()  # [grid_size, grid_size, hidden_dim]
        new_som_vectors = old_som_vectors[:, :, keep_indices]  # [grid_size, grid_size, new_hidden_dim]

        # 4. Create new layers with updated dimensions
        # Update hidden layer
        hidden_layer.weight = nn.Parameter(new_weight)
        hidden_layer.bias = nn.Parameter(new_bias)
        hidden_layer.hidden_dim = new_hidden_dim

        # Update activation statistics buffers
        hidden_layer.activation_sum = hidden_layer.activation_sum[keep_indices]
        hidden_layer.activation_sum_sq = hidden_layer.activation_sum_sq[keep_indices]

        # Update SOM
        som.input_dim = new_hidden_dim
        som.som_vectors = nn.Parameter(new_som_vectors, requires_grad=False)

        # Update classifier
        classifier.in_features = new_hidden_dim
        classifier.weight = nn.Parameter(new_classifier_weight)
        classifier.bias = nn.Parameter(old_classifier_bias)

        print(f"Hidden layer dimension: {hidden_dim} -> {new_hidden_dim}")

        # Record pruning event
        self.pruning_history.append({
            'units_pruned': units_to_prune,
            'old_dim': hidden_dim,
            'new_dim': new_hidden_dim,
        })

        return True

    def check_and_prune(
        self,
        model,
        epoch: int
    ) -> bool:
        """
        Check activation statistics and prune if necessary.

        Args:
            model: BPSOMBertForSequenceClassification model
            epoch: Current epoch number

        Returns:
            True if pruning occurred
        """
        if not self.enabled:
            return False

        # Get activation statistics
        mean_act, std_act = model.bpsom_hidden.get_activation_statistics()

        # Identify prunable units
        units_to_prune = self.identify_prunable_units(mean_act, std_act)

        if units_to_prune:
            print(f"\nEpoch {epoch + 1}: Detected {len(units_to_prune)} inactive units")
            print(f"  Units: {units_to_prune}")
            print(f"  Std values: {[f'{std_act[i].item():.4f}' for i in units_to_prune]}")

            # Prune
            pruned = self.prune_units(model, units_to_prune, mean_act)

            if pruned:
                # Reset statistics after pruning
                model.bpsom_hidden.reset_activation_statistics()

            return pruned

        return False

    def get_pruning_summary(self) -> dict:
        """
        Get summary of pruning events.

        Returns:
            Dictionary with pruning statistics
        """
        if not self.pruning_history:
            return {
                'total_events': 0,
                'total_units_pruned': 0,
                'initial_dim': None,
                'final_dim': None,
            }

        total_pruned = sum(len(event['units_pruned']) for event in self.pruning_history)
        initial_dim = self.pruning_history[0]['old_dim']
        final_dim = self.pruning_history[-1]['new_dim']

        return {
            'total_events': len(self.pruning_history),
            'total_units_pruned': total_pruned,
            'initial_dim': initial_dim,
            'final_dim': final_dim,
            'reduction_pct': (initial_dim - final_dim) / initial_dim * 100,
            'events': self.pruning_history,
        }


def add_pruning_to_trainer(trainer, pruner: UnitPruner):
    """
    Add pruning callback to trainer.

    Modifies the trainer's run_epoch method to check for pruning
    after evaluation.

    Args:
        trainer: BPSOMTrainer instance
        pruner: UnitPruner instance
    """
    original_run_epoch = trainer.run_epoch

    def run_epoch_with_pruning():
        # Run normal epoch
        metrics = original_run_epoch()

        # Check for pruning after evaluation
        pruned = pruner.check_and_prune(trainer.model, trainer.current_epoch)

        if pruned:
            # Pruning occurred, may affect convergence
            # Reset early stopping counter (optional)
            # trainer.epochs_without_improvement = 0
            pass

        return metrics

    trainer.run_epoch = run_epoch_with_pruning
    trainer.pruner = pruner

    return trainer
