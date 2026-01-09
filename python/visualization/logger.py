"""
Detailed Logging for BP-SOM Training

Provides comprehensive logging similar to the C implementation's log files.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import json
from datetime import datetime


class BPSOMLogger:
    """
    Detailed logger for BP-SOM training experiments.

    Logs similar information to the C version's log files:
    - Configuration parameters
    - Epoch-by-epoch progress
    - SOM organization metrics
    - Unit activation statistics
    - Pruning events
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict] = None
    ):
        """
        Args:
            log_dir: Directory for log files
            experiment_name: Name of experiment
            config: Configuration dictionary to log
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.json_file = self.log_dir / f"{experiment_name}.json"

        # Setup file logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_file, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Structured log data
        self.log_data = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': config or {},
            'epochs': [],
        }

        # Log header
        self._log_header(config)

    def _log_header(self, config: Optional[Dict]):
        """Log experiment header with configuration."""
        self.logger.info("=" * 80)
        self.logger.info(f"BP-SOM Training Log - {self.experiment_name}")
        self.logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

        if config:
            self.logger.info("\nConfiguration:")
            self.logger.info("-" * 40)
            for key, value in config.items():
                if isinstance(value, dict):
                    self.logger.info(f"{key}:")
                    for k, v in value.items():
                        self.logger.info(f"  {k}: {v}")
                else:
                    self.logger.info(f"{key}: {value}")
            self.logger.info("-" * 40)

    def log_epoch_start(self, epoch: int, max_epochs: int):
        """Log epoch start."""
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"EPOCH {epoch + 1}/{max_epochs}")
        self.logger.info(f"{'=' * 80}")

    def log_training_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = "train"
    ):
        """
        Log training/evaluation metrics.

        Args:
            epoch: Current epoch
            metrics: Metrics dictionary
            phase: 'train', 'eval', or 'test'
        """
        phase_name = {
            'train': 'TRAIN',
            'eval': 'DEV  ',
            'test': 'TEST ',
        }.get(phase, phase.upper())

        log_line = f"{phase_name} - Epoch {epoch + 1}: "
        log_line += f"Loss={metrics.get('loss', 0):.4f}, "
        log_line += f"Acc={metrics.get('accuracy', 0):.2f}%"

        if 'som_usage_pct' in metrics:
            log_line += f", SOM Usage={metrics['som_usage_pct']:.1f}%"
        if 'som_avg_distance' in metrics:
            log_line += f", SOM Dist={metrics['som_avg_distance']:.3f}"

        self.logger.info(log_line)

    def log_som_state(
        self,
        epoch: int,
        som,
        class_names: Optional[list] = None
    ):
        """
        Log detailed SOM state (like C version's dump_som_labeling).

        Args:
            epoch: Current epoch
            som: SelfOrganizingMap instance
            class_names: Optional class names
        """
        self.logger.info(f"\nSOM State at Epoch {epoch + 1}:")
        self.logger.info("-" * 80)

        grid_size = som.grid_size
        labels = som.cell_labels.cpu().numpy()
        reliability = som.cell_reliability.cpu().numpy()

        # Header row (X coordinates)
        header = "     "
        for x in range(grid_size):
            header += f"{x:4d}"
        self.logger.info(header)

        # Each row
        for y in range(grid_size):
            # Reliability row
            rel_row = f"{y:3d}  "
            for x in range(grid_size):
                rel_row += f"{int(reliability[x, y]):4d}"
            self.logger.info(rel_row)

            # Label row
            label_row = "     "
            for x in range(grid_size):
                label = int(labels[x, y])
                if class_names and label < len(class_names):
                    label_str = class_names[label][:3]  # First 3 chars
                else:
                    label_str = f"{label}"
                label_row += f"{label_str:>4s}"
            self.logger.info(label_row)

        self.logger.info("-" * 80)

    def log_activation_statistics(
        self,
        epoch: int,
        mean_activations,
        std_activations,
        prune_threshold: Optional[float] = None
    ):
        """
        Log unit activation statistics.

        Args:
            epoch: Current epoch
            mean_activations: Mean activation per unit
            std_activations: Std activation per unit
            prune_threshold: Threshold for pruning indicator
        """
        self.logger.info(f"\nUnit Activation Statistics (Epoch {epoch + 1}):")
        self.logger.info("-" * 80)

        hidden_dim = len(mean_activations)

        # Header
        header = "Unit:  "
        for i in range(hidden_dim):
            marker = "*" if prune_threshold and std_activations[i] <= prune_threshold else " "
            header += f"{i:6d}{marker}"
        self.logger.info(header)

        # Mean row
        mean_row = "Mean:  "
        for m in mean_activations:
            mean_row += f"{m:.5f} "
        self.logger.info(mean_row)

        # Std row
        std_row = "Std:   "
        for s in std_activations:
            std_row += f"{s:.5f} "
        self.logger.info(std_row)

        if prune_threshold:
            self.logger.info(f"\n* indicates std <= {prune_threshold} (prunable)")

        self.logger.info("-" * 80)

    def log_pruning_event(
        self,
        epoch: int,
        units_pruned: list,
        old_dim: int,
        new_dim: int
    ):
        """
        Log pruning event.

        Args:
            epoch: Current epoch
            units_pruned: List of pruned unit indices
            old_dim: Dimension before pruning
            new_dim: Dimension after pruning
        """
        self.logger.info(f"\nPRUNING EVENT at Epoch {epoch + 1}:")
        self.logger.info(f"  Units pruned: {units_pruned}")
        self.logger.info(f"  Dimension: {old_dim} -> {new_dim}")
        self.logger.info(f"  Reduction: {len(units_pruned)} units ({len(units_pruned)/old_dim*100:.1f}%)")

    def log_epoch_summary(
        self,
        epoch_data: Dict
    ):
        """
        Save epoch data to structured log.

        Args:
            epoch_data: Dictionary with epoch information
        """
        self.log_data['epochs'].append(epoch_data)

        # Save JSON log
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def log_final_summary(
        self,
        best_epoch: int,
        best_dev_acc: float,
        final_test_acc: Optional[float] = None,
        pruning_summary: Optional[Dict] = None
    ):
        """
        Log final training summary.

        Args:
            best_epoch: Best epoch number
            best_dev_acc: Best dev accuracy
            final_test_acc: Final test accuracy
            pruning_summary: Pruning statistics
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Best Dev Accuracy: {best_dev_acc:.2f}% (Epoch {best_epoch + 1})")

        if final_test_acc is not None:
            self.logger.info(f"Final Test Accuracy: {final_test_acc:.2f}%")

        if pruning_summary and pruning_summary['total_events'] > 0:
            self.logger.info(f"\nPruning Summary:")
            self.logger.info(f"  Total pruning events: {pruning_summary['total_events']}")
            self.logger.info(f"  Total units pruned: {pruning_summary['total_units_pruned']}")
            self.logger.info(f"  Initial dimension: {pruning_summary['initial_dim']}")
            self.logger.info(f"  Final dimension: {pruning_summary['final_dim']}")
            self.logger.info(f"  Reduction: {pruning_summary['reduction_pct']:.1f}%")

        end_time = datetime.now().isoformat()
        self.logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

        # Update structured log
        self.log_data['end_time'] = end_time
        self.log_data['best_epoch'] = best_epoch
        self.log_data['best_dev_accuracy'] = best_dev_acc
        if final_test_acc:
            self.log_data['final_test_accuracy'] = final_test_acc
        if pruning_summary:
            self.log_data['pruning_summary'] = pruning_summary

        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def close(self):
        """Close logger."""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
