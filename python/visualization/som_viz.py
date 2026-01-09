"""
SOM Visualization Tools

Provides visualization utilities for analyzing SOM organization and behavior.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List


class SOMVisualizer:
    """
    Visualizer for Self-Organizing Map analysis.

    Creates various plots to understand SOM organization:
    - Class label heatmaps
    - Reliability heatmaps
    - U-matrix (distance between neighbors)
    - Activation patterns
    """

    def __init__(self, save_dir: Optional[str] = None):
        """
        Args:
            save_dir: Directory to save plots (if None, displays instead)
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

    def plot_cell_labels(
        self,
        som,
        class_names: Optional[List[str]] = None,
        epoch: Optional[int] = None,
        title: str = "SOM Cell Labels"
    ):
        """
        Plot heatmap of class labels assigned to each SOM cell.

        Based on dump_som_labeling in som.h:100-145

        Args:
            som: SelfOrganizingMap instance
            class_names: Optional list of class names
            epoch: Optional epoch number for title
            title: Plot title
        """
        grid_size = som.grid_size
        labels = som.cell_labels.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 9))

        # Create heatmap
        sns.heatmap(
            labels,
            annot=True,
            fmt='d',
            cmap='tab20',
            square=True,
            cbar_kws={'label': 'Class Label'},
            ax=ax
        )

        if epoch is not None:
            title = f"{title} (Epoch {epoch})"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('SOM X', fontsize=12)
        ax.set_ylabel('SOM Y', fontsize=12)

        self._save_or_show(f'som_labels_epoch_{epoch}.png' if epoch is not None else 'som_labels.png')
        plt.close()

    def plot_cell_reliability(
        self,
        som,
        epoch: Optional[int] = None,
        title: str = "SOM Cell Reliability"
    ):
        """
        Plot heatmap of reliability (purity) of each SOM cell.

        Args:
            som: SelfOrganizingMap instance
            epoch: Optional epoch number
            title: Plot title
        """
        reliability = som.cell_reliability.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 9))

        sns.heatmap(
            reliability,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            vmin=0,
            vmax=100,
            square=True,
            cbar_kws={'label': 'Reliability (%)'},
            ax=ax
        )

        if epoch is not None:
            title = f"{title} (Epoch {epoch})"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('SOM X', fontsize=12)
        ax.set_ylabel('SOM Y', fontsize=12)

        self._save_or_show(f'som_reliability_epoch_{epoch}.png' if epoch is not None else 'som_reliability.png')
        plt.close()

    def plot_combined_som_info(
        self,
        som,
        class_names: Optional[List[str]] = None,
        epoch: Optional[int] = None
    ):
        """
        Plot combined view: labels + reliability in one figure.

        Args:
            som: SelfOrganizingMap instance
            class_names: Optional class names
            epoch: Optional epoch number
        """
        labels = som.cell_labels.cpu().numpy()
        reliability = som.cell_reliability.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Labels
        sns.heatmap(
            labels,
            annot=True,
            fmt='d',
            cmap='tab20',
            square=True,
            cbar_kws={'label': 'Class'},
            ax=axes[0]
        )
        axes[0].set_title('Class Labels', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')

        # Reliability
        sns.heatmap(
            reliability,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            vmin=0,
            vmax=100,
            square=True,
            cbar_kws={'label': 'Reliability (%)'},
            ax=axes[1]
        )
        axes[1].set_title('Reliability', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')

        if epoch is not None:
            fig.suptitle(f'SOM Organization at Epoch {epoch}', fontsize=16, fontweight='bold')

        plt.tight_layout()
        self._save_or_show(f'som_combined_epoch_{epoch}.png' if epoch is not None else 'som_combined.png')
        plt.close()

    def plot_u_matrix(
        self,
        som,
        epoch: Optional[int] = None,
        title: str = "U-Matrix (Prototype Distances)"
    ):
        """
        Plot U-matrix showing distances between neighboring prototypes.

        Useful for visualizing cluster boundaries.

        Args:
            som: SelfOrganizingMap instance
            epoch: Optional epoch number
            title: Plot title
        """
        grid_size = som.grid_size
        som_vectors = som.som_vectors.cpu().numpy()

        # Compute average distance to neighbors for each cell
        u_matrix = np.zeros((grid_size, grid_size))

        for x in range(grid_size):
            for y in range(grid_size):
                distances = []
                # Check 4-connected neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        dist = np.linalg.norm(som_vectors[x, y] - som_vectors[nx, ny])
                        distances.append(dist)

                u_matrix[x, y] = np.mean(distances) if distances else 0

        fig, ax = plt.subplots(figsize=(10, 9))

        sns.heatmap(
            u_matrix,
            annot=False,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Avg Distance to Neighbors'},
            ax=ax
        )

        if epoch is not None:
            title = f"{title} (Epoch {epoch})"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('SOM X', fontsize=12)
        ax.set_ylabel('SOM Y', fontsize=12)

        self._save_or_show(f'som_umatrix_epoch_{epoch}.png' if epoch is not None else 'som_umatrix.png')
        plt.close()

    def plot_training_history(
        self,
        history: dict,
        title: str = "Training History"
    ):
        """
        Plot training curves: loss and accuracy.

        Args:
            history: Training history dictionary
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(history['eval_loss'], label='Dev', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(history['train_accuracy'], label='Train', marker='o')
        axes[0, 1].plot(history['eval_accuracy'], label='Dev', marker='s')
        if 'test_accuracy' in history and history['test_accuracy']:
            axes[0, 1].plot(history['test_accuracy'], label='Test', marker='^')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # SOM usage percentage
        if 'som_stats' in history:
            som_usage = [stat['som_usage_pct'] for stat in history['som_stats']]
            axes[1, 0].plot(som_usage, marker='o', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('SOM Usage (%)')
            axes[1, 0].set_title('SOM Error Usage')
            axes[1, 0].grid(True)

            # SOM average distance
            som_dist = [stat['som_avg_distance'] for stat in history['som_stats']]
            axes[1, 1].plot(som_dist, marker='o', color='purple')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Average Distance')
            axes[1, 1].set_title('SOM Average Distance')
            axes[1, 1].grid(True)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        self._save_or_show('training_history.png')
        plt.close()

    def plot_comparison(
        self,
        baseline_history: dict,
        bpsom_history: dict,
        metric: str = 'eval_accuracy',
        title: Optional[str] = None
    ):
        """
        Plot comparison between baseline and BP-SOM.

        Args:
            baseline_history: Baseline training history
            bpsom_history: BP-SOM training history
            metric: Metric to compare
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(baseline_history[metric], label='Baseline BERT', marker='o', linewidth=2)
        ax.plot(bpsom_history[metric], label='BP-SOM BERT', marker='s', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)

        if title is None:
            title = f'Comparison: {metric.replace("_", " ").title()}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show(f'comparison_{metric}.png')
        plt.close()

    def _save_or_show(self, filename: str):
        """Save figure or show if no save directory."""
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
        else:
            plt.show()
