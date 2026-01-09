"""
BP-SOM Training Loop

Implements training logic with SOM updates, parameter scheduling,
and statistics tracking following the C implementation.

Based on complete_epoch in bpsom.cc:231-288
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict, List
from tqdm import tqdm
import numpy as np


class BPSOMTrainer:
    """
    Trainer for BP-SOM BERT models.

    Manages:
    - Training/evaluation loops
    - SOM parameter scheduling
    - Statistics tracking
    - Early stopping
    - Checkpoint saving
    """

    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        early_stopping_patience: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_best_path: Optional[str] = None,
    ):
        """
        Args:
            model: BPSOMBertForSequenceClassification model
            train_dataloader: Training data loader
            eval_dataloader: Validation data loader
            test_dataloader: Test data loader (optional)
            learning_rate: Learning rate for BERT and classifier
            num_epochs: Maximum number of epochs
            warmup_steps: Warmup steps for learning rate schedule
            early_stopping_patience: Epochs without improvement before stopping
            device: Device to train on
            save_best_path: Path to save best model
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.save_best_path = save_best_path

        # Optimizer (for BERT and classifier, not SOM)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        # Learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training state
        self.current_epoch = 0
        self.best_eval_accuracy = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # History
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'test_accuracy': [],
            'som_stats': [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.

        Based on complete_epoch with lrn_yn=1 in bpsom.cc:231

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.model.set_epoch(self.current_epoch, self.max_epochs)

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # SOM statistics
        som_stats_accum = {
            'som_usage_pct': [],
            'avg_distance': [],
        }

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # Update progress bar
            current_accuracy = correct_predictions / total_predictions * 100
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_accuracy:.2f}%"
            })

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = correct_predictions / total_predictions * 100

        # Get SOM statistics
        som_stats = self.model.bpsom_hidden.som.get_statistics()

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'som_usage_pct': som_stats['som_usage_pct'],
            'som_avg_distance': som_stats['avg_distance'],
        }

        return metrics

    def eval_epoch(self, dataloader: DataLoader, desc: str = "Eval") -> Dict[str, float]:
        """
        Run evaluation on given dataloader.

        Based on complete_epoch with lrn_yn=0 in bpsom.cc:231

        Args:
            dataloader: Data loader to evaluate on
            desc: Description for progress bar

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [{desc}]")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                # Update progress bar
                current_accuracy = correct_predictions / total_predictions * 100
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{current_accuracy:.2f}%"
                })

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
        }

        return metrics

    def run_epoch(self) -> Dict[str, Dict[str, float]]:
        """
        Run complete epoch: train, update SOM labels, evaluate.

        Following the C implementation's epoch structure:
        1. Training pass
        2. Update SOM cell labels on training data
        3. Evaluate on dev set
        4. Optionally evaluate on test set

        Returns:
            Dictionary with train/eval/test metrics
        """
        # Reset SOM statistics
        self.model.bpsom_hidden.som.reset_statistics()
        self.model.bpsom_hidden.reset_activation_statistics()

        # Training
        train_metrics = self.train_epoch()

        # Update SOM cell labels (like C version's classlabeling epoch)
        print("Updating SOM cell labels...")
        self.model.update_som_labels(self.train_dataloader, self.device)

        # Evaluation on dev set
        eval_metrics = self.eval_epoch(self.eval_dataloader, desc="Dev")

        # Test set (if available)
        test_metrics = None
        if self.test_dataloader is not None:
            test_metrics = self.eval_epoch(self.test_dataloader, desc="Test")

        # Compile all metrics
        all_metrics = {
            'train': train_metrics,
            'eval': eval_metrics,
        }
        if test_metrics is not None:
            all_metrics['test'] = test_metrics

        return all_metrics

    def check_early_stopping(self, eval_accuracy: float) -> bool:
        """
        Check early stopping criteria.

        Args:
            eval_accuracy: Current evaluation accuracy

        Returns:
            True if should stop training
        """
        if eval_accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = eval_accuracy
            self.best_epoch = self.current_epoch
            self.epochs_without_improvement = 0

            # Save best model
            if self.save_best_path is not None:
                print(f"Saving best model (acc: {eval_accuracy:.2f}%) to {self.save_best_path}")
                torch.save(self.model.state_dict(), self.save_best_path)

            return False
        else:
            self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                print(f"Best accuracy: {self.best_eval_accuracy:.2f}% at epoch {self.best_epoch + 1}")
                return True

            return False

    def train(self) -> Dict[str, List]:
        """
        Main training loop.

        Following the C implementation's main loop in bpsom.cc:195-228

        Returns:
            Training history
        """
        print(f"\nStarting BP-SOM training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print("-" * 80)

        self.max_epochs = self.num_epochs

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Run epoch
            metrics = self.run_epoch()

            # Log metrics
            train_metrics = metrics['train']
            eval_metrics = metrics['eval']

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, "
                  f"SOM Usage: {train_metrics['som_usage_pct']:.1f}%")
            print(f"  Dev   - Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.2f}%")

            if 'test' in metrics:
                test_metrics = metrics['test']
                print(f"  Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%")
                self.history['test_accuracy'].append(test_metrics['accuracy'])

            # Save to history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['eval_loss'].append(eval_metrics['loss'])
            self.history['eval_accuracy'].append(eval_metrics['accuracy'])
            self.history['som_stats'].append({
                'som_usage_pct': train_metrics['som_usage_pct'],
                'som_avg_distance': train_metrics['som_avg_distance'],
            })

            # Early stopping check
            if self.check_early_stopping(eval_metrics['accuracy']):
                break

            print("-" * 80)

        # Final summary
        print("\nTraining complete!")
        print(f"Best dev accuracy: {self.best_eval_accuracy:.2f}% at epoch {self.best_epoch + 1}")

        # Load best model and evaluate on test
        if self.save_best_path is not None and self.test_dataloader is not None:
            print(f"\nLoading best model from {self.save_best_path}")
            self.model.load_state_dict(torch.load(self.save_best_path))
            final_test_metrics = self.eval_epoch(self.test_dataloader, desc="Final Test")
            print(f"Final test accuracy: {final_test_metrics['accuracy']:.2f}%")
            self.history['final_test_accuracy'] = final_test_metrics['accuracy']

        return self.history


class BaselineBertTrainer:
    """
    Baseline BERT trainer without BP-SOM for comparison.

    Simpler training loop without SOM components.
    """

    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        early_stopping_patience: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_best_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.save_best_path = save_best_path

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        # Scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # State
        self.current_epoch = 0
        self.best_eval_accuracy = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # History
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'test_accuracy': [],
        }

    def train_epoch(self):
        """Training epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total*100:.2f}%"
            })

        return {
            'loss': total_loss / len(self.train_dataloader),
            'accuracy': correct / total * 100
        }

    def eval_epoch(self, dataloader, desc="Eval"):
        """Evaluation epoch."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [{desc}]"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total * 100
        }

    def train(self):
        """Main training loop."""
        print(f"\nStarting baseline BERT training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print("-" * 80)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            eval_metrics = self.eval_epoch(self.eval_dataloader, "Dev")

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Dev   - Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.2f}%")

            if self.test_dataloader:
                test_metrics = self.eval_epoch(self.test_dataloader, "Test")
                print(f"  Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%")
                self.history['test_accuracy'].append(test_metrics['accuracy'])

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['eval_loss'].append(eval_metrics['loss'])
            self.history['eval_accuracy'].append(eval_metrics['accuracy'])

            # Early stopping
            if eval_metrics['accuracy'] > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_metrics['accuracy']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0

                if self.save_best_path:
                    torch.save(self.model.state_dict(), self.save_best_path)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            print("-" * 80)

        print(f"\nBest dev accuracy: {self.best_eval_accuracy:.2f}% at epoch {self.best_epoch + 1}")

        if self.save_best_path and self.test_dataloader:
            self.model.load_state_dict(torch.load(self.save_best_path))
            final_test = self.eval_epoch(self.test_dataloader, "Final Test")
            print(f"Final test accuracy: {final_test['accuracy']:.2f}%")
            self.history['final_test_accuracy'] = final_test['accuracy']

        return self.history
