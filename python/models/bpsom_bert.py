"""
BP-SOM BERT Model for Sequence Classification

Integrates BERT encoder with BP-SOM hidden layer for enhanced learning
through combined supervised and unsupervised signals.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple

from .som_layer import SelfOrganizingMap


class BPSOMFunction(torch.autograd.Function):
    """
    Custom autograd function to inject SOM-based gradients during backpropagation.

    Forward: Standard linear transformation
    Backward: Combine BP gradient with SOM error
    """

    @staticmethod
    def forward(ctx, x, weight, bias, som_error, som_error_weight):
        """
        Args:
            x: Input activations [batch_size, hidden_size]
            weight: Linear layer weight
            bias: Linear layer bias
            som_error: SOM-derived error [batch_size, hidden_size] or None
            som_error_weight: Weight for SOM error (α)
        """
        ctx.save_for_backward(x, weight, bias, som_error)
        ctx.som_error_weight = som_error_weight

        # Standard linear transformation
        output = torch.nn.functional.linear(x, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Combine standard BP gradient with SOM error.

        Based on bp.h:111-119:
        error[l][i] = (bp_error_use * bp_error) + (SOM_ERROR_USE * som_error)
        """
        x, weight, bias, som_error = ctx.saved_tensors
        som_error_weight = ctx.som_error_weight

        # Standard gradients
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output.mm(weight)

            # Inject SOM error if available
            if som_error is not None:
                # Combine gradients: (1 - α) * bp_grad + α * som_grad
                # Note: SOM error is already scaled by 0.01 * reliability in SOM layer
                bp_weight = 1.0 - som_error_weight
                grad_x = bp_weight * grad_x + som_error_weight * som_error

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(x)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_x, grad_weight, grad_bias, None, None


class BPSOMHiddenLayer(nn.Module):
    """
    Hidden layer with attached SOM for BP-SOM learning.

    Combines:
    1. Standard feedforward transformation
    2. SOM-based clustering
    3. Gradient modification during backprop
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        som_grid_size: int = 20,
        som_lr_max: float = 0.20,
        som_lr_min: float = 0.05,
        som_context_max: int = 2,
        som_context_min: int = 0,
        reliability_threshold: float = 0.95,
        som_error_weight: float = 0.25,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear transformation
        self.weight = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Activation function (sigmoid as in C implementation)
        self.activation = nn.Sigmoid()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Attached SOM
        self.som = SelfOrganizingMap(
            input_dim=hidden_dim,
            grid_size=som_grid_size,
            num_classes=num_classes,
            som_lr_max=som_lr_max,
            som_lr_min=som_lr_min,
            som_context_max=som_context_max,
            som_context_min=som_context_min,
            reliability_threshold=reliability_threshold,
            som_error_weight=som_error_weight,
        )

        self.som_error_weight = som_error_weight

        # For tracking activations (used by pruning)
        self.register_buffer('activation_sum', torch.zeros(hidden_dim))
        self.register_buffer('activation_sum_sq', torch.zeros(hidden_dim))
        self.register_buffer('activation_count', torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epochs: int = 100,
        training: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with SOM processing.

        Args:
            x: Input [batch_size, input_dim]
            labels: Class labels [batch_size] (for SOM training)
            epoch: Current epoch
            max_epochs: Total epochs
            training: Training mode flag

        Returns:
            activations: [batch_size, hidden_dim] output activations
            stats: SOM statistics
        """
        # Linear transformation
        z = torch.nn.functional.linear(x, self.weight, self.bias)

        # Activation
        activations = self.activation(z)

        # Track activation statistics (for pruning)
        if training:
            with torch.no_grad():
                self.activation_sum += activations.sum(dim=0)
                self.activation_sum_sq += (activations ** 2).sum(dim=0)
                self.activation_count += activations.size(0)

        # Process through SOM
        som_errors, som_stats = self.som.forward(
            activations,
            labels=labels,
            epoch=epoch,
            max_epochs=max_epochs,
            training=training
        )

        # Inject SOM error via gradient hook
        if training and labels is not None and som_errors is not None:
            # Register a hook to modify gradients during backward pass
            def som_gradient_hook(grad):
                # Combine BP gradient with SOM error
                # BP gradient comes from upstream, SOM error is computed from SOM
                bp_weight = 1.0 - self.som_error_weight
                # grad is gradient w.r.t. activations
                # som_errors is the SOM error term
                combined_grad = bp_weight * grad + self.som_error_weight * som_errors
                return combined_grad

            activations.register_hook(som_gradient_hook)

        # Dropout
        activations = self.dropout(activations)

        return activations, som_stats

    def get_activation_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mean and std of activations (for pruning).

        Returns:
            mean: [hidden_dim] mean activation per unit
            std: [hidden_dim] std activation per unit
        """
        if self.activation_count > 0:
            mean = self.activation_sum / self.activation_count
            variance = (self.activation_sum_sq / self.activation_count) - (mean ** 2)
            std = torch.sqrt(torch.clamp(variance, min=0))
        else:
            mean = torch.zeros_like(self.activation_sum)
            std = torch.zeros_like(self.activation_sum)

        return mean, std

    def reset_activation_statistics(self):
        """Reset activation tracking."""
        self.activation_sum.zero_()
        self.activation_sum_sq.zero_()
        self.activation_count.zero_()


class BPSOMBertForSequenceClassification(BertPreTrainedModel):
    """
    BERT model with BP-SOM hidden layer for sequence classification.

    Architecture:
        BERT Encoder → [CLS] → BP-SOM Hidden Layer → Classifier
    """

    def __init__(self, config, bpsom_config=None):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        # BERT encoder
        self.bert = BertModel(config)

        # BP-SOM configuration
        if bpsom_config is None:
            bpsom_config = {
                'hidden_size': 128,
                'dropout': 0.1,
                'som_grid_size': 20,
                'som_lr_max': 0.20,
                'som_lr_min': 0.05,
                'som_context_max': 2,
                'som_context_min': 0,
                'reliability_threshold': 0.95,
                'som_error_weight': 0.25,
            }

        # BP-SOM hidden layer
        self.bpsom_hidden = BPSOMHiddenLayer(
            input_dim=config.hidden_size,  # BERT output dimension
            hidden_dim=bpsom_config.get('hidden_size', 128),
            num_classes=config.num_labels,
            dropout=bpsom_config.get('dropout', 0.1),
            som_grid_size=bpsom_config.get('som_grid_size', 20),
            som_lr_max=bpsom_config.get('som_lr_max', 0.20),
            som_lr_min=bpsom_config.get('som_lr_min', 0.05),
            som_context_max=bpsom_config.get('som_context_max', 2),
            som_context_min=bpsom_config.get('som_context_min', 0),
            reliability_threshold=bpsom_config.get('reliability_threshold', 0.95),
            som_error_weight=bpsom_config.get('som_error_weight', 0.25),
        )

        # Classification head
        self.classifier = nn.Linear(bpsom_config.get('hidden_size', 128), config.num_labels)

        # Initialize weights
        self.post_init()

        # Training state
        self.current_epoch = 0
        self.max_epochs = 100

    def set_epoch(self, epoch: int, max_epochs: int):
        """Set current epoch for SOM parameter scheduling."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        """
        Forward pass through BERT + BP-SOM + Classifier.

        Args:
            input_ids: [batch_size, seq_len] input token IDs
            attention_mask: [batch_size, seq_len] attention mask
            token_type_ids: [batch_size, seq_len] token type IDs
            labels: [batch_size] class labels
            return_dict: Whether to return ModelOutput

        Returns:
            SequenceClassifierOutput with loss, logits, and additional info
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # BERT encoding
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # Get [CLS] representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # BP-SOM hidden layer
        hidden_output, som_stats = self.bpsom_hidden(
            cls_output,
            labels=labels,
            epoch=self.current_epoch,
            max_epochs=self.max_epochs,
            training=self.training
        )

        # Classification
        logits = self.classifier(hidden_output)

        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def update_som_labels(self, dataloader, device):
        """
        Update SOM cell labels based on training data.
        Should be called after each epoch.

        Args:
            dataloader: Training data loader
            device: Device to use
        """
        all_activations = []
        all_labels = []

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward through BERT
                outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
                cls_output = outputs.last_hidden_state[:, 0, :]

                # Forward through BP-SOM hidden (without training)
                z = torch.nn.functional.linear(cls_output, self.bpsom_hidden.weight, self.bpsom_hidden.bias)
                activations = self.bpsom_hidden.activation(z)

                all_activations.append(activations.cpu())
                all_labels.append(labels.cpu())

        # Concatenate all
        all_activations = torch.cat(all_activations, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Update SOM labels
        self.bpsom_hidden.som.update_cell_labels(all_activations, all_labels)

        self.train()
