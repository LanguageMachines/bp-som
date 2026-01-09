# BP-SOM for Transformers

Python implementation of BP-SOM (Back-Propagation with Self-Organizing Maps) for BERT fine-tuning on GLUE benchmark tasks.

This implementation adapts the original BP-SOM algorithm (from the C implementation in `src/`) to modern Transformer architectures, specifically BERT-based models.

## Overview

**BP-SOM** combines supervised backpropagation learning with unsupervised clustering from Self-Organizing Maps (SOMs). During training:

1. **Standard backpropagation** computes error gradients from task loss
2. **SOM clustering** organizes hidden layer activations by class
3. **Combined learning** uses both signals: `error = (1 - α) * bp_error + α * som_error`
4. **Dynamic pruning** removes inactive units based on activation variance

## Architecture

```
Input Text
    ↓
BERT Encoder (bert-base-uncased)
    ↓
[CLS] token representation (768-dim)
    ↓
BP-SOM Hidden Layer (128 units)
    ├── Linear transformation + Sigmoid
    ├── SOM (20×20 grid) - continuously trained
    └── Gradient injection (BP + SOM errors)
    ↓
Classification Head
    ↓
Loss
```

## Installation

```bash
# From the bp-som directory
cd python

# Install dependencies
pip install -r ../requirements.txt
```

## Quick Start

### Run BP-SOM on SST-2 (Sentiment Analysis)

```bash
cd experiments

# BP-SOM experiment
python run_glue.py --config configs/bpsom.yaml --task sst2 --mode bpsom --output_dir ./output

# Baseline BERT for comparison
python run_glue.py --config configs/baseline.yaml --task sst2 --mode baseline --output_dir ./output

# Run both and compare
python run_glue.py --config configs/bpsom.yaml --task sst2 --mode both --output_dir ./output
```

### Supported GLUE Tasks

- **SST-2**: Sentiment analysis (binary classification)
- **MRPC**: Paraphrase detection
- **CoLA**: Linguistic acceptability

More tasks can be added easily by extending `GLUEDataProcessor` in `run_glue.py`.

## Configuration

### BP-SOM Configuration (`configs/bpsom.yaml`)

Key parameters (based on original C implementation):

```yaml
bpsom:
  hidden_size: 128              # BP-SOM hidden layer dimension
  som_grid_size: 20             # SOM grid is 20×20
  som_error_weight: 0.25        # α: weight of SOM error (0.25 = 25% SOM, 75% BP)
  som_lr_max: 0.20              # SOM learning rate (max)
  som_lr_min: 0.05              # SOM learning rate (min)
  som_context_max: 2            # Neighborhood radius (max)
  som_context_min: 0            # Neighborhood radius (min)
  reliability_threshold: 0.95   # Min reliability to use SOM error

pruning:
  enabled: true                 # Enable unit pruning
  threshold: 0.02               # Prune units with std < 0.02
```

## Project Structure

```
python/
├── models/
│   ├── som_layer.py          # Self-Organizing Map implementation
│   └── bpsom_bert.py         # BP-SOM BERT model
├── training/
│   ├── trainer.py            # Training loops (BP-SOM & baseline)
│   └── pruning.py            # Unit pruning logic
├── visualization/
│   ├── som_viz.py            # SOM visualization tools
│   └── logger.py             # Detailed logging
├── experiments/
│   ├── run_glue.py           # Main experiment runner
│   └── configs/
│       ├── baseline.yaml     # Baseline BERT config
│       └── bpsom.yaml        # BP-SOM config
└── utils/
    └── metrics.py
```

## Understanding the Output

### Training Progress

```
Epoch 1/10:
  Train - Loss: 0.4523, Acc: 78.32%, SOM Usage: 42.3%
  Dev   - Loss: 0.3891, Acc: 82.15%
  Test  - Loss: 0.3854, Acc: 81.90%

Pruning Event at Epoch 5:
  Units pruned: [7, 8, 23]
  Dimension: 128 -> 125
```

- **SOM Usage**: Percentage of examples where SOM error was reliable enough to use
- **Pruning**: Units with activation std < threshold are removed

### Visualizations

Generated in `output/<task>/<mode>/visualizations/`:

1. **`som_combined_epoch_X.png`**: SOM class labels + reliability heatmaps
2. **`som_umatrix_epoch_X.png`**: U-matrix showing cluster boundaries
3. **`training_history.png`**: Loss, accuracy, and SOM statistics over time
4. **`comparison_*.png`**: Baseline vs BP-SOM comparison (when using `--mode both`)

### Logs

- **`logs/bpsom_experiment.log`**: Detailed text log (similar to C version)
- **`logs/bpsom_experiment.json`**: Structured JSON log with all metrics
- **`bpsom_history.json`**: Training history for analysis

## Algorithm Details

### SOM Error Computation

Based on `bp.h:111-119` in the C implementation:

```python
# Find best matching SOM cell with same class label (partial winner)
som_error = 0.01 * reliability * (prototype - activation)

# Combine with BP error
total_error = (1 - α) * bp_error + α * som_error
```

The SOM error pulls activations toward class-specific prototype vectors, encouraging hidden units to specialize.

### SOM Update

Based on `som.h:147-159`:

```python
# Update BMU and neighborhood with distance-based learning rate
update_power = som_lr / (2 ** manhattan_distance)
prototype += update_power * (activation - prototype)
```

### Pruning

Based on `bp.h:304-352`:

- After each epoch, compute mean and std of each hidden unit's activations
- Units with `std < threshold` (default 0.02) are marked for pruning
- Pruned units are removed, and their contribution is absorbed into the bias

## Expected Results

On SST-2 sentiment analysis:

- **Baseline BERT**: ~91-93% dev accuracy
- **BP-SOM BERT**: Comparable accuracy with potentially:
  - More interpretable hidden representations (SOM organization)
  - Reduced model size (through pruning)
  - Different training dynamics

The main goal is to understand how SOM-guided learning affects:
1. Hidden unit specialization (via SOM visualizations)
2. Pruning patterns (which units become inactive)
3. Generalization performance

## Advanced Usage

### Custom SOM Parameters

Experiment with different SOM configurations:

```yaml
bpsom:
  som_grid_size: 30          # Larger grid for finer clustering
  som_error_weight: 0.5      # Equal BP and SOM influence
  reliability_threshold: 0.8  # More lenient reliability
```

### Disable Pruning

```yaml
pruning:
  enabled: false
```

### Different BERT Models

```yaml
model:
  name: "bert-large-uncased"  # Or roberta-base, etc.
```

## Research Questions

This implementation is designed to explore:

1. **Does BP-SOM improve BERT fine-tuning?**
   - Compare accuracy, training stability, generalization

2. **How does the SOM organize?**
   - Visualize class-specific clustering
   - Analyze reliability and organization quality

3. **What pruning patterns emerge?**
   - Which units become inactive?
   - How much can we reduce model size?
   - Does pruning affect performance?

4. **Optimal hyperparameters?**
   - SOM error weight (α)
   - Grid size
   - Pruning threshold

## Citation

Original BP-SOM papers:

- Weijters, A. (1995). The BP-SOM architecture and learning algorithm. *Neural Processing Letters*, 2:6, pp. 13-16.

- Weijters, A., Van den Bosch, A., and Van den Herik, H.J. (1997). Behavioural aspects of combining back-propagation and self-organizing maps. *Connection Science*, 9:3, pp. 253-252.

- Weijters, A., Van den Herik, H.J., Van den Bosch, A., and Postma, E. (1997). [Avoiding overfitting with BP-SOM.](https://www.ijcai.org/Proceedings/97-2/Papers/051.pdf) In *Proceedings of IJCAI-97*, pp. 1140-1145.

## License

GNU Public License version 3.0 (same as the C implementation)

## Acknowledgments

- Original BP-SOM concept and implementation by Ton Weijters and Antal van den Bosch
- This Python/Transformer adaptation builds on the ideas from the original C implementation in `src/`
