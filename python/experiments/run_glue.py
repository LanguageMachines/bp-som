"""
Run GLUE Benchmark Experiments

Main entry point for running BP-SOM experiments on GLUE tasks.
Supports both baseline BERT and BP-SOM BERT configurations.
"""

import argparse
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig
)
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.bpsom_bert import BPSOMBertForSequenceClassification
from training.trainer import BPSOMTrainer, BaselineBertTrainer
from training.pruning import UnitPruner, add_pruning_to_trainer
from visualization.som_viz import SOMVisualizer
from visualization.logger import BPSOMLogger


class GLUEDataProcessor:
    """
    Process GLUE datasets for BERT models.
    """

    TASK_CONFIGS = {
        'sst2': {
            'dataset_name': 'glue',
            'dataset_config': 'sst2',
            'text_fields': ('sentence',),
            'num_labels': 2,
            'label_names': ['negative', 'positive'],
        },
        'mrpc': {
            'dataset_name': 'glue',
            'dataset_config': 'mrpc',
            'text_fields': ('sentence1', 'sentence2'),
            'num_labels': 2,
            'label_names': ['not_equivalent', 'equivalent'],
        },
        'cola': {
            'dataset_name': 'glue',
            'dataset_config': 'cola',
            'text_fields': ('sentence',),
            'num_labels': 2,
            'label_names': ['unacceptable', 'acceptable'],
        },
    }

    def __init__(self, task: str, tokenizer, max_length: int = 128):
        """
        Args:
            task: GLUE task name (e.g., 'sst2')
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Task {task} not supported. Choose from: {list(self.TASK_CONFIGS.keys())}")

        self.task = task
        self.config = self.TASK_CONFIGS[task]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self):
        """Load GLUE dataset."""
        dataset = load_dataset(
            self.config['dataset_name'],
            self.config['dataset_config']
        )
        return dataset

    def preprocess_function(self, examples):
        """Tokenize examples."""
        text_fields = self.config['text_fields']

        if len(text_fields) == 1:
            # Single sentence task
            texts = examples[text_fields[0]]
            tokenized = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        else:
            # Sentence pair task
            texts1 = examples[text_fields[0]]
            texts2 = examples[text_fields[1]]
            tokenized = self.tokenizer(
                texts1,
                texts2,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

        tokenized['labels'] = torch.tensor(examples['label'])
        return tokenized

    def get_dataloaders(self, batch_size: int = 32):
        """
        Get train/validation/test dataloaders.

        Args:
            batch_size: Batch size

        Returns:
            train_loader, eval_loader, test_loader
        """
        dataset = self.load_data()

        # Process datasets
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
                'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
                'labels': torch.tensor([item['labels'] for item in batch])
            }

        # Tokenize
        train_data = []
        for example in dataset['train']:
            tokenized = self.preprocess_function({k: [example[k]] for k in example.keys()})
            train_data.append({k: v[0] for k, v in tokenized.items()})

        eval_data = []
        for example in dataset['validation']:
            tokenized = self.preprocess_function({k: [example[k]] for k in example.keys()})
            eval_data.append({k: v[0] for k, v in tokenized.items()})

        # Test set (if available)
        test_data = []
        if 'test' in dataset:
            for example in dataset['test']:
                tokenized = self.preprocess_function({k: [example[k]] for k in example.keys()})
                test_data.append({k: v[0] for k, v in tokenized.items()})

        # Create dataloaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if test_data else None

        return train_loader, eval_loader, test_loader


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_baseline_experiment(
    config: dict,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path
):
    """
    Run baseline BERT experiment.

    Args:
        config: Configuration dictionary
        train_loader: Training dataloader
        eval_loader: Validation dataloader
        test_loader: Test dataloader
        output_dir: Output directory

    Returns:
        Training history
    """
    print("\n" + "=" * 80)
    print("BASELINE BERT EXPERIMENT")
    print("=" * 80)

    # Initialize model
    bert_config = BertConfig.from_pretrained(config['model']['name'])
    bert_config.num_labels = config['task']['num_labels']

    model = BertForSequenceClassification.from_pretrained(
        config['model']['name'],
        config=bert_config
    )

    # Initialize trainer
    trainer = BaselineBertTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        test_dataloader=test_loader,
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['epochs'],
        warmup_steps=config['training']['warmup_steps'],
        early_stopping_patience=config['training'].get('early_stopping_patience', 3),
        save_best_path=str(output_dir / 'best_baseline_model.pt')
    )

    # Train
    history = trainer.train()

    # Save history
    import json
    with open(output_dir / 'baseline_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return history


def run_bpsom_experiment(
    config: dict,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path
):
    """
    Run BP-SOM BERT experiment.

    Args:
        config: Configuration dictionary
        train_loader: Training dataloader
        eval_loader: Validation dataloader
        test_loader: Test dataloader
        output_dir: Output directory

    Returns:
        Training history, SOM visualizer, pruner
    """
    print("\n" + "=" * 80)
    print("BP-SOM BERT EXPERIMENT")
    print("=" * 80)

    # Initialize model
    bert_config = BertConfig.from_pretrained(config['model']['name'])
    bert_config.num_labels = config['task']['num_labels']

    model = BPSOMBertForSequenceClassification.from_pretrained(
        config['model']['name'],
        config=bert_config,
        bpsom_config=config.get('bpsom', {})
    )

    # Initialize logger
    logger = BPSOMLogger(
        log_dir=str(output_dir / 'logs'),
        experiment_name='bpsom_experiment',
        config=config
    )

    # Initialize trainer
    trainer = BPSOMTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        test_dataloader=test_loader,
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['epochs'],
        warmup_steps=config['training']['warmup_steps'],
        early_stopping_patience=config['training'].get('early_stopping_patience', 3),
        save_best_path=str(output_dir / 'best_bpsom_model.pt')
    )

    # Initialize pruner if enabled
    pruner = None
    if config.get('pruning', {}).get('enabled', False):
        print("\nPruning enabled")
        pruner = UnitPruner(
            prune_threshold=config['pruning']['threshold'],
            enabled=True
        )
        trainer = add_pruning_to_trainer(trainer, pruner)

    # Train
    history = trainer.train()

    # Save history
    import json
    with open(output_dir / 'bpsom_history.json', 'w') as f:
        json.dump({k: v for k, v in history.items() if not isinstance(v, dict)}, f, indent=2)

    # Visualize
    visualizer = SOMVisualizer(save_dir=str(output_dir / 'visualizations'))

    # Final SOM visualization
    print("\nCreating visualizations...")
    visualizer.plot_combined_som_info(
        model.bpsom_hidden.som,
        class_names=config['task'].get('label_names'),
        epoch=trainer.best_epoch
    )
    visualizer.plot_training_history(history)

    # Log final summary
    pruning_summary = pruner.get_pruning_summary() if pruner else None
    logger.log_final_summary(
        best_epoch=trainer.best_epoch,
        best_dev_acc=trainer.best_eval_accuracy,
        final_test_acc=history.get('final_test_accuracy'),
        pruning_summary=pruning_summary
    )
    logger.close()

    return history, visualizer, pruner


def main():
    parser = argparse.ArgumentParser(description='Run GLUE experiments with BP-SOM')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--task', type=str, default='sst2', help='GLUE task (sst2, mrpc, cola)')
    parser.add_argument('--mode', type=str, default='bpsom', choices=['baseline', 'bpsom', 'both'],
                       help='Experiment mode')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['task'] = {
        'name': args.task,
        **GLUEDataProcessor.TASK_CONFIGS[args.task]
    }

    # Setup output directory
    output_dir = Path(args.output_dir) / args.task / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTask: {args.task}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])

    # Load data
    print("\nLoading dataset...")
    data_processor = GLUEDataProcessor(args.task, tokenizer)
    train_loader, eval_loader, test_loader = data_processor.get_dataloaders(batch_size=args.batch_size)

    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Dev examples: {len(eval_loader.dataset)}")
    if test_loader:
        print(f"Test examples: {len(test_loader.dataset)}")

    # Run experiments
    if args.mode == 'baseline':
        history = run_baseline_experiment(config, train_loader, eval_loader, test_loader, output_dir)

    elif args.mode == 'bpsom':
        history, visualizer, pruner = run_bpsom_experiment(
            config, train_loader, eval_loader, test_loader, output_dir
        )

    elif args.mode == 'both':
        # Run baseline
        baseline_output = output_dir.parent / 'baseline'
        baseline_output.mkdir(parents=True, exist_ok=True)
        baseline_history = run_baseline_experiment(
            config, train_loader, eval_loader, test_loader, baseline_output
        )

        # Run BP-SOM
        bpsom_output = output_dir.parent / 'bpsom'
        bpsom_output.mkdir(parents=True, exist_ok=True)
        bpsom_history, visualizer, pruner = run_bpsom_experiment(
            config, train_loader, eval_loader, test_loader, bpsom_output
        )

        # Comparison plots
        print("\nCreating comparison plots...")
        comp_visualizer = SOMVisualizer(save_dir=str(output_dir.parent / 'comparison'))
        comp_visualizer.plot_comparison(baseline_history, bpsom_history, 'eval_accuracy')
        comp_visualizer.plot_comparison(baseline_history, bpsom_history, 'train_loss')

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Baseline - Best Dev Acc: {max(baseline_history['eval_accuracy']):.2f}%")
        print(f"BP-SOM   - Best Dev Acc: {max(bpsom_history['eval_accuracy']):.2f}%")
        if 'final_test_accuracy' in baseline_history and 'final_test_accuracy' in bpsom_history:
            print(f"Baseline - Test Acc: {baseline_history['final_test_accuracy']:.2f}%")
            print(f"BP-SOM   - Test Acc: {bpsom_history['final_test_accuracy']:.2f}%")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
