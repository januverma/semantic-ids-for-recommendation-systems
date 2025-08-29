import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse

from src.tiger_model import TIGERModel, TIGERSeq2SeqDataset, collate_fn, SemanticIDMapper


def calculate_metrics(predictions: List[List[int]], ground_truth: int, k_values: List[int]) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of prediction lists for each k
        ground_truth: True item ID
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    for i, k in enumerate(k_values):
        pred_k = predictions[i] if i < len(predictions) else []
        
        # Hit Rate @ K
        hit = 1.0 if ground_truth in pred_k else 0.0
        metrics[f'HitRate@{k}'] = hit
        
        # Recall @ K (same as hit rate for single item)
        metrics[f'Recall@{k}'] = hit
        
        # NDCG @ K
        if ground_truth in pred_k:
            rank = pred_k.index(ground_truth) + 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        metrics[f'NDCG@{k}'] = ndcg
    
    return metrics


class TIGERTrainer:
    """Trainer for TIGER model."""
    
    def __init__(self, 
                 model: TIGERModel,
                 semantic_mapper: SemanticIDMapper,
                 device: torch.device):
        self.model = model.to(device)
        self.semantic_mapper = semantic_mapper
        self.device = device
        
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for input_seqs, target_seqs, input_masks in tqdm(train_loader, desc="Training"):
            input_seqs = input_seqs.to(self.device)
            target_seqs = target_seqs.to(self.device)
            input_masks = input_masks.to(self.device)
            
            optimizer.zero_grad()
            
            # Prepare decoder input (shift target right)
            batch_size = target_seqs.size(0)
            seq_len = target_seqs.size(1)
            decoder_input = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
            decoder_input[:, 1:] = target_seqs[:, :-1]  # Shift right
            
            # Forward pass
            outputs = self.model(input_seqs, decoder_input, src_key_padding_mask=input_masks)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_seqs.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, eval_loader: DataLoader, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        all_metrics = {f'HitRate@{k}': [] for k in k_values}
        all_metrics.update({f'Recall@{k}': [] for k in k_values})
        all_metrics.update({f'NDCG@{k}': [] for k in k_values})
        
        with torch.no_grad():
            for input_seqs, target_seqs, input_masks in tqdm(eval_loader, desc="Evaluating"):
                input_seqs = input_seqs.to(self.device)
                target_seqs = target_seqs.to(self.device)
                input_masks = input_masks.to(self.device)
                
                # Generate predictions for entire batch at once
                max_k = max(k_values)
                batch_generated_sequences = self.model.generate_batch(
                    input_seqs, 
                    max_length=4, 
                    beam_size=max_k,
                    temperature=1.5,
                    src_key_padding_mask=input_masks
                )
                
                batch_size = input_seqs.size(0)
                
                # Pre-compute ground truth items for the entire batch
                batch_ground_truth = []
                for i in range(batch_size):
                    target_semantic = target_seqs[i].cpu().tolist()
                    ground_truth_items = self.semantic_mapper.semantic_to_items(target_semantic)
                    ground_truth = ground_truth_items[0] if len(ground_truth_items) > 0 and ground_truth_items[0] != -1 else -1
                    batch_ground_truth.append(ground_truth)
                
                # Process predictions for the entire batch
                for i in range(batch_size):
                    ground_truth = batch_ground_truth[i]
                    if ground_truth == -1:
                        continue
                    
                    generated_sequences = batch_generated_sequences[i]
                    
                    # Map all generated sequences to items at once
                    all_pred_items = []
                    for seq in generated_sequences:
                        items = self.semantic_mapper.semantic_to_items(seq)
                        all_pred_items.extend([item for item in items if item != -1])
                    
                    # Create predictions for each k using vectorized operations
                    predictions_by_k = []
                    seen_global = set()
                    unique_items = []
                    
                    # Remove duplicates once for all k values
                    for item in all_pred_items:
                        if item not in seen_global:
                            unique_items.append(item)
                            seen_global.add(item)
                    
                    # Generate predictions for each k
                    for k in k_values:
                        pred_k = unique_items[:k]
                        predictions_by_k.append(pred_k)
                    
                    # Calculate metrics for this sample
                    sample_metrics = calculate_metrics(predictions_by_k, ground_truth, k_values)
                    
                    # Add to overall metrics
                    for metric, value in sample_metrics.items():
                        all_metrics[metric].append(value)
        
        # Average metrics
        avg_metrics = {}
        for metric, values in all_metrics.items():
            avg_metrics[metric] = np.mean(values) if len(values) > 0 else 0.0
        
        return avg_metrics
    
    def print_validation_examples(self, eval_loader: DataLoader, num_examples: int = 3):
        """Print a few validation examples to see what the model is generating."""
        self.model.eval()
        
        print(f"\n{'='*60}")
        print("VALIDATION EXAMPLES")
        print(f"{'='*60}")
        
        with torch.no_grad():
            for batch_idx, (input_seqs, target_seqs, input_masks) in enumerate(eval_loader):
                if batch_idx >= 1:  # Only process first batch
                    break
                    
                input_seqs = input_seqs.to(self.device)
                target_seqs = target_seqs.to(self.device)
                input_masks = input_masks.to(self.device)
                
                batch_size = min(num_examples, input_seqs.size(0))
                
                for i in range(batch_size):
                    print(f"\n--- Example {i+1} ---")
                    
                    # Get input sequence and convert to items
                    input_semantic = input_seqs[i].cpu().tolist()
                    # Remove padding (zeros)
                    input_semantic = [token for token in input_semantic if token != 0]
                    
                    # Convert semantic tokens to items (groups of 4)
                    input_items = []
                    for j in range(0, len(input_semantic), 4):
                        if j + 3 < len(input_semantic):
                            semantic_group = input_semantic[j:j+4]
                            items = self.semantic_mapper.semantic_to_items(semantic_group)
                            if items and items[0] != -1:
                                input_items.append(items[0])
                    
                    # Get ground truth
                    target_semantic = target_seqs[i].cpu().tolist()
                    ground_truth_items = self.semantic_mapper.semantic_to_items(target_semantic)
                    ground_truth = ground_truth_items[0] if ground_truth_items and ground_truth_items[0] != -1 else -1
                    
                    print(f"Input sequence ({len(input_items)} items): {input_items}")
                    print(f"Ground truth item: {ground_truth}")
                    
                    # Generate predictions
                    input_seq_single = input_seqs[i:i+1]  # Single sample
                    input_mask_single = input_masks[i:i+1] if input_masks is not None else None
                    
                    generated_sequences = self.model.generate_batch(
                        input_seq_single, 
                        max_length=4, 
                        beam_size=5,
                        temperature=1.5,
                        src_key_padding_mask=input_mask_single
                    )[0]  # Get first (and only) sample
                    
                    print(f"Generated sequences:")
                    for j, seq in enumerate(generated_sequences[:3]):  # Show top 3
                        items = self.semantic_mapper.semantic_to_items(seq)
                        predicted_item = items[0] if items and items[0] != -1 else -1
                        hit_marker = "✓" if predicted_item == ground_truth else "✗"
                        print(f"  {j+1}. Tokens: {seq} → Item: {predicted_item} {hit_marker}")
                
                break  # Only process one batch
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate TIGER model')
    parser.add_argument('--data_dir', type=str, default='beauty', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=150, help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='tiger_checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    sequential_data_path = os.path.join(args.data_dir, 'sequential_data.txt')
    semantic_ids_path = os.path.join(args.data_dir, 'semantic_ids.json')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TIGERSeq2SeqDataset(sequential_data_path, semantic_ids_path, args.max_seq_len, 'train')
    val_dataset = TIGERSeq2SeqDataset(sequential_data_path, semantic_ids_path, args.max_seq_len, 'val')
    test_dataset = TIGERSeq2SeqDataset(sequential_data_path, semantic_ids_path, args.max_seq_len, 'test')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("Creating model...")
    model = TIGERModel(
        vocab_size=1024,  # num_levels (4) * codebook_size (256) = 1024
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    )
    
    # Create semantic mapper
    semantic_mapper = SemanticIDMapper(semantic_ids_path)
    
    # Create trainer
    trainer = TIGERTrainer(model, semantic_mapper, device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    best_val_metric = 0.0
    k_values = [1, 3, 5, 10]
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = trainer.evaluate(val_loader, k_values)
        val_score = val_metrics['HitRate@5']  # Use HitRate@5 as main metric
        
        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Print validation examples periodically (first 3 epochs, then every 5 epochs)
        if epoch < 3 or (epoch + 1) % 5 == 0:
            trainer.print_validation_examples(val_loader, num_examples=2)
        
        # Save best model
        if val_score > best_val_metric:
            best_val_metric = val_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'vocab_size': 1024,  # Save vocab_size for inference
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"Saved new best model (HitRate@5: {val_score:.4f})")
        
        scheduler.step()
    
    # Test evaluation
    print("\nFinal test evaluation...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = trainer.evaluate(test_loader, k_values)
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save test results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nTraining completed! Best model saved to {args.save_dir}")


if __name__ == "__main__":
    main()