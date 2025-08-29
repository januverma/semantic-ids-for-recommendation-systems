import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Initialize with default max_len, but allow dynamic expansion
        self._create_pe(max_len)
    
    def _create_pe(self, max_len: int):
        """Create positional encoding matrix."""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) for batch_first=True
        seq_len = x.size(1)
        
        # Expand positional encoding if needed
        if seq_len > self.pe.size(0):
            self._create_pe(seq_len)
        
        # pe shape: (seq_len, 1, d_model) -> need (1, seq_len, d_model) for broadcasting
        pos_encoding = self.pe[:seq_len, :].transpose(0, 1).to(x.device)  # (1, seq_len, d_model)
        return x + pos_encoding


class TIGERSeq2SeqDataset(Dataset):
    """Dataset for TIGER sequence-to-sequence training."""
    
    def __init__(self, 
                 sequential_data_path: str,
                 semantic_ids_path: str,
                 max_seq_len: int = 50,
                 split: str = 'train'):
        """
        Args:
            sequential_data_path: Path to sequential_data.txt
            semantic_ids_path: Path to semantic_ids.json
            max_seq_len: Maximum sequence length in semantic tokens
            split: 'train', 'val', or 'test'
        """
        self.max_seq_len = max_seq_len
        self.split = split
        
        # Load semantic IDs
        with open(semantic_ids_path, 'r') as f:
            data = json.load(f)
        
        self.semantic_ids = {int(k): v for k, v in data['semantic_ids'].items()}
        self.codebook_size = 256  # From KMeans codebook size
        self.num_levels = 4  # Number of quantization levels
        self.vocab_size = self.num_levels * self.codebook_size  # 1024
        
        # Load and process sequential data
        self.sequences = self._load_sequences(sequential_data_path)
        
        # Create semantic token sequences with leave-one-out splits
        self.data = self._prepare_sequences()
    
    def _load_sequences(self, sequential_data_path: str) -> List[List[int]]:
        """Load user interaction sequences."""
        sequences = []
        with open(sequential_data_path, 'r') as f:
            for line in f:
                items = [int(x) for x in line.strip().split()[1:]]  # Skip user_id
                if len(items) >= 3:  # Need at least 3 items for train/val/test split
                    sequences.append(items)
        return sequences
    
    def _prepare_sequences(self) -> List[Tuple[List[int], List[int]]]:
        """Convert item sequences to semantic token sequences with splits."""
        data = []
        
        for item_seq in self.sequences:
            # Fixed data splits to prevent leakage
            if self.split == 'train':
                # Training: everything before val (all except last 2 items)
                items = item_seq[:-2]  # All except last two
            elif self.split == 'val':
                # Validation: everything except last 2 items, predict second-to-last
                items = item_seq[:-2]  # All except last two
                target_item = item_seq[-2]  # Second to last
            else:  # test
                # Test: everything except last item, predict last
                items = item_seq[:-1]  # All except last one
                target_item = item_seq[-1]  # Last item
            
            if len(items) < 1:
                continue
                
            # Convert items to level-aware flattened semantic tokens
            semantic_tokens = []
            for item_id in items:
                if item_id in self.semantic_ids:
                    # Use all 4 levels with level-aware tokenization
                    semantic_id = self.semantic_ids[item_id]  # All 4 levels
                    for level, code in enumerate(semantic_id):
                        token = level * self.codebook_size + code
                        semantic_tokens.append(token)
            
            if len(semantic_tokens) == 0:
                continue
                
            if self.split == 'train':
                # For training, predict next item's semantic ID
                for i in range(4, len(semantic_tokens), 4):  # Step by 4 tokens per item
                    input_tokens = semantic_tokens[:i]
                    target_tokens = semantic_tokens[i:i+4] if i+4 <= len(semantic_tokens) else None
                    
                    if target_tokens and len(target_tokens) == 4:
                        data.append((input_tokens, target_tokens))
            else:
                # For val/test, predict target item's semantic ID with level-aware tokenization
                if target_item in self.semantic_ids:
                    semantic_id = self.semantic_ids[target_item]  # All 4 levels
                    target_tokens = []
                    for level, code in enumerate(semantic_id):
                        token = level * self.codebook_size + code
                        target_tokens.append(token)
                    data.append((semantic_tokens, target_tokens))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_tokens, target_tokens = self.data[idx]
        
        # For training: truncate to exactly 20 items (80 semantic tokens)
        # For val/test: use full sequences without truncation
        if self.split == 'train':
            max_training_tokens = 20 * 4  # 20 items × 4 tokens per item = 80 tokens
            if len(input_tokens) > max_training_tokens:
                # Keep the most recent 20 items
                input_tokens = input_tokens[-max_training_tokens:]
        
        # Convert to tensors
        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        target_tensor = torch.tensor(target_tokens, dtype=torch.long)
        
        return input_tensor, target_tensor


class TIGERModel(nn.Module):
    """TIGER: Transformer Index for GEnerative Recommenders."""
    
    def __init__(self,
                 vocab_size: int = 1024,  # num_levels (4) * codebook_size (256) = 1024
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 200):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.max_seq_len = max_seq_len
        
        # Transformer encoder (bidirectional for input history)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder (autoregressive for next item prediction)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_key_padding_mask: Mask for source padding
            tgt_key_padding_mask: Mask for target padding
            
        Returns:
            Output logits (batch_size, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        src_emb = self.dropout(self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model)))
        
        # Encode source sequence
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Generate causal mask for decoder
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Decode target sequence
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to vocabulary
        return self.output_proj(output)
    
    def generate(self, 
                 src: torch.Tensor,
                 max_length: int = 3,
                 beam_size: int = 5,
                 src_key_padding_mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Generate sequences using beam search (single sample).
        
        Args:
            src: Source sequence (1, src_len)
            max_length: Maximum generation length (3 for semantic ID)
            beam_size: Beam search width
            
        Returns:
            List of generated sequences
        """
        return self.generate_batch(src, max_length, beam_size, src_key_padding_mask)[0]

    def generate_batch(self, 
                       src: torch.Tensor,
                       max_length: int = 3,
                       beam_size: int = 5,
                       temperature: float = 1.0,
                       src_key_padding_mask: Optional[torch.Tensor] = None) -> List[List[List[int]]]:
        """
        Generate sequences using vectorized beam search for a batch of samples.
        
        Args:
            src: Source sequences (batch_size, src_len)
            max_length: Maximum generation length (3 for semantic ID)
            beam_size: Beam search width
            src_key_padding_mask: Padding mask for source sequences
            
        Returns:
            List of lists of generated sequences for each sample in batch
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode all source sequences at once
        src_emb = self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Expand memory and masks for beam search
        # Shape: (batch_size * beam_size, src_len, d_model)
        expanded_memory = memory.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, memory.size(1), memory.size(2))
        expanded_src_mask = None
        if src_key_padding_mask is not None:
            expanded_src_mask = src_key_padding_mask.unsqueeze(1).repeat(1, beam_size, 1).view(-1, src_key_padding_mask.size(1))
        
        # Initialize beams: (batch_size * beam_size, current_length)
        beam_tokens = torch.zeros(batch_size * beam_size, 0, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size * beam_size, device=device)
        
        
        for step in range(max_length):
            # Current sequence length
            current_length = beam_tokens.size(1)
            
            # Prepare decoder input
            if current_length == 0:
                tgt = torch.zeros(batch_size * beam_size, 1, dtype=torch.long, device=device)
            else:
                # Pad with zeros at the beginning for decoder input
                tgt = torch.cat([
                    torch.zeros(batch_size * beam_size, 1, dtype=torch.long, device=device),
                    beam_tokens
                ], dim=1)
            
            # Generate embeddings and masks
            tgt_emb = self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model))
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Decoder forward pass for all beams at once
            output = self.decoder(tgt_emb, expanded_memory, tgt_mask=tgt_mask,
                                memory_key_padding_mask=expanded_src_mask)
            logits = self.output_proj(output[:, -1, :])  # Last position
            
            # Apply temperature scaling
            logits = logits / temperature
                
            log_probs = F.log_softmax(logits, dim=-1)  # (batch_size * beam_size, vocab_size)
            
            # Reshape to separate batch and beam dimensions
            log_probs = log_probs.view(batch_size, beam_size, -1)  # (batch_size, beam_size, vocab_size)
            
            # Add current beam scores
            beam_scores_reshaped = beam_scores.view(batch_size, beam_size)  # (batch_size, beam_size)
            scores = log_probs + beam_scores_reshaped.unsqueeze(-1)  # (batch_size, beam_size, vocab_size)
            
            # Sample diverse candidates instead of always taking top-k
            scores_flat = scores.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
            
            # For diversity, sample from top candidates rather than always taking the best
            top_k_for_sampling = min(beam_size * 5, scores_flat.size(-1))
            
            top_scores = torch.zeros(batch_size, beam_size, device=scores_flat.device)
            top_indices = torch.zeros(batch_size, beam_size, dtype=torch.long, device=scores_flat.device)
            
            for b in range(batch_size):
                # Get top candidates
                cand_scores, cand_indices = torch.topk(scores_flat[b], top_k_for_sampling)
                
                # Convert to probabilities for sampling
                probs = F.softmax(cand_scores, dim=0)
                
                # Sample beam_size indices without replacement
                sampled_idx = torch.multinomial(probs, beam_size, replacement=False)
                
                # Get the actual scores and indices
                for k in range(beam_size):
                    idx = sampled_idx[k]
                    top_scores[b, k] = cand_scores[idx]
                    top_indices[b, k] = cand_indices[idx]
            
            # Convert flat indices back to beam and token indices
            beam_indices = top_indices // log_probs.size(-1)  # Which beam each candidate came from
            token_indices = top_indices % log_probs.size(-1)   # Which token each candidate selected
            
            # Update beam tokens and scores
            new_beam_tokens = []
            new_beam_scores = []
            
            for b in range(batch_size):
                batch_new_tokens = []
                batch_new_scores = []
                
                for k in range(beam_size):
                    # Get the beam index and token for this candidate
                    beam_idx = beam_indices[b, k].item()
                    token_idx = token_indices[b, k].item()
                    
                    # Get the original beam tokens for this batch and beam
                    original_beam_start = b * beam_size + beam_idx
                    if current_length > 0:
                        original_tokens = beam_tokens[original_beam_start].tolist()
                    else:
                        original_tokens = []
                    
                    # Add new token
                    new_tokens = original_tokens + [token_idx]
                    new_score = top_scores[b, k].item()
                    
                    batch_new_tokens.append(new_tokens)
                    batch_new_scores.append(new_score)
                
                new_beam_tokens.extend(batch_new_tokens)
                new_beam_scores.extend(batch_new_scores)
            
            # Update beam_tokens and beam_scores
            max_len = max(len(tokens) for tokens in new_beam_tokens)
            beam_tokens = torch.zeros(batch_size * beam_size, max_len, dtype=torch.long, device=device)
            
            for i, tokens in enumerate(new_beam_tokens):
                if len(tokens) > 0:
                    beam_tokens[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            
            beam_scores = torch.tensor(new_beam_scores, device=device)
        
        # Convert results back to list format
        batch_results = []
        for b in range(batch_size):
            sample_results = []
            for k in range(beam_size):
                beam_idx = b * beam_size + k
                tokens = beam_tokens[beam_idx].tolist()
                # Remove padding zeros
                tokens = [t for t in tokens if t != 0] if any(t != 0 for t in tokens) else tokens
                sample_results.append(tokens)
            batch_results.append(sample_results)
        
        return batch_results


class SemanticIDMapper:
    """Maps between semantic IDs and item IDs with caching for performance."""
    
    def __init__(self, semantic_ids_path: str):
        with open(semantic_ids_path, 'r') as f:
            data = json.load(f)
        
        self.item_to_semantic = {int(k): v for k, v in data['semantic_ids'].items()}  # Use all 4 levels
        self.codebook_size = 256
        self.num_levels = 4
        
        # Create reverse mapping: level-aware tokens -> item_id
        self.tokens_to_item = {}
        for item_id, semantic_id in self.item_to_semantic.items():
            # Convert to level-aware tokens
            tokens = tuple(level * self.codebook_size + code for level, code in enumerate(semantic_id))
            if tokens not in self.tokens_to_item:
                self.tokens_to_item[tokens] = []
            self.tokens_to_item[tokens].append(item_id)
        
        # Cache for semantic_to_items lookups
        self._cache = {}
    
    def semantic_to_items(self, semantic_tokens: List[int]) -> List[int]:
        """Map 4 level-aware semantic tokens to corresponding item IDs with caching."""
        if len(semantic_tokens) != 4:
            return [-1]
        
        tokens_tuple = tuple(semantic_tokens)
        
        # Check cache first
        if tokens_tuple in self._cache:
            return self._cache[tokens_tuple]
        
        # Compute and cache result
        if tokens_tuple in self.tokens_to_item:
            result = self.tokens_to_item[tokens_tuple]
        else:
            result = [-1]
        
        self._cache[tokens_tuple] = result
        return result
    
    def item_to_semantic_tokens(self, item_id: int) -> List[int]:
        """Map item ID to level-aware semantic tokens."""
        if item_id in self.item_to_semantic:
            semantic_id = self.item_to_semantic[item_id]
            return [level * self.codebook_size + code for level, code in enumerate(semantic_id)]
        else:
            return [-1, -1, -1, -1]


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    input_seqs, target_seqs = zip(*batch)
    
    # Convert input sequences to lists if they're tensors
    input_seqs = [seq.tolist() if torch.is_tensor(seq) else seq for seq in input_seqs]
    
    # Pad input sequences
    max_input_len = max(len(seq) for seq in input_seqs)
    padded_inputs = []
    input_masks = []
    
    for seq in input_seqs:
        padding = [0] * (max_input_len - len(seq))
        padded_inputs.append(seq + padding)
        input_masks.append([False] * len(seq) + [True] * len(padding))
    
    # Stack target sequences (all should be length 4)
    padded_inputs = torch.tensor(padded_inputs, dtype=torch.long)
    target_seqs = torch.stack(target_seqs)
    input_masks = torch.tensor(input_masks, dtype=torch.bool)
    
    return padded_inputs, target_seqs, input_masks