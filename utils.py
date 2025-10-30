import torch
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.colors as colors
from IPython.display import display, HTML


"""
Scaling Law Experiment: TN vs Regular Transformer
Implements both tensor network (bilinear + quadratic attention) and
regular transformer (MLP + softmax attention) with matched parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from einops import rearrange, einsum
import math
from dataclasses import dataclass
from typing import Optional, Literal
import time
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.amp import autocast, GradScaler


# ============================================================================
# Muon Optimizer from modded-nanogpt
# ============================================================================

def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration for orthogonalization used in Muon."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer from modded-nanogpt
    Memory efficient, ~1.5x better sample efficiency than Adam
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                # Orthogonalize momentum buffer
                if len(grad.shape) >= 2:
                    grad_2d = buf.view(buf.shape[0], -1)
                    orthogonal_grad = zeroth_power_via_newtonschulz5(grad_2d, steps=ns_steps)
                    buf = orthogonal_grad.view_as(buf)

                p.data.add_(buf, alpha=-lr)


# ============================================================================
# Shared Components (RoPE, RMSNorm, Masking)
# ============================================================================

class Rotary(nn.Module):
    """Rotary positional embeddings"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache()

    def _build_cache(self):
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, :, None, :]
        sin_cached = emb.sin()[None, :, None, :]
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len]
        sin = self.sin_cached[:, :seq_len]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.stack(
            [-x_odd, x_even],
            dim=-1
        ).flatten(-2)

        return x * cos + x_rotated * sin


class Mask(nn.Module):
    """Masking for attention patterns"""
    def __init__(self, n_ctx, mask_type='causal'):
        super().__init__()
        self.n_ctx = n_ctx
        self.mask_type = mask_type

        if mask_type == 'causal':
            mask = torch.tril(torch.ones(n_ctx, n_ctx))
        else:  # no mask
            mask = torch.ones(n_ctx, n_ctx)

        self.register_buffer('mask', mask)

    def forward(self, scores):
        seq_len = scores.shape[-1]
        mask = self.mask[:seq_len, :seq_len]
        return scores * mask


# ============================================================================
# Unified Architecture Components with Control Flags
# ============================================================================

class UnifiedAttention(nn.Module):
    """Unified attention mechanism with squaring_attn flag"""
    def __init__(self, d_model: int, n_head: int, n_ctx: int,
                 mask: str = 'causal', scale: int = 1,
                 norm: bool = True, bias: bool = True,
                 squaring_attn: bool = False) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        self.squaring_attn = squaring_attn

        self.rotary = Rotary(self.d_head, n_ctx)
        self.norm = nn.RMSNorm(self.d_head) if norm else nn.Identity()

        # Causal mask (used differently for squared vs softmax)
        if mask == 'causal':
            # For softmax: binary mask
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
            )
        else:
            self.causal_mask = None

        # Initialize QKV projections
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)

        # Zero-initialize output projection (muP-like)
        self.o = nn.Linear(d_model, d_model, bias=False)
        init.zeros_(self.o.weight)

    def forward(self, x, attention_mask=None):
        q, k, v = [rearrange(op(x), '... (n_head d_head) -> ... n_head d_head',
                            n_head=self.n_head) for op in [self.q, self.k, self.v]]

        # Apply rotary embeddings and normalization
        q, k = self.rotary(self.norm(q)), self.rotary(self.norm(k))

        # Compute attention scores
        scores = einsum(q, k, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")

        if self.squaring_attn:
            # Quadratic scoring: scale then square
            scores = scores / self.d_head
            # Apply causal mask before squaring
            seq_len = scores.shape[-1]
            if self.causal_mask is not None:
                mask = self.causal_mask[:, :, :seq_len, :seq_len]
                scores = scores * mask
            pattern = scores.square()
        else:
            # Softmax scoring: scale then softmax
            scores = scores / math.sqrt(self.d_head)
            # Apply causal mask before softmax
            seq_len = scores.shape[-1]
            if self.causal_mask is not None:
                scores = scores.masked_fill(
                    self.causal_mask[:, :, :seq_len, :seq_len] == 0,
                    float('-inf')
                )
            pattern = F.softmax(scores, dim=-1)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]
            pattern = pattern * attn_mask

        # Aggregate values
        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... seq_q n_head d_head")
        z = rearrange(z, '... seq n_head d_head -> ... seq (n_head d_head)')

        return x + self.o(z) * self.scale


class UnifiedFFN(nn.Module):
    """Unified feed-forward network with bilinear flag

    Both variants use the same architecture (2 up-projections + 1 down):
    - bilinear=False: SwiGLU-style with swish activation on one branch
    - bilinear=True: Pure bilinear (no activation on either branch)

    This ensures identical parameter counts for fair comparison.
    """
    def __init__(self, d_model, d_hidden=None, bias=True, bilinear=False):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        self.bilinear = bilinear

        # Both variants have two up-projections (same param count)
        self.gate = nn.Linear(d_model, d_hidden, bias=bias)
        self.up = nn.Linear(d_model, d_hidden, bias=bias)

        # Output projection (same for both)
        self.down = nn.Linear(d_hidden, d_model, bias=bias)

        # Zero-initialize down projection for stability
        init.zeros_(self.down.weight)
        if bias:
            init.zeros_(self.down.bias)

    def forward(self, x):
        if self.bilinear:
            # Bilinear: element-wise product of two projections (no activation)
            hidden = self.gate(x) * self.up(x)
        else:
            # SwiGLU-style: swish activation on gate branch
            hidden = F.silu(self.gate(x)) * self.up(x)

        return self.down(hidden)


# ============================================================================
# Transformer Blocks
# ============================================================================

class UnifiedTransformerBlock(nn.Module):
    """Unified transformer block with squaring_attn and bilinear flags"""
    def __init__(self, d_model, n_head, n_ctx, dropout=0.0,
                 squaring_attn=False, bilinear=False):
        super().__init__()
        self.norm_attn = nn.RMSNorm(d_model)
        self.attn = UnifiedAttention(d_model, n_head, n_ctx, squaring_attn=squaring_attn)
        self.norm_ffn = nn.RMSNorm(d_model)
        self.ffn = UnifiedFFN(d_model, bilinear=bilinear)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        x = self.attn(self.norm_attn(x), attention_mask)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


# ============================================================================
# Model Configurations
# ============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = None
    d_model: int = 128
    n_ctx: int = 256
    n_head: int = 4
    dropout: float = 0.1
    n_layers: int = 2
    # Control flags for architecture variants
    squaring_attn: bool = False  # If True: square attention scores, else: softmax
    bilinear: bool = False  # If True: bilinear FFN (no activation), else: GELU MLP


class ToyTransformer(nn.Module):
    """
    Unified transformer architecture controlled by squaring_attn and bilinear flags
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Build layers using unified components
        layers = []
        for _ in range(config.n_layers):
            layers.append(UnifiedTransformerBlock(
                config.d_model,
                config.n_head,
                config.n_ctx,
                config.dropout,
                squaring_attn=config.squaring_attn,
                bilinear=config.bilinear
            ))

        self.layers = nn.ModuleList(layers)

        # Output head
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        """Forward pass with loss calculation"""
        original_x = x
        x = self.embed(x)
        x = self.dropout(x)

        # Forward through layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output projection
        logits = self.head(x)

        # Calculate loss
        targets = original_x[:, 1:]
        logit_predictions = logits[:, :-1, :]

        if attention_mask is not None:
            target_mask = attention_mask[:, 1:]
            loss = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            loss = (loss * target_mask.reshape(-1)).sum() / target_mask.sum()
        else:
            loss = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================================
# Data Loading
# ============================================================================

class StreamingTextDataset:
    """Streaming dataset that tokenizes text on-the-fly"""
    def __init__(self, dataset_name='HuggingFaceFW/fineweb', split='train',
                 tokenizer_name='gpt2', seq_length=1024, subset='sample-10BT',
                 validation_ratio=0.001, seed=42):
        from transformers import AutoTokenizer
        from datasets import load_dataset
        import hashlib

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_length = seq_length
        self.validation_ratio = validation_ratio
        self.is_validation = (split == 'validation')

        # Load streaming dataset
        if 'fineweb' in dataset_name.lower():
            actual_split = 'train'
        elif 'simplestories' in dataset_name.lower() and split == 'validation':
            actual_split = 'test'
        else:
            actual_split = split

        if subset:
            self.dataset = load_dataset(dataset_name, subset, split=actual_split, streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, split=actual_split, streaming=True)

        # Hash-based train/val split for datasets with only train split
        if 'fineweb' in dataset_name.lower() or actual_split == 'train':
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)

            def should_include(example):
                text = example.get('text', example.get('content', ''))
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                is_val_sample = (hash_val % int(1/validation_ratio)) == 0
                return is_val_sample == self.is_validation

            self.dataset = self.dataset.filter(should_include)

        # Create iterator
        self.iterator = iter(self.dataset)
        self.token_buffer = []

    def get_batch(self, batch_size, device='cuda'):
        """Get a batch of tokenized sequences"""
        batch_tokens = []

        while len(batch_tokens) < batch_size:
            # Refill token buffer if needed
            while len(self.token_buffer) < self.seq_length + 1:
                try:
                    sample = next(self.iterator)
                    text = sample.get('text', sample.get('content', sample.get('story', '')))

                    if not text:
                        print(f"\n=== DATASET KEY ERROR ===")
                        print(f"Available keys: {list(sample.keys())}")
                        import sys
                        sys.exit(1)

                    tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=False)
                    self.token_buffer.extend(tokens)
                except StopIteration:
                    self.iterator = iter(self.dataset)
                    if len(self.token_buffer) == 0:
                        self.token_buffer = [self.tokenizer.eos_token_id] * (self.seq_length + 1)
                        break

            # Extract sequence from buffer
            if len(self.token_buffer) >= self.seq_length + 1:
                seq = self.token_buffer[:self.seq_length + 1]
                batch_tokens.append(seq)
                self.token_buffer = self.token_buffer[self.seq_length:]

        # Convert to tensors
        batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        x = batch[:, :-1]  # Input sequences
        y = batch[:, 1:]   # Target sequences (shifted by 1)

        return x, y


# ============================================================================
# Training Infrastructure
# ============================================================================

class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Setup optimizer (Muon)
        self.optimizer = Muon(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum
        )
        self.scaler = GradScaler()

        self.iteration = 0

    def get_lr(self):
        """Cosine learning rate schedule with warmup"""
        if self.iteration < self.config.warmup_iters:
            return self.config.learning_rate * self.iteration / self.config.warmup_iters
        if self.iteration > self.config.lr_decay_iters:
            return self.config.min_lr
        decay_ratio = (self.iteration - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def train_step(self, input_ids, attention_mask):
        """Single training step"""
        lr = self.config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.zero_grad()

        # Forward pass with AMP
        with autocast(device_type="cuda"):
            _, loss = self.model(input_ids, attention_mask)

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.iteration += 1

        return loss.item(), lr

    def evaluate(self, val_dataset, max_batches=50):
        """Evaluate model on validation set"""
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i in range(max_batches):
                x, y = val_dataset.get_batch(self.config.batch_size, self.device)
                _, loss = self.model(x)
                losses.append(loss.item())
        self.model.train()
        return np.mean(losses)


@dataclass
class TrainingConfig:
    # Model
    model_config: ModelConfig = None

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-3
    momentum: float = 0.95
    min_lr: float = 3e-4
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    max_iters: int = 10000
    grad_clip: float = 1.0

    # Logging
    eval_interval: int = 100
    log_interval: int = 10


# ============================================================================
# Parameter Counting Utility
# ============================================================================

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def compare_architectures(config):
    """Compare parameter counts between TN and regular architectures"""
    tn_config = ModelConfig(**{**config.__dict__, 'model_type': config.model_type.replace('regular', 'tn')})
    regular_config = ModelConfig(**{**config.__dict__, 'model_type': config.model_type.replace('tn', 'regular')})

    tn_model = ToyTransformer(tn_config)
    regular_model = ToyTransformer(regular_config)

    tn_params = count_parameters(tn_model)
    regular_params = count_parameters(regular_model)

    print(f"TN Model: {tn_params['total']:,} parameters")
    print(f"Regular Model: {regular_params['total']:,} parameters")
    print(f"Difference: {abs(tn_params['total'] - regular_params['total']):,} parameters")

    return tn_params, regular_params
