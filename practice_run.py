"""
Practice script to compare transformer vs bilinear transformer
Runs both architectures on 10M tokens and plots loss curves
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    ModelConfig, ToyTransformer, StreamingTextDataset,
    Trainer, TrainingConfig, count_parameters
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
n_layers = 2
d_model = 64 * n_layers  # 128
n_ctx = 256
n_head = 4
batch_size = 64
seq_length = 256

# Calculate iterations needed for 10M tokens
tokens_per_batch = batch_size * seq_length
total_tokens = 10_000_000
max_iters = total_tokens // tokens_per_batch
print(f"Training for {max_iters} iterations to reach ~{total_tokens:,} tokens")
print(f"Tokens per iteration: {tokens_per_batch:,}")

# Setup datasets
print("\nLoading datasets...")
train_dataset = StreamingTextDataset(
    dataset_name='HuggingFaceFW/fineweb',
    split='train',
    tokenizer_name='gpt2',
    seq_length=seq_length,
    subset='sample-10BT'
)

# Training function
def train_model(model_name, squaring_attn, bilinear):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Create model config
    model_config = ModelConfig(
        vocab_size=50257,  # GPT-2 tokenizer
        d_model=d_model,
        n_ctx=n_ctx,
        n_head=n_head,
        dropout=0.0,
        n_layers=n_layers,
        squaring_attn=squaring_attn,
        bilinear=bilinear
    )

    # Create model
    model = ToyTransformer(model_config)
    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,}")

    # Create training config
    training_config = TrainingConfig(
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=3e-3,
        momentum=0.95,
        min_lr=3e-4,
        warmup_iters=100,
        lr_decay_iters=max_iters,
        max_iters=max_iters,
        grad_clip=1.0,
        eval_interval=max_iters + 1,  # Disable eval
        log_interval=10
    )

    # Create trainer
    trainer = Trainer(model, training_config, device='cuda')

    # Training loop
    losses = []
    iterations = []

    for iter_num in range(max_iters):
        # Get batch
        x, y = train_dataset.get_batch(batch_size, device='cuda')

        # Training step
        loss, lr = trainer.train_step(x, attention_mask=None)

        # Log
        if iter_num % 10 == 0 or iter_num == max_iters - 1:
            losses.append(loss)
            iterations.append(iter_num * tokens_per_batch)

            if iter_num % 50 == 0:
                print(f"Iter {iter_num}/{max_iters} | Loss: {loss:.4f} | LR: {lr:.6f}")

    print(f"Final loss: {losses[-1]:.4f}")

    return iterations, losses


# Train both models
print("\n" + "="*60)
print("TRAINING COMPARISON")
print("="*60)

# Regular transformer (softmax attention + SwiGLU FFN)
transformer_iters, transformer_losses = train_model(
    "transformer",
    squaring_attn=False,
    bilinear=False
)

# Bilinear transformer (squaring attention + bilinear FFN)
bilinear_iters, bilinear_losses = train_model(
    "bilinear transformer",
    squaring_attn=True,
    bilinear=True
)

# Plot results
print("\n" + "="*60)
print("PLOTTING RESULTS")
print("="*60)

plt.figure(figsize=(10, 6))
plt.plot(
    np.array(transformer_iters) / 1e6,
    transformer_losses,
    label='transformer',
    linewidth=2
)
plt.plot(
    np.array(bilinear_iters) / 1e6,
    bilinear_losses,
    label='bilinear transformer',
    linewidth=2
)
plt.xlabel('Tokens (millions)', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.title(f'Training Loss Comparison (d_model={d_model}, n_layers={n_layers})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
output_file = 'practice_loss_curves.png'
plt.savefig(output_file, dpi=150)
print(f"Plot saved to {output_file}")

plt.show()

print("\nDone!")
