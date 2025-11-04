"""
Train bilinear transformer on GPU 1
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

import torch
import numpy as np
import wandb
import time
from utils import (
    ModelConfig, ToyTransformer, Trainer, TrainingConfig, count_parameters
)
from dataloader import distributed_data_generator

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
n_layers = 12
d_model = 64 * n_layers  # 768
n_ctx = 256  # Must be >= seq_length for rotary embeddings
n_head = 6
batch_size = 32  # Reduced for larger 12-layer model
seq_length = 256

# Calculate iterations needed for 1B tokens
tokens_per_batch = batch_size * seq_length  # 8,192 tokens
total_tokens = 1_000_000_000  # 1B tokens
max_iters = total_tokens // tokens_per_batch
print(f"Training for {max_iters} iterations to reach ~{total_tokens:,} tokens")
print(f"Tokens per iteration: {tokens_per_batch:,}")

# Setup binary dataloader
print("\nSetting up binary dataloader...")
train_data_gen = distributed_data_generator(
    filename_pattern="../modded-nanogpt/data/fineweb10B/fineweb_train_*.bin",
    num_tokens=tokens_per_batch,
    max_seq_len=seq_length,
    grad_accum_steps=1,
    align_to_bos=True
)


def get_batch_from_generator(data_gen, device='cuda'):
    """Get batch from binary dataloader generator"""
    inputs, targets, cum_lengths = next(data_gen)
    inputs = inputs[:batch_size * seq_length].view(batch_size, seq_length)
    targets = targets[:batch_size * seq_length].view(batch_size, seq_length)
    inputs = torch.clamp(inputs.long(), 0, 50256)
    targets = torch.clamp(targets.long(), 0, 50256)
    return inputs, targets


print(f"\n{'='*60}")
print(f"Training bilinear_transformer")
print(f"{'='*60}")

# Initialize WandB
wandb_run = wandb.init(
    project='scaling_tns',
    name=f'bilinear_transformer_d{d_model}_n{n_layers}_binary',
    config={
        'model_name': 'bilinear_transformer',
        'squaring_attn': True,
        'bilinear': True,
        'd_model': d_model,
        'n_ctx': n_ctx,
        'n_head': n_head,
        'n_layers': n_layers,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'total_tokens': total_tokens,
        'max_iters': max_iters,
        'learning_rate': 3e-3,
        'dataloader': 'binary',
    },
    reinit=True
)

# Create model config
model_config = ModelConfig(
    vocab_size=50257,  # GPT-2 tokenizer
    d_model=d_model,
    n_ctx=n_ctx,
    n_head=n_head,
    dropout=0.0,
    n_layers=n_layers,
    squaring_attn=True,
    bilinear=True
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
    log_interval=50
)

# Create trainer
trainer = Trainer(model, training_config, device='cuda')

# Training loop
losses = []
iterations = []
start_time = time.time()

for iter_num in range(max_iters):
    # Get batch from binary dataloader
    x, y = get_batch_from_generator(train_data_gen, device='cuda')

    # Training step
    loss, lr = trainer.train_step(x, attention_mask=None)

    # Log
    if iter_num % 50 == 0 or iter_num == max_iters - 1:
        losses.append(loss)
        iterations.append(iter_num * tokens_per_batch)
        elapsed_time = time.time() - start_time

        # Log to WandB
        wandb.log({
            "loss": loss,
            "lr": lr,
            "step": iter_num,
            "tokens": iter_num * tokens_per_batch,
            "time_elapsed": elapsed_time,
        })

        if iter_num % 200 == 0:
            print(f"Iter {iter_num}/{max_iters} | Loss: {loss:.4f} | LR: {lr:.6f} | Time: {elapsed_time:.1f}s | Tokens: {iter_num * tokens_per_batch / 1e6:.1f}M")

print(f"Final loss: {losses[-1]:.4f}")

# Log final metrics
wandb.log({
    "final_loss": losses[-1],
})

# Finish WandB run
wandb.finish()

print("\nDone!")
