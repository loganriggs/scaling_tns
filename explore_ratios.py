"""
Ratio exploration: Train models with different layer/d_model ratios
Explores architecture shapes around 6 layers, 384 d_model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from utils import (
    ModelConfig, ToyTransformer, StreamingTextDataset,
    Trainer, TrainingConfig, count_parameters
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Shared training parameters
n_ctx = 256
n_head = 4
batch_size = 64
seq_length = 256
tokens_per_batch = batch_size * seq_length
total_tokens = 10_000_000
max_iters = total_tokens // tokens_per_batch

print(f"Training configuration:")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Iterations: {max_iters}")
print(f"  Tokens per batch: {tokens_per_batch:,}")
print()

# Define configurations to test (varying layer/d_model ratios)
# Center: 6 layers, 384 d_model
configs = [
    {"n_layers": 3, "d_model": 512, "name": "3L-512D"},   # Few layers, wide
    {"n_layers": 4, "d_model": 448, "name": "4L-448D"},   #
    {"n_layers": 6, "d_model": 384, "name": "6L-384D"},   # Center point
    {"n_layers": 8, "d_model": 320, "name": "8L-320D"},   #
    {"n_layers": 12, "d_model": 256, "name": "12L-256D"}, # Many layers, narrow
]

# Adjust d_model to be divisible by n_head
for config in configs:
    if config["d_model"] % n_head != 0:
        config["d_model"] = (config["d_model"] // n_head) * n_head
        print(f"Adjusted {config['name']} d_model to {config['d_model']} (divisible by n_head={n_head})")

print("\nConfigurations to test:")
for config in configs:
    print(f"  {config['name']}: {config['n_layers']} layers, {config['d_model']} d_model")
print()

# Setup dataset
print("Loading dataset...")
train_dataset = StreamingTextDataset(
    dataset_name='HuggingFaceFW/fineweb',
    split='train',
    tokenizer_name='gpt2',
    seq_length=seq_length,
    subset='sample-10BT'
)
print("Dataset loaded.\n")

# Results storage
results = {
    "configs": configs,
    "transformer": {},
    "bilinear_transformer": {}
}

def train_model(model_name, config_name, n_layers, d_model, squaring_attn, bilinear):
    """Train a single model configuration"""
    print(f"\n{'='*70}")
    print(f"Training: {model_name} - {config_name}")
    print(f"{'='*70}")

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
    print(f"Parameters: {params['total']:,}")

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

            if iter_num % 100 == 0:
                tokens_seen = iter_num * tokens_per_batch
                print(f"  {tokens_seen/1e6:.1f}M tokens | Loss: {loss:.4f} | LR: {lr:.6f}")

    final_loss = losses[-1]
    print(f"Final loss: {final_loss:.4f}")

    return {
        "iterations": iterations,
        "losses": losses,
        "final_loss": final_loss,
        "params": params['total'],
        "config": {
            "n_layers": n_layers,
            "d_model": d_model,
            "squaring_attn": squaring_attn,
            "bilinear": bilinear
        }
    }


# Train all configurations
print("\n" + "="*70)
print("STARTING RATIO EXPLORATION EXPERIMENTS")
print("="*70)

for config in configs:
    config_name = config["name"]
    n_layers = config["n_layers"]
    d_model = config["d_model"]

    # Train transformer
    results["transformer"][config_name] = train_model(
        "Transformer",
        config_name,
        n_layers,
        d_model,
        squaring_attn=False,
        bilinear=False
    )

    # Train bilinear transformer
    results["bilinear_transformer"][config_name] = train_model(
        "Bilinear Transformer",
        config_name,
        n_layers,
        d_model,
        squaring_attn=True,
        bilinear=True
    )

    # Save results after each config (in case of crashes)
    with open('ratio_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to ratio_results.json")

# Create plots
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

# Plot 1: All loss curves for transformers
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for config in configs:
    config_name = config["name"]

    # Transformer
    data = results["transformer"][config_name]
    iters = np.array(data["iterations"]) / 1e6
    ax1.plot(iters, data["losses"], label=config_name, linewidth=2)

    # Bilinear transformer
    data = results["bilinear_transformer"][config_name]
    iters = np.array(data["iterations"]) / 1e6
    ax2.plot(iters, data["losses"], label=config_name, linewidth=2)

ax1.set_xlabel('Tokens (millions)', fontsize=12)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax1.set_title('Transformer (Softmax + SwiGLU)', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Tokens (millions)', fontsize=12)
ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax2.set_title('Bilinear Transformer (Squaring + Bilinear)', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ratio_loss_curves.png', dpi=150)
print("Saved: ratio_loss_curves.png")

# Plot 2: Final loss vs config for both architectures
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

config_names = [c["name"] for c in configs]
transformer_final = [results["transformer"][name]["final_loss"] for name in config_names]
bilinear_final = [results["bilinear_transformer"][name]["final_loss"] for name in config_names]

x = np.arange(len(config_names))
width = 0.35

ax.bar(x - width/2, transformer_final, width, label='Transformer', alpha=0.8)
ax.bar(x + width/2, bilinear_final, width, label='Bilinear Transformer', alpha=0.8)

ax.set_xlabel('Configuration (layers-dmodel)', fontsize=12)
ax.set_ylabel('Final Loss', fontsize=12)
ax.set_title('Final Loss Across Architecture Configurations', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(config_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ratio_final_loss.png', dpi=150)
print("Saved: ratio_final_loss.png")

# Plot 3: Parameter count for each config
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

transformer_params = [results["transformer"][name]["params"] for name in config_names]
bilinear_params = [results["bilinear_transformer"][name]["params"] for name in config_names]

x = np.arange(len(config_names))
width = 0.35

ax.bar(x - width/2, np.array(transformer_params)/1e6, width, label='Transformer', alpha=0.8)
ax.bar(x + width/2, np.array(bilinear_params)/1e6, width, label='Bilinear Transformer', alpha=0.8)

ax.set_xlabel('Configuration (layers-dmodel)', fontsize=12)
ax.set_ylabel('Parameters (millions)', fontsize=12)
ax.set_title('Parameter Count Across Configurations', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(config_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ratio_param_count.png', dpi=150)
print("Saved: ratio_param_count.png")

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nTransformer:")
for config in configs:
    name = config["name"]
    data = results["transformer"][name]
    print(f"  {name:12s} | Params: {data['params']:>10,} | Final Loss: {data['final_loss']:.4f}")

print("\nBilinear Transformer:")
for config in configs:
    name = config["name"]
    data = results["bilinear_transformer"][name]
    print(f"  {name:12s} | Params: {data['params']:>10,} | Final Loss: {data['final_loss']:.4f}")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print(f"\nResults saved to:")
print(f"  - ratio_results.json")
print(f"  - ratio_loss_curves.png")
print(f"  - ratio_final_loss.png")
print(f"  - ratio_param_count.png")
