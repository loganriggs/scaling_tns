"""
Efficient binary data loader from modded-nanogpt
Supports distributed training and variable-length sequences with BOS alignment
"""

import glob
import threading
from pathlib import Path
import torch
from torch import Tensor
import torch.distributed as dist


# GPT-2 Beginning of Sequence token ID
BOS_ID = 50256


def _load_data_shard(filename: Path) -> Tensor:
    """
    Load a single .bin data shard into memory.

    Binary file format:
    - Header: 256 int32 values (first value is number of tokens)
    - Data: tokens as uint16 values

    Args:
        filename: Path to .bin file

    Returns:
        Tensor of token IDs
    """
    with open(filename, "rb") as f:
        header = torch.from_file(f.name, False, 256, dtype=torch.int32)
        num_tokens = int(header[0])
        tokens = torch.empty(num_tokens, dtype=torch.int16)
        f.seek(256 * 4)  # Skip header
        nbytes = f.readinto(tokens.numpy())  # Avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


class BOSFinder:
    """
    Helper for getting sequences that start at the beginning of documents.
    Precomputes BOS positions for efficient batch sampling.

    Based on work by @varunneal and @classiclarryd
    """

    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        """
        Initialize BOSFinder with a token shard.

        Args:
            tokens: Tensor of token IDs
            world_size: Number of distributed processes
            quickload: If True, scan only first 4M tokens initially, then async scan rest
        """
        self.tokens = tokens
        self.size = tokens.numel()
        self.quickload = quickload

        if quickload:
            # Only scan first 4 million tokens, then kickoff async thread to scan rest
            self.bos_idx = (tokens[:4_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()

        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        """Background thread function to scan full token array for BOS positions."""
        self.bos_idx_async = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()

    def start(self):
        """Start async BOS scanning thread (used with quickload)."""
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        """Wait for async BOS scan to complete and update bos_idx."""
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        """
        Get next batch of sequences aligned to document boundaries.

        Args:
            num_tokens_local: Number of tokens per rank
            max_seq_len: Maximum sequence length

        Returns:
            Tuple of (starts, ends) where each is a list of lists:
            - starts[rank] = list of start positions for that rank
            - ends[rank] = list of end positions for that rank

        Raises:
            StopIteration: When shard is exhausted
        """
        # If quickload was used, switch to full dataset after 5 batches
        if self.quickload and self.batch_iter == 5:
            self.get()

        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")

                cur = self.bos_idx[idx]
                starts[r].append(cur)

                # Calculate end: min of (next BOS, max_seq_len, remaining tokens needed)
                end = min(
                    self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                    cur + max_seq_len,
                    cur + num_tokens_local - cur_len + 1
                )
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1, f"Expected {num_tokens_local + 1}, got {cur_len}"

        self.i = idx
        self.batch_iter += 1
        return starts, ends


class DataPreloader:
    """
    Helper for asynchronously loading next shard and indexing BOS tokens.
    Overlaps I/O with computation for better performance.
    """

    def __init__(self, file_iter, world_size: int = 1):
        """
        Initialize DataPreloader.

        Args:
            file_iter: Iterator over data file paths
            world_size: Number of distributed processes
        """
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        """Background thread function to load next shard."""
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        """Start async loading of next shard."""
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        """Wait for load to complete and return (tokens, finder)."""
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data


def distributed_data_generator(
    filename_pattern: str,
    num_tokens: int,
    max_seq_len: int,
    grad_accum_steps: int = 1,
    align_to_bos: bool = True
):
    """
    Main data generator for distributed training with variable-length sequences.

    Yields batches of (inputs, targets, cumulative_lengths) where:
    - inputs: token IDs for model input [num_tokens_local]
    - targets: shifted token IDs for loss computation [num_tokens_local]
    - cumulative_lengths: document boundary positions for varlen attention

    Args:
        filename_pattern: Glob pattern for .bin files (e.g., "data/train_*.bin")
        num_tokens: Total tokens per batch across all ranks
        max_seq_len: Maximum sequence length
        grad_accum_steps: Gradient accumulation steps
        align_to_bos: If True, align sequences to document boundaries

    Yields:
        Tuple of (inputs, targets, cumulative_lengths) as CUDA tensors

    Note:
        Generator supports dynamic parameter updates via .send((num_tokens, max_seq_len, grad_accum_steps))
    """
    # Get distributed info
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    assert num_tokens % (world_size * grad_accum_steps) == 0, \
        "Batch size must be divisible by world size * grad_accum_steps"
    num_tokens = num_tokens // grad_accum_steps

    # Load file list
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    # Initialize data loading
    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))

    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one
                tokens, finder = preloader.get()
                preloader.start()
                continue

            # Extract sequences from token buffer
            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            # Unaligned mode: just chunk tokens sequentially
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens

        # Pad cumulative lengths to fixed size for efficient batching
        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Yield batch (move to GPU)
        new_params = yield (
            _inputs.to(device="cuda", dtype=torch.int32, non_blocking=True),
            _targets.to(device="cuda", dtype=torch.int64, non_blocking=True),
            _cum_lengths.to(device="cuda", dtype=torch.int32, non_blocking=True)
        )

        # Allow dynamic parameter updates via .send()
        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0, \
                "Num tokens must be divisible by world size * grad_accum_steps"
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps


def next_multiple_of_n(v: float | int, *, n: int) -> int:
    """Round up to next multiple of n."""
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use the distributed data generator.

    Usage:
        # Single GPU
        python dataloader.py

        # Multi-GPU
        torchrun --standalone --nproc_per_node=2 dataloader.py
    """
    import torch.distributed as dist

    # Initialize distributed (if running with torchrun)
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0

    # Create data generator
    data_gen = distributed_data_generator(
        filename_pattern="data/fineweb10B/fineweb_train_*.bin",
        num_tokens=2048 * 16,  # 32K tokens per batch
        max_seq_len=128 * 16,  # 2048 max sequence length
        grad_accum_steps=1,
        align_to_bos=True
    )

    # Get a few batches
    for i in range(5):
        inputs, targets, cum_lengths = next(data_gen)
        if rank == 0:
            print(f"Batch {i}:")
            print(f"  inputs.shape: {inputs.shape}")
            print(f"  targets.shape: {targets.shape}")
            print(f"  cum_lengths.shape: {cum_lengths.shape}")
            print(f"  num_docs: {(cum_lengths > 0).sum().item()}")

    if dist.is_initialized():
        dist.destroy_process_group()
