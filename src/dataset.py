import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Sequential pretraining dataset backed by memmap.

    __getitem__(idx) returns the idx-th contiguous chunk of seq_len+1 tokens
    as (x, y) where x = tokens[:-1], y = tokens[1:].

    No shuffling at dataset level — leave that to the DataLoader / Sampler.
    """

    def __init__(self, filepath: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(filepath, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)

    def __len__(self) -> int:
        return self.num_tokens // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.seq_len
        chunk = self.data[i : i + self.seq_len + 1].copy()
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y


class SFTDataset(Dataset):
    def __init__(self, filepath: str):
        data = torch.load(filepath)
        self.ids, self.mask = data["input_ids"], data["assistant_mask"]
        self.num_tokens = len(self.ids) * len(self.ids[0])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx][:-1], self.ids[idx][1:], self.mask[idx][:-1]


class MixedSFTDataset(Dataset):
    """Two SFT datasets mixed at a fixed ratio, with shuffled interleaving.

    Samples are randomly shuffled at init time so that each contiguous slice
    of indices contains a roughly proportional mix of both datasets.  This
    avoids needing ``shuffle=True`` on ``DistributedSampler``, which can cause
    device-generator mismatches in newer PyTorch versions.

    Args:
        path_a: minority dataset filepath (e.g., general SFT data).
        path_b: majority dataset filepath (e.g., roast SFT data).
        ratio_a: fraction of total data that comes from path_a (0.0 ~ 1.0).
        seed: random seed for the index shuffle.
    """

    def __init__(self, path_a: str, path_b: str, ratio_a: float, seed: int = 42):
        self.a = SFTDataset(path_a)
        self.b = SFTDataset(path_b)
        self._len = int(len(self.b) / (1 - ratio_a))
        self._limit_a = self._len - len(self.b)
        self.num_tokens = self.a.num_tokens + self.b.num_tokens  # ~ approximate

        # Build and shuffle the per-index dataset routing
        rng = np.random.RandomState(seed)
        routing = np.empty(self._len, dtype=np.int32)
        routing[: self._limit_a] = 0  # from dataset A (cycled)
        routing[self._limit_a :] = 1  # from dataset B
        rng.shuffle(routing)
        self._routing = routing

        # Per-dataset sample counter so we cycle each one independently
        self._counters = np.zeros(2, dtype=np.int64)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        which = int(self._routing[idx])
        if which == 0:
            src_idx = self._counters[0] % len(self.a)
            self._counters[0] += 1
            return self.a[src_idx]
        else:
            src_idx = self._counters[1] % len(self.b)
            self._counters[1] += 1
            return self.b[src_idx]


class GRPOOffPolicyDataset(Dataset):
    """Off-policy GRPO dataset: pre-tokenized prompts + 4 candidates + editor scores.

    Each original record is flattened into 4 rows (one per candidate response).
    ``group_ids`` link the 4 candidates belonging to the same prompt so the
    GRPO trainer can compute advantages within each group.

    Returns
        inputs:       (1023,)  input_ids[:-1]
        targets:      (1023,)  input_ids[1:]
        comp_mask:    (1023,)  completion_mask[:-1] — True on candidate tokens
        scores:       (5,)     5-dim editor scores
        group_id:     scalar   which prompt this candidate belongs to
        is_winner:    scalar   bool, whether this candidate is the editor's winner
    """

    def __init__(self, filepath: str):
        data = torch.load(filepath)
        self.ids = data["input_ids"]
        self.mask = data["completion_mask"]
        self.scores = data["scores"]
        self.group_ids = data["group_ids"]
        self.is_winner = data["is_winner"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            self.ids[idx][:-1],
            self.ids[idx][1:],
            self.mask[idx][:-1],
            self.scores[idx],
            self.group_ids[idx],
            self.is_winner[idx],
        )


class GRPOOnPolicyDataset(Dataset):
    """On-policy GRPO dataset: pre-tokenized prompts only (no responses).

    Returns
        input_ids:   (1024,)  full padded prompt ending with ``<|assistant|>\\n``
        prompt_mask: (1024,)  True on non-padding tokens
    """

    def __init__(self, filepath: str):
        data = torch.load(filepath)
        self.ids = data["input_ids"]
        self.prompt_mask = data["prompt_mask"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.prompt_mask[idx]
