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
