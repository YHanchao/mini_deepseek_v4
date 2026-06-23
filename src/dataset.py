import ctypes
import mmap
import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Sequential pretraining dataset backed by memmap.

    __getitem__(idx) returns the idx-th contiguous chunk of seq_len+1 tokens
    as (x, y) where x = tokens[:-1], y = tokens[1:].

    Uses MADV_SEQUENTIAL to prevent the page cache from accumulating the
    entire dataset in CPU RAM over long training runs.
    """

    def __init__(self, filepath: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(filepath, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)

        # MADV_SEQUENTIAL: tell kernel to aggressively drop pages behind the
        # read cursor so the dataset isn't cached entirely in CPU RAM
        try:
            MADV_SEQUENTIAL = 2
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            buf = memoryview(self.data)
            ptr = ctypes.c_void_p.from_buffer(buf)
            size = buf.nbytes
            libc.madvise(ptr, ctypes.c_size_t(size), ctypes.c_int(MADV_SEQUENTIAL))
        except Exception:
            pass

    def __len__(self) -> int:
        return self.num_tokens // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.seq_len
        chunk = self.data[i : i + self.seq_len + 1].copy()
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y
