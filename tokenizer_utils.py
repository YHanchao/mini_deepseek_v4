"""
Implementation of BPE
"""

import os
from typing import BinaryIO
import multiprocessing as mp
from collections import defaultdict
import regex as re

# This function is directly copied from CS336
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class BPETokenizer:
    def __init__(self, fname: str, vocab_size: int, special_tokens: list[bytes],
                 max_chunk_size: int, regex_pattern: str, num_processes: int):
        self.fname = fname
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.max_chunk_size = max_chunk_size
        self.vocabs = {i: bytes([i]) for i in range(256)}
        for idx, token in enumerate(special_tokens):
            self.vocabs[256 + idx] = token
        self.regex_pattern = regex_pattern
        self.num_processes = num_processes

    def _pre_tokenize_chunk(self, chunk_text):
        counts = defaultdict(int)
        # 先按 special tokens 切分，保证 special token 不被拆开
        if self.special_tokens:
            special_pattern = '|'.join(re.escape(t.decode('utf-8')) for t in self.special_tokens)
            parts = re.split(f'({special_pattern})', chunk_text)
        else:
            parts = [chunk_text]

        for part in parts:
            if not part:
                continue
            # 检查是否是 special token
            if self.special_tokens and any(part == t.decode('utf-8') for t in self.special_tokens):
                counts[part] += 1
            else:
                for match in re.finditer(self.regex_pattern, part):
                    word = match.group()
                    counts[word] += 1
        return counts

    def _pre_tokenize_worker(self, task_queue: mp.Queue, result_queue: mp.Queue):
        while True:
            chunk_start, chunk_end = task_queue.get()
            if chunk_start is None:
                result_queue.put(None)  # 哨兵，通知主进程该 worker 已结束
                break

            with open(self.fname, "rb") as f:
                f.seek(chunk_start)
                chunk_data = f.read(chunk_end - chunk_start)  # 读入内存
                chunk_text = chunk_data.decode("utf-8", errors="ignore")

                local_counts = self._pre_tokenize_chunk(chunk_text)
                result_queue.put(local_counts)

    def pre_tokenize(self):
        # 每个chunk比方说只占用50MB
        with open(self.fname, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            if file_size <= self.max_chunk_size:
                chunk_num = 1
            else:
                chunk_num = file_size // self.max_chunk_size
            boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")
            chunks = list(zip(boundaries[:-1], boundaries[1:]))

        # Chunk 太少时直接用单进程，避免 fork 多进程的开销
        num_workers = min(self.num_processes, len(chunks))
        if num_workers <= 1:
            total_counts = self._pre_tokenize_chunk(
                open(self.fname, "rb").read().decode("utf-8", errors="ignore")
            )
            return total_counts

        # 多进程并行处理
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        workers = []
        for _ in range(num_workers):
            w = mp.Process(target=self._pre_tokenize_worker, args=(task_queue, result_queue))
            w.start()
            workers.append(w)

        # 主进程分发任务
        for chunk in chunks:
            task_queue.put(chunk)
        for _ in workers:
            task_queue.put((None, None))

        # 收集结果
        total_counts = defaultdict(int)
        finished_workers = 0
        while finished_workers < len(workers):
            local_counts = result_queue.get()
            if local_counts is None:  # worker 结束哨兵
                finished_workers += 1
            else:
                for word, count in local_counts.items():
                    total_counts[word] += count

        for w in workers:
            w.join()

        return total_counts
    
    def create_index(self, pre_tokenize_counts):
        # 构建一个索引，后面BPE合并的时候基于索引去做，而不是遍历查询字符串
        word_registry: dict[int, dict] = {}  # word: (seq, freq)
        pair_to_words: dict[tuple, set] = {}  # pair: {(word, pos_in_word)}
        pair_freq: dict[tuple, int] = {}      # pair: freq
        next_word_id = 0

        # special token bytes → token ID 的映射，用于识别预分词结果中的 special token
        special_token_to_id = {t: 256 + i for i, t in enumerate(self.special_tokens)}

        for word, freq in pre_tokenize_counts.items():
            word_bytes = word.encode('utf-8')
            word_id = next_word_id
            next_word_id += 1

            # 如果这个词是 special token，作为一个原子 token 存储，不参与 pair 统计
            if word_bytes in special_token_to_id:
                token_id = special_token_to_id[word_bytes]
                word_registry[word_id] = {'seq': (token_id,), 'freq': freq}
                continue

            byte_seq = tuple(word_bytes)
            word_registry[word_id] = {'seq': byte_seq, 'freq': freq}

            # 记录这个词中的所有pair
            for pos in range(len(byte_seq) - 1):
                pair = (byte_seq[pos], byte_seq[pos+1])

                pair_freq[pair] = pair_freq.get(pair, 0) + freq

                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add((word_id, pos))

        return word_registry, pair_to_words, pair_freq

    def bpe_merge(self, word_registry: dict[int, dict], pair_to_words: dict[tuple, set], pair_freq: dict[tuple, int]):
        """
        BPE的实现思路

        - Step 1: 找到这轮要合并的token，记录到vocabs。这一步里这个pair的两个东西要拼在一起形成一块东西
        - Step 2: 找到受影响的那些word，换言之，只要pair里某个元素出现在要合并的那个pair，就算做受影响
        """
        next_token_id = len(self.vocabs)
        merge = []
        while len(self.vocabs) < self.vocab_size:
            # Step 1: Find best pair
            # 频率相同时，按 pair 的字节表示字母序降序打破平局
            best_pair = max(pair_freq.items(), key=lambda x: (x[1], self.vocabs[x[0][0]], self.vocabs[x[0][1]]))[0]
            merge.append((self.vocabs[best_pair[0]], self.vocabs[best_pair[1]]))
            new_token_bytes = self.vocabs[best_pair[0]] + self.vocabs[best_pair[1]]
            new_token_id = next_token_id
            self.vocabs[new_token_id] = new_token_bytes
            next_token_id += 1
            
            # 接下来找包含这个最新合并的token的词
            # 举个例子，比方说w o r l d 中的rl要合并
            # 那么用(r, l)查询pair_to_words，能够拿到world这个词
            # 1. word_registry里面的修改：rl合并
            # 2. pair_to_words和pair_freq中，涉及到r 和 l 的pair都要改
            # (o, r) -> (o, rl)
            # (r, l) -> 没这个pair
            # (l, d) -> (rl, d)
            # 2.1: 找到pair
            # 2.2: 更新pair_to_words和pair_freq
            # 实现方式，先把这个受影响的词的freq和seq去掉，然后再重新加入修改后的freq和seq


            affected_words = pair_to_words.get(best_pair, set())
            # 去重：同一个 word_id 可能在多个位置包含 best_pair
            affected_word_ids = set(wid for wid, _ in affected_words)

            # 1. 从所有pair中移除受影响词的所有出现
            for word_id in affected_word_ids:
                word_info = word_registry[word_id]
                old_seq = word_info['seq']

                for pos in range(len(old_seq) - 1):
                    old_pair = (old_seq[pos], old_seq[pos+1])
                    if old_pair in pair_to_words:
                        pair_to_words[old_pair].discard((word_id, pos))
                        if not pair_to_words[old_pair]:
                            del pair_to_words[old_pair]

                    if old_pair in pair_freq:
                        pair_freq[old_pair] -= word_info['freq']
                        if pair_freq[old_pair] <= 0:
                            del pair_freq[old_pair]

            # 2. 更新word_registry中的seq：左到右贪心扫描，合并所有 best_pair 出现
            for word_id in affected_word_ids:
                word_info = word_registry[word_id]
                old_seq = word_info['seq']
                new_seq = []
                i = 0
                while i < len(old_seq):
                    if i < len(old_seq) - 1 and (old_seq[i], old_seq[i+1]) == best_pair:
                        new_seq.append(new_token_id)
                        i += 2
                    else:
                        new_seq.append(old_seq[i])
                        i += 1
                word_info['seq'] = tuple(new_seq)

            # 3. 重新添加新seq的所有pair
            for word_id in affected_word_ids:
                word_info = word_registry[word_id]
                new_seq = word_info['seq']
                freq = word_info['freq']

                for new_pos in range(len(new_seq) - 1):
                    new_pair = (new_seq[new_pos], new_seq[new_pos+1])
                    pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
                    if new_pair not in pair_to_words:
                        pair_to_words[new_pair] = set()
                    pair_to_words[new_pair].add((word_id, new_pos))
        
        return self.vocabs, merge


    def train(self):
        # Step 1: pretokenized
        # 这样得到的就是词频，直接对词频算pair occurance即可
        pre_tokenize_counts = self.pre_tokenize()

        # Step 2: 构造索引，方便后续BPE合并
        word_registry, pair_to_words, pair_freq = self.create_index(pre_tokenize_counts)

        # Step 3: BPE merge
        vocabs, merges = self.bpe_merge(word_registry, pair_to_words, pair_freq)

        return vocabs, merges