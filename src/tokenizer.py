"""
Implementation of BPE
"""

import os
import heapq
from typing import BinaryIO, Iterable, Iterator
import multiprocessing as mp
from collections import defaultdict
import regex as re
import json


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
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

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
    # GPT-2 预分词正则，直接写死
    PRETOKENIZE_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    @staticmethod
    def _bytes_to_unicode() -> dict[int, str]:
        """byte → printable unicode 的映射，与 GPT-2 一致"""
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocabs = dict(vocab)
        self.vocab_size = len(vocab)
        self.merges = list(merges) if merges else []
        self.byte_to_id = {v: k for k, v in self.vocabs.items()}

        if special_tokens:
            self.special_tokens = [t.encode("utf-8") for t in special_tokens]
        else:
            self.special_tokens = []

        # 预编译正则和 special token 查找结构
        self._compiled_regex = re.compile(self.PRETOKENIZE_PATTERN)
        self._special_token_strs = {t.decode("utf-8") for t in self.special_tokens}
        if self.special_tokens:
            # 按长度降序排列，保证更长的 special token 优先匹配（如 <|endoftext|><|endoftext|> 优先于 <|endoftext|>）
            escaped = "|".join(
                re.escape(t.decode("utf-8"))
                for t in sorted(self.special_tokens, key=lambda x: -len(x))
            )
            self._special_split_re = re.compile(f"({escaped})")
        else:
            self._special_split_re = None
        self._special_token_to_id = {}
        for st in self.special_tokens:
            for tid, tbytes in self.vocabs.items():
                if tbytes == st:
                    self._special_token_to_id[st] = tid
                    break

        # pair → rank 映射，rank 越小优先级越高（等于 merges 列表中的下标）
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        byte_decoder = {v: k for k, v in cls._bytes_to_unicode().items()}

        with open(vocab_filepath) as f:
            raw_vocab = json.load(f)
        # {unicode_encoded_str: token_id} → {token_id: bytes}
        vocab = {
            token_id: bytes([byte_decoder[ch] for ch in encoded_str])
            for encoded_str, token_id in raw_vocab.items()
        }

        with open(merges_filepath) as f:
            merges = []
            for line_ in f:
                line_ = line_.rstrip()
                if line_:
                    parts = line_.split()
                    if len(parts) == 2:
                        t1 = bytes([byte_decoder[ch] for ch in parts[0]])
                        t2 = bytes([byte_decoder[ch] for ch in parts[1]])
                        merges.append((t1, t2))

        return cls(vocab, merges, special_tokens)

    def to_file(self, vocab_filepath: str, merges_filepath: str):
        byte_to_unicode = self._bytes_to_unicode()

        with open(vocab_filepath, "w", encoding="utf-8") as f:
            serialized = {
                "".join(byte_to_unicode[b] for b in token_bytes): token_id
                for token_id, token_bytes in self.vocabs.items()
            }
            json.dump(serialized, f, ensure_ascii=False)

        with open(merges_filepath, "w", encoding="utf-8") as f:
            for t1, t2 in self.merges:
                s1 = "".join(byte_to_unicode[b] for b in t1)
                s2 = "".join(byte_to_unicode[b] for b in t2)
                f.write(f"{s1} {s2}\n")

    def _pre_tokenize_chunk(self, chunk_text):
        counts = defaultdict(int)
        # 先按 special tokens 切分，保证 special token 不被拆开
        if self._special_split_re is not None:
            parts = self._special_split_re.split(chunk_text)
        else:
            parts = [chunk_text]

        for part in parts:
            if not part:
                continue
            # 检查是否是 special token（O(1) 集合查找）
            if part in self._special_token_strs:
                counts[part] += 1
            else:
                for match in self._compiled_regex.finditer(part):
                    word = match.group()
                    counts[word] += 1
        return counts

    def _pre_tokenize_worker(self, task_queue: mp.Queue, result_queue: mp.Queue):
        while True:
            fname, chunk_start, chunk_end = task_queue.get()
            if chunk_start is None:
                result_queue.put(None)  # 哨兵，通知主进程该 worker 已结束
                break

            with open(fname, "rb") as f:
                f.seek(chunk_start)
                chunk_data = f.read(chunk_end - chunk_start)  # 读入内存
                chunk_text = chunk_data.decode("utf-8", errors="ignore")

                local_counts = self._pre_tokenize_chunk(chunk_text)
                result_queue.put(local_counts)

    def pre_tokenize(self, fname):
        # 每个chunk比方说只占用50MB
        with open(fname, "rb") as f:
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
                open(fname, "rb").read().decode("utf-8", errors="ignore")
            )
            return total_counts

        # 多进程并行处理
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        workers = []
        for _ in range(num_workers):
            w = mp.Process(
                target=self._pre_tokenize_worker, args=(task_queue, result_queue)
            )
            w.start()
            workers.append(w)

        # 主进程分发任务
        for chunk_start, chunk_end in chunks:
            task_queue.put((fname, chunk_start, chunk_end))
        for _ in workers:
            task_queue.put((fname, None, None))

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
        pair_freq: dict[tuple, int] = {}  # pair: freq
        next_word_id = 0

        for word, freq in pre_tokenize_counts.items():
            word_bytes = word.encode("utf-8")
            word_id = next_word_id
            next_word_id += 1

            # 如果这个词是 special token，作为一个原子 token 存储，不参与 pair 统计
            if word_bytes in self._special_token_to_id:
                token_id = self._special_token_to_id[word_bytes]
                word_registry[word_id] = {"seq": (token_id,), "freq": freq}
                continue

            byte_seq = tuple(word_bytes)
            word_registry[word_id] = {"seq": byte_seq, "freq": freq}

            # 记录这个词中的所有pair
            for pos in range(len(byte_seq) - 1):
                pair = (byte_seq[pos], byte_seq[pos + 1])

                pair_freq[pair] = pair_freq.get(pair, 0) + freq

                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add((word_id, pos))

        return word_registry, pair_to_words, pair_freq

    def bpe_merge(
        self,
        word_registry: dict[int, dict],
        pair_to_words: dict[tuple, set],
        pair_freq: dict[tuple, int],
    ):
        """
        BPE的实现思路

        - Step 1: 找到这轮要合并的token，记录到vocabs。这一步里这个pair的两个东西要拼在一起形成一块东西
        - Step 2: 找到受影响的那些word，换言之，只要pair里某个元素出现在要合并的那个pair，就算做受影响
        """
        next_token_id = len(self.vocabs)
        self.merges = []

        # 用最大堆替代每轮 max() 全量扫描，将选最优 pair 从 O(P) 降到 O(log P)
        heap = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)

        def _pop_best():
            candidates = []
            best_neg_freq = None
            while heap:
                neg_freq, pair = heap[0]
                cur = pair_freq.get(pair)
                if cur is None or cur != -neg_freq:
                    heapq.heappop(heap)
                    continue
                if best_neg_freq is None:
                    best_neg_freq = neg_freq
                if neg_freq != best_neg_freq:
                    break
                candidates.append(pair)
                heapq.heappop(heap)
            if not candidates:
                return None
            best_pair = max(
                candidates, key=lambda p: (self.vocabs[p[0]], self.vocabs[p[1]])
            )
            for p in candidates:
                if p != best_pair and p in pair_freq:
                    heapq.heappush(heap, (-pair_freq[p], p))
            return best_pair

        total_merges = self.vocab_size - len(self.vocabs)
        milestone = max(1, total_merges // 10)
        print(
            f"[BPE] 开始合并，目标 {self.vocab_size} tokens，当前 {len(self.vocabs)}，预计 {total_merges} 轮"
        )

        while len(self.vocabs) < self.vocab_size:
            best_pair = _pop_best()
            if best_pair is None:
                break

            self.merges.append((self.vocabs[best_pair[0]], self.vocabs[best_pair[1]]))
            new_token_bytes = self.vocabs[best_pair[0]] + self.vocabs[best_pair[1]]
            new_token_id = next_token_id
            self.vocabs[new_token_id] = new_token_bytes
            next_token_id += 1

            if len(self.merges) % milestone == 0:
                print(
                    f"[BPE] merge {len(self.merges)}/{total_merges}, vocab 已到 {next_token_id}"
                )

            affected_words = pair_to_words.get(best_pair, set())
            affected_word_ids = set(wid for wid, _ in affected_words)

            modified_pairs = set()

            # 1. 从所有pair中移除受影响词的所有出现
            for word_id in affected_word_ids:
                word_info = word_registry[word_id]
                old_seq = word_info["seq"]
                freq = word_info["freq"]

                for pos in range(len(old_seq) - 1):
                    old_pair = (old_seq[pos], old_seq[pos + 1])
                    if old_pair in pair_to_words:
                        pair_to_words[old_pair].discard((word_id, pos))
                        if not pair_to_words[old_pair]:
                            del pair_to_words[old_pair]

                    if old_pair in pair_freq:
                        pair_freq[old_pair] -= freq
                        if pair_freq[old_pair] <= 0:
                            del pair_freq[old_pair]

                    modified_pairs.add(old_pair)

            # 2. 更新 word_registry 中的 seq
            for word_id in affected_word_ids:
                word_info = word_registry[word_id]
                old_seq = word_info["seq"]
                new_seq = []
                i = 0
                while i < len(old_seq):
                    if (
                        i < len(old_seq) - 1
                        and (old_seq[i], old_seq[i + 1]) == best_pair
                    ):
                        new_seq.append(new_token_id)
                        i += 2
                    else:
                        new_seq.append(old_seq[i])
                        i += 1
                word_info["seq"] = tuple(new_seq)

            # 3. 重新添加新 seq 的所有 pair
            for word_id in affected_word_ids:
                word_info = word_registry[word_id]
                new_seq = word_info["seq"]
                freq = word_info["freq"]

                for new_pos in range(len(new_seq) - 1):
                    new_pair = (new_seq[new_pos], new_seq[new_pos + 1])
                    pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
                    if new_pair not in pair_to_words:
                        pair_to_words[new_pair] = set()
                    pair_to_words[new_pair].add((word_id, new_pos))
                    modified_pairs.add(new_pair)

            # 将频率发生变化的 pair 推入堆（惰性删除处理旧条目）
            for pair in modified_pairs:
                if pair in pair_freq:
                    heapq.heappush(heap, (-pair_freq[pair], pair))

        return self.vocabs, self.merges

    def train(
        self, fname, vocab_size, max_chunk_size=64 * 1024 * 1024, num_processes=1
    ):
        import time

        # 重置 vocabs 为初始状态：256 基础字节 + special tokens
        self.vocab_size = vocab_size
        self.max_chunk_size = max_chunk_size
        self.num_processes = num_processes
        self.vocabs = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(self.special_tokens):
            self.vocabs[256 + i] = token
            self._special_token_to_id[token] = 256 + i

        print(
            f"[train] 语料: {fname}, 目标 vocab: {vocab_size}, workers: {num_processes}"
        )

        # Step 1: pretokenized
        t0 = time.time()
        pre_tokenize_counts = self.pre_tokenize(fname)
        print(
            f"[train] 预分词完成，{len(pre_tokenize_counts)} 个唯一 pre-token，耗时 {time.time() - t0:.1f}s"
        )

        # Step 2: 构造索引，方便后续BPE合并
        t0 = time.time()
        word_registry, pair_to_words, pair_freq = self.create_index(pre_tokenize_counts)
        print(
            f"[train] 索引构建完成，{len(pair_freq)} 对 pair，耗时 {time.time() - t0:.1f}s"
        )

        # Step 3: BPE merge
        t0 = time.time()
        vocabs, merges = self.bpe_merge(word_registry, pair_to_words, pair_freq)
        print(
            f"[train] BPE 合并完成，{len(merges)} 条 merges，耗时 {time.time() - t0:.1f}s"
        )

        self.byte_to_id = {v: k for k, v in vocabs.items()}
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        return vocabs, merges

    def _apply_merges(self, byte_seq: list[bytes]) -> list[bytes]:
        if len(byte_seq) <= 1:
            return byte_seq.copy()

        current = list(byte_seq)
        heap = []

        # 扫描初始序列，将可合并 pair 入堆
        for pos in range(len(current) - 1):
            rank = self._merge_rank.get((current[pos], current[pos + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, pos))

        while heap:
            rank, pos = heapq.heappop(heap)

            # 惰性删除：确认该位置仍有效
            if pos >= len(current) - 1:
                continue
            cur_rank = self._merge_rank.get((current[pos], current[pos + 1]))
            if cur_rank != rank:
                continue

            # 合并
            current[pos : pos + 2] = [current[pos] + current[pos + 1]]

            # 合并后右侧所有 pair 位置都偏移了一位，重新扫描整个序列
            heap.clear()
            for scan_pos in range(len(current) - 1):
                scan_rank = self._merge_rank.get(
                    (current[scan_pos], current[scan_pos + 1])
                )
                if scan_rank is not None:
                    heapq.heappush(heap, (scan_rank, scan_pos))

        return current

    def encode(self, text: str):
        if not text:
            return []

        if self._special_split_re is not None:
            parts = self._special_split_re.split(text)
        else:
            parts = [text]

        all_token_ids = []

        for part in parts:
            if part in self._special_token_strs:
                all_token_ids.append(self._special_token_to_id[part.encode("utf-8")])
                continue

            # 用同样的 regex pattern 做预分词，保证 BPE 合并不跨 pre-token 边界
            for _match in self._compiled_regex.finditer(part):
                word = _match.group()
                byte_seq = [bytes([b]) for b in word.encode("utf-8")]
                merged = self._apply_merges(byte_seq)
                token_ids = [self.byte_to_id[token] for token in merged]
                all_token_ids.extend(token_ids)

        return all_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        惰性编码，适用于大文件
        """
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            for token_id in token_ids:
                yield token_id

    def decode(self, token_ids: list[int]) -> str:
        all_bytes = b"".join([self.vocabs[token_id] for token_id in token_ids])
        return all_bytes.decode("utf-8", errors="replace")
