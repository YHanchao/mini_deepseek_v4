"""
Inference engine for DeepSeekV4 models.

Lightweight, extensible inference framework:
  InferenceEngine  —  base class for text completion
  ChatEngine       —  chat template support
  Sampling utils   —  top-k, top-p, greedy
"""

from dataclasses import dataclass
from typing import Optional, Union, List

import torch

from config import MODEL_CONFIGS, SPECIAL_TOKENS
from src.deepseek import DeepSeekV4, DSArgs
from src.tokenizer import BPETokenizer


# ======================================================================
# Sampling utilities
# ======================================================================


@torch.no_grad()
def sample_top_k(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    """Top-k filtered temperature sampling with repetition penalty.

    Args:
        logits: (batch, vocab_size)
        temperature: softmax temperature
        top_k: number of top tokens to keep
        generated_ids: list of already-generated token ids (for penalty)
        repetition_penalty: > 1 penalises repeated tokens, = 1 disabled

    Returns:
        (batch, 1) tensor of sampled token ids
    """
    if generated_ids and repetition_penalty != 1.0:
        for tid in set(generated_ids):
            logits[:, tid] = logits[:, tid] / repetition_penalty

    logits = logits / max(temperature, 1e-8)
    k = min(top_k, logits.size(-1))
    vals, _ = torch.topk(logits, k, dim=-1)
    logits[logits < vals[:, -1:]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def sample_top_p(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.9,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    """Nucleus (top-p) sampling with repetition penalty."""
    if generated_ids and repetition_penalty != 1.0:
        for tid in set(generated_ids):
            logits[:, tid] = logits[:, tid] / repetition_penalty

    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > top_p
    mask[:, 1:] = mask[:, :-1].clone()
    mask[:, 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_idx)


@torch.no_grad()
def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy (argmax) sampling."""
    return logits.argmax(dim=-1, keepdim=True)


# ======================================================================
# InferenceConfig
# ======================================================================


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Model
    checkpoint_path: str = ""
    config_name: str = "small"
    device: str = "cuda:0"

    # Tokenizer
    tokenizer_vocab: str = "checkpoints/tokenizer_vocab.json"
    tokenizer_merges: str = "checkpoints/tokenizer_merges.txt"

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1

    # EOS
    eos_token: str = "<|endoftext|>"


# ======================================================================
# InferenceEngine
# ======================================================================


class InferenceEngine:
    """Base inference engine for DeepSeekV4 text completion.

    Lifecycle::

        engine = InferenceEngine(config)
        engine.setup()               # load model + tokenizer
        text = engine.generate(p)    # generate (str or list[str])

    Subclass and override:
        - preprocess(prompt) -> (ids, eos_id)
        - postprocess(ids) -> str
        - sample(logits) -> Tensor
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model: Optional[DeepSeekV4] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self._step: int = 0
        self._model_args: Optional[DSArgs] = None
        self._init_buffers: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        """Load model, tokenizer, and snapshot initial buffer state."""
        # deepseek.py sets default device at import time — match it
        old_device = torch.get_default_device()
        torch.set_default_device(self.device)

        self.build_model()
        self.build_tokenizer()
        self._capture_buffers()
        self.model.eval()

        torch.set_default_device(old_device)

        print(f"InferenceEngine ready: step={self._step}, "
              f"d_model={self._model_args.d_model}, "
              f"n_layer={self._model_args.n_layer}")

    def build_model(self):
        """Build model from config and load checkpoint."""
        cfg = MODEL_CONFIGS[self.config.config_name].copy()

        base_fields = DSArgs.__dataclass_fields__
        args = DSArgs(**{k: v for k, v in cfg.items() if k in base_fields})
        args.device = str(self.device)

        # Pad compress_ratios for MTP layers (same patch as trainer)
        needed = args.n_layer + args.n_mtp_layer
        if len(args.compress_ratios) < needed:
            args.compress_ratios = args.compress_ratios + tuple(
                [0] * (needed - len(args.compress_ratios))
            )

        self._model_args = args
        # Minimum prompt length to avoid Indexer crash on short inputs
        # (Compressor returns None when seq_len < compress_ratio, and
        #  the Indexer does not handle that case gracefully).
        nonzero = [r for r in args.compress_ratios if r > 0]
        self._min_prompt_tokens = min(nonzero) if nonzero else 1

        model = DeepSeekV4(args)
        ckpt = torch.load(
            self.config.checkpoint_path,
            map_location=str(self.device),
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        self._step = ckpt.get("step", 0)
        self.model = model

    def build_tokenizer(self):
        """Load BPE tokenizer."""
        self.tokenizer = BPETokenizer.from_file(
            self.config.tokenizer_vocab,
            self.config.tokenizer_merges,
            special_tokens=SPECIAL_TOKENS,
        )

    def _capture_buffers(self):
        """Snapshot all model buffers after initialization.

        Buffers (kv_cache, kv_state, score_state, etc.) are registered
        as persistent=False — they start at zeros / -inf and are never
        saved to checkpoints.  We snapshot them here so we can restore
        them between generations without touching model code.
        """
        self._init_buffers = {
            name: buf.clone() for name, buf in self.model.named_buffers()
        }

    # ------------------------------------------------------------------
    # KV Cache
    # ------------------------------------------------------------------

    def reset_kv_cache(self):
        """Restore all internal KV-cache buffers to initial (empty) state.

        Called automatically at the start of each ``generate()`` call.
        """
        for name, buf in self.model.named_buffers():
            buf.copy_(self._init_buffers[name])

    # ------------------------------------------------------------------
    # Pre / Post processing  (override in subclasses)
    # ------------------------------------------------------------------

    def preprocess(self, prompt: str):
        """Convert raw prompt to token ids + EOS id.

        Returns:
            (token_ids: list[int], eos_id: int | None)
        """
        ids = self.tokenizer.encode(prompt)
        eos_id = self.tokenizer._special_token_to_id.get(
            self.config.eos_token.encode(), None
        )
        return ids, eos_id

    def postprocess(self, ids: list) -> str:
        """Convert token ids to text."""
        return self.tokenizer.decode(ids)

    # ------------------------------------------------------------------
    # Sampling  (override for custom strategies)
    # ------------------------------------------------------------------

    def sample(self, logits: torch.Tensor, generated_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Sample next token.  Default: top-k temperature sampling with rep penalty.

        Override for custom strategies (e.g. beam search).
        """
        if self.config.temperature <= 0:
            return sample_greedy(logits)
        if self.config.top_p < 1.0:
            return sample_top_p(logits, self.config.temperature, self.config.top_p,
                                generated_ids, self.config.repetition_penalty)
        return sample_top_k(logits, self.config.temperature, self.config.top_k,
                            generated_ids, self.config.repetition_penalty)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[str]]) -> Union[str, List[str]]:
        """Generate text from one or more prompts.

        For a list of prompts, each prompt is generated sequentially
        (the model's absolute position encoding and shared start_pos
        make batched forward-passes with variable-length prompts
        impractical without model changes).  KV cache is reset between
        prompts.

        Args:
            prompt: A single prompt string or a list of prompt strings.

        Returns:
            Generated text (str) or list of generated texts (list[str]).
        """
        if isinstance(prompt, str):
            self.reset_kv_cache()
            return self._generate_one(prompt)
        else:
            results = []
            for p in prompt:
                self.reset_kv_cache()
                results.append(self._generate_one(p))
            return results

    def _generate_one(self, prompt: str) -> str:
        """Generate for a single prompt (KV cache must be reset by caller)."""
        token_ids, eos_id = self.preprocess(prompt)
        prompt_len = len(token_ids)

        if prompt_len == 0:
            return ""
        if prompt_len < self._min_prompt_tokens:
            raise ValueError(
                f"Prompt too short: {prompt_len} tokens < "
                f"{self._min_prompt_tokens} (minimum for compress_ratio). "
                f"Use a longer prompt."
            )

        prompt_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )

        # ---- prefill ----
        ntp, _, _ = self.model(prompt_tensor, start_pos=0)
        generated: List[int] = []
        next_token = self.sample(ntp[:, -1, :], generated)
        generated.append(next_token.item())
        current = next_token  # (1, 1)

        # ---- autoregressive decode ----
        for pos in range(prompt_len, prompt_len + self.config.max_new_tokens - 1):
            ntp, _, _ = self.model(current, start_pos=pos)
            next_token = self.sample(ntp[:, -1, :], generated)
            token_id = next_token.item()
            generated.append(token_id)
            current = next_token
            if eos_id is not None and token_id == eos_id:
                break

        return self.postprocess(token_ids + generated)


# ======================================================================
# ChatEngine
# ======================================================================


class ChatEngine(InferenceEngine):
    """Inference engine with chat-template support.

    Wraps user prompts in ``<|user|>\\n...\\n<|assistant|>\\n`` format
    and extracts only the assistant's reply from the generated output.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self._user_tag = "<|user|>"
        self._assistant_tag = "<|assistant|>"

    def preprocess(self, prompt: str):
        """Insert chat template before tokenization."""
        formatted = f"{self._user_tag}\n{prompt}\n{self._assistant_tag}\n"
        return super().preprocess(formatted)

    def postprocess(self, ids: list) -> str:
        """Return only the text after the first <|assistant|> marker."""
        full_text = super().postprocess(ids)
        marker = f"{self._assistant_tag}\n"
        idx = full_text.find(marker)
        if idx != -1:
            return full_text[idx + len(marker):]
        return full_text
