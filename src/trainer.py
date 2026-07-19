import gc
import math
import os
import time
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import MODEL_CONFIGS
from src.deepseek import DeepSeekV4, DSArgs
from src.dataset import (
    MixedSFTDataset,
    PretrainDataset,
    SFTDataset,
    GRPOOffPolicyDataset,
    DPODataset,
    SimPODataset,
    WeightedSFTDataset,
)
from src.loss import (
    cross_entropy,
    indexer_kl_loss,
    cross_entropy_masked,
    weighted_sft_loss,
    grpo_loss,
    dpo_loss,
    simpo_loss,
)
from src.optimizer import (
    Muon,
    AdamW,
    group_params,
    get_indexer_params,
    grad_clip,
    cosine_annealing_lr_schedule,
)

# ======================================================================
# TrainerArgs dataclasses
# ======================================================================


@dataclass
class TrainerArgs:
    # Output & checkpointing
    output_dir: str = "checkpoints/pretrain"
    total_steps: int = 500000
    ckpt_every: int = 5000
    val_every: int = 500
    log_every: int = 10
    keep_last_ckpt: int = 5

    # Training
    batch_size: int = 4
    max_seq_len: int = 1024
    grad_accum: int = 1
    max_grad_norm: float = 1.0

    # Reproducibility & logging
    seed: int = 42
    wandb_project: str = ""

    # Resume
    resume: str = ""
    resume_step: bool = True


@dataclass
class PretrainTrainerArgs(TrainerArgs):
    # Model
    config_name: str = "small"

    # Data
    data_train: str = ""
    data_val: str = ""

    # Optimization
    lr: float = 2.7e-4
    lr_min: float = 2.7e-5
    warmup_steps: int = 2000

    # Data loading
    num_workers: int = 0


@dataclass
class SFTTrainerArgs(TrainerArgs):
    # Model
    config_name: str = "small"
    base_model_path: str = ""

    # Data
    data_train: str = ""
    data_val: str = ""

    # Data mixing: mix another SFT dataset into training at the given ratio
    data_mix: str = ""
    mix_ratio: float = 0.0

    # Optimization
    lr: float = 2.7e-4
    lr_min: float = 2.7e-5
    warmup_steps: int = 2000

    # Data loading
    num_workers: int = 0


@dataclass
class GRPOOffPolicyTrainerArgs(TrainerArgs):
    # Model
    config_name: str = "small"
    base_model_path: str = ""
    ref_model_ckpt_path: str = ""

    # Data
    data_train: str = ""
    data_val: str = ""
    group_size: int = 4
    batch_size: int = 1

    # Optimization
    lr: float = 2.7e-4
    lr_min: float = 2.7e-5
    warmup_steps: int = 2000
    kl_penalty: float = 0.05
    clip_eps: float = 0.2

    # Data loading
    num_workers: int = 0


@dataclass
class DPOTrainerArgs(TrainerArgs):
    # Model
    config_name: str = "small"
    base_model_path: str = ""
    ref_model_ckpt_path: str = ""

    # Data
    data_train: str = ""
    data_val: str = ""
    group_size: int = 4
    batch_size: int = 1

    # Optimization
    lr: float = 2.7e-4
    lr_min: float = 2.7e-5
    warmup_steps: int = 2000
    beta: float = 0.1

    # Data loading
    num_workers: int = 0


@dataclass
class SimPOTrainerArgs(TrainerArgs):
    config_name: str = "small"
    base_model_path: str = ""

    data_train: str = ""
    data_val: str = ""

    lr: float = 2.7e-4
    lr_min: float = 2.7e-5
    warmup_steps: int = 2000

    beta: float = 1.0
    gamma: float = 0.1
    lambda_simpo: float = 1.0

    num_workers: int = 0


@dataclass
class WeightedSFTTrainerArgs(TrainerArgs):
    config_name: str = "small"
    base_model_path: str = ""

    data_train: str = ""
    data_val: str = ""

    lr: float = 2.7e-4
    lr_min: float = 2.7e-5
    warmup_steps: int = 2000
    weight_min_start: float = 0.7
    weight_min_end: float = 0.2

    num_workers: int = 0

    resume_step: bool = False


# ======================================================================
# Trainer
# ======================================================================


class Trainer:
    """Base trainer with DDP, checkpointing, logging, and training loop.

    Subclass and override the hook methods:
        build_model_and_optimizers()
        build_dataloaders()
        train_step(batch, is_last_micro) -> dict
        validate() -> dict
        get_lr(step) -> float
        _optimizer_step() -> float  (optional, default clips + steps all opts)
    """

    def __init__(self, args: TrainerArgs):
        # Copy all dataclass fields to self
        for field_name in args.__dataclass_fields__:
            setattr(self, field_name, getattr(args, field_name))

        self.rank, self.world_size, self.local_rank = self._setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.is_main = self.rank == 0  # rank 0的设备才能保存ckpt

        self.model = None
        self.optimizers: list = []
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.start_step = 0
        self.best_val_loss = float("inf")
        self._current_step = 0
        self._step_start_time = 0.0
        self._train_start_time = 0.0

        self.wandb_run = None
        self.log_file_path = os.path.join(self.output_dir, "train.log")

    # DDP
    def _setup_distributed(self) -> Tuple[int, int, int]:
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            return 0, 1, 0
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank

    # -------------------
    # ckpt 相关
    def save_checkpoint(self, path: str, step: int, extra: Optional[Dict] = None):
        if self.is_main:
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizers_state_dict": [opt.state_dict() for opt in self.optimizers],
                "best_val_loss": self.best_val_loss,
                "torch_rng_state": torch.get_rng_state().cpu().clone(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state().cpu().clone()
                    if torch.cuda.is_available()
                    else None
                ),
                "extra": extra or {},
            }
            torch.save(ckpt, path)

        if self.world_size > 1:
            dist.barrier()

    def load_checkpoint(self, path: str, resume_step: bool = True) -> Tuple[int, Dict]:
        state = torch.load(
            path, map_location=f"cuda:{self.local_rank}", weights_only=False
        )
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(state["model_state_dict"])

        for opt, sd in zip(self.optimizers, state["optimizers_state_dict"]):
            opt.load_state_dict(sd)
        torch.set_rng_state(state["torch_rng_state"].cpu())

        if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu())
        extra = state.get("extra", {})

        if resume_step:
            return state["step"] + 1, extra
        else:
            return 0, {}

    def _cleanup_checkpoints(self):
        if not self.is_main:
            return
        ckpts = sorted(Path(self.output_dir).glob("ckpt_[0-9]*.pt"))
        for old in ckpts[: -self.keep_last_ckpt]:
            old.unlink(missing_ok=True)

    # ----------------
    # 日志相关
    # 顺便加上wandb
    def _log(self, message: str):
        if not self.is_main:
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(message)
        with open(self.log_file_path, "a") as f:
            f.write(f"[{now}] {message}\n")

    def _log_step(
        self,
        step: int,
        losses: Dict[str, float],
        lr: float,
        grad_norm: float,
        step_time: float,
    ):
        elapsed = time.time() - self._train_start_time
        steps_done = step - self.start_step + 1
        eta_steps = self.total_steps - step
        eta_seconds = eta_steps * (elapsed / max(1, steps_done))
        eta_days = eta_seconds / 86400

        tokens_this_step = (
            self.world_size * self.batch_size * self.grad_accum * self.max_seq_len
        )
        tok_per_sec = tokens_this_step / step_time if step_time > 0 else 0
        mem_mb = torch.cuda.max_memory_allocated(self.local_rank) / 1024**2

        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in sorted(losses.items()))
        self._log(
            f"Step {step:>6d}/{self.total_steps} | "
            f"{metrics_str} | "
            f"lr: {lr:.2e} | "
            f"grad_norm: {grad_norm:.4f} | "
            f"tok/s: {tok_per_sec:,.0f} | "
            f"mem: {mem_mb:.0f}MB | "
            f"ETA: {eta_days:.1f}d"
        )

        if self.wandb_run:
            wandb_metrics = {f"train/{k}": v for k, v in losses.items()}
            wandb_metrics.update(
                {
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mem_mb": mem_mb,
                    "train/step": step,
                }
            )
            self.wandb_run.log(wandb_metrics, step=step)

    def _set_lr(self, lr: float):
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr

    def _infinite_train_iter(self):
        while True:
            for batch in self.train_loader:
                yield batch

    def _init_wandb(self):
        if not self.wandb_project or not self.is_main:
            return
        import wandb

        self.wandb_run = wandb.init(
            project=self.wandb_project,
            name=os.path.basename(self.output_dir.rstrip("/")),
        )

    # ----------------
    # 具体的训练流程交给子类自行实现

    def build_model_and_optimizers(self):
        raise NotImplementedError

    def build_dataloaders(self):
        raise NotImplementedError

    def train_step(self, batch, is_last_micro: bool) -> Dict[str, float]:
        raise NotImplementedError

    def validate(self) -> Dict[str, float]:
        return {}

    def get_lr(self, step: int) -> float:
        raise NotImplementedError

    def _optimizer_step(self) -> float:
        if self.max_grad_norm is not None:
            gn = grad_clip(self.model.parameters(), self.max_grad_norm)
        else:
            gn = 0.0
        for opt in self.optimizers:
            opt.step()
        self.model.zero_grad()
        return gn

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        if self.is_main:
            os.makedirs(self.output_dir, exist_ok=True)

        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(self.seed + self.rank)
        torch.cuda.manual_seed(self.seed + self.rank)

        self._log(
            f"Trainer init: rank={self.rank}, world_size={self.world_size}, "
            f"device=cuda:{self.local_rank}"
        )

        self.build_model_and_optimizers()
        self.build_dataloaders()

        self._init_wandb()

        if self.resume:
            self.start_step, extra = self.load_checkpoint(self.resume, self.resume_step)
            self.best_val_loss = extra.get("best_val_loss", float("inf"))
            if self.is_main:
                self._log(
                    f"Resumed from {self.resume} at step {self.start_step}, "
                    f"best_val_loss={self.best_val_loss:.4f}"
                )

        train_iter = self._infinite_train_iter()
        self._train_start_time = time.time()
        self._current_step = self.start_step

        if self.is_main:
            self._log(f"Training from step {self.start_step} to {self.total_steps}")

        for step in range(self.start_step, self.total_steps):
            # 正式开始训练循环
            self._current_step = step
            self._step_start_time = time.time()

            self._set_lr(self.get_lr(step))

            accum_losses: Dict[str, float] = {}
            for micro in range(self.grad_accum):
                batch = next(train_iter)
                is_last = micro == self.grad_accum - 1
                losses = self.train_step(batch, is_last)
                for k, v in losses.items():
                    accum_losses[k] = accum_losses.get(k, 0.0) + v
            for k in accum_losses:
                accum_losses[k] /= self.grad_accum

            if not all(torch.isfinite(torch.tensor(v)) for v in accum_losses.values()):
                details = " ".join(f"{k}={v:.4f}" for k, v in accum_losses.items())
                self._log(f"Step {step}: NaN/Inf — losses: {details}")
                self.save_checkpoint(
                    os.path.join(self.output_dir, f"ckpt_nan_{step:07d}.pt"),
                    step,
                    extra={"best_val_loss": self.best_val_loss},
                )
                raise RuntimeError(f"Loss is NaN/Inf at step {step}")

            grad_norm = self._optimizer_step()

            # 定期触发 GC 回收 autograd 图循环引用
            if step % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            step_time = time.time() - self._step_start_time

            if step % 50 == 0 or step == self.start_step:
                self._log(
                    f"RSS_TRACE step={step} rss_mb={psutil.Process().memory_info().rss/1024**2:.0f}"
                )

            if step % self.log_every == 0:
                self._log_step(
                    step, accum_losses, self.get_lr(step), grad_norm, step_time
                )

            if self.val_loader and step % self.val_every == 0 and step > 0:
                val_metrics = self.validate()
                if self.is_main:
                    self._log(
                        f"Validation step {step}: "
                        f"lm={val_metrics.get('val/lm_loss', 0):.4f} "
                        f"ntp={val_metrics.get('val/ntp_loss', 0):.4f} "
                        f"kl={val_metrics.get('val/kl_loss', 0):.4f}"
                    )
                    if self.wandb_run:
                        self.wandb_run.log(val_metrics, step=step)

                val_lm = val_metrics.get("val/lm_loss", float("inf"))
                if val_lm <= self.best_val_loss:
                    self.best_val_loss = val_lm
                    self.save_checkpoint(
                        os.path.join(self.output_dir, "ckpt_best.pt"),
                        step,
                        extra={"best_val_loss": self.best_val_loss},
                    )

            if step % self.ckpt_every == 0 and step > 0:
                self.save_checkpoint(
                    os.path.join(self.output_dir, f"ckpt_{step:07d}.pt"),
                    step,
                    extra={"best_val_loss": self.best_val_loss},
                )
                self._cleanup_checkpoints()

        self.save_checkpoint(
            os.path.join(self.output_dir, "ckpt_final.pt"),
            self.total_steps,
            extra={"best_val_loss": self.best_val_loss},
        )

        if self.is_main:
            elapsed = time.time() - self._train_start_time
            self._log(f"Training complete. Total time: {elapsed / 3600:.1f}h")

        if self.wandb_run:
            self.wandb_run.finish()

        if self.world_size > 1:
            dist.destroy_process_group()


class PretrainTrainer(Trainer):
    """Trainer for language model pretraining with DeepSeekV4."""

    def __init__(self, args: PretrainTrainerArgs):
        super().__init__(args)
        self._model_args: Optional[DSArgs] = None

    def build_model_and_optimizers(self):
        cfg = MODEL_CONFIGS[self.config_name].copy()
        cfg["max_seq_len"] = self.max_seq_len
        self.max_seq_len = cfg["max_seq_len"]

        base_fields = DSArgs.__dataclass_fields__
        args = DSArgs(**{k: v for k, v in cfg.items() if k in base_fields})

        needed = args.n_layer + args.n_mtp_layer
        if len(args.compress_ratios) < needed:
            args.compress_ratios = args.compress_ratios + tuple(
                [0] * (needed - len(args.compress_ratios))
            )  # MTP layer当时没有写compress_ratios，当时只写了前面block的，打个补丁

        self._model_args = args

        model = DeepSeekV4(args)
        model.train()
        model = model.to(self.local_rank)

        muon_p, adamw_p = group_params(model)
        idx_p = get_indexer_params(model)

        muon_opt = Muon(muon_p, lr=self.lr, momentum=0.95, weight_decay=0.1)
        adamw_opt = AdamW(adamw_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)
        idx_opt = AdamW(idx_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)

        self.optimizers = [muon_opt, adamw_opt, idx_opt]

        if self.world_size > 1:
            self.model = DDP(
                model, device_ids=[self.local_rank], find_unused_parameters=True
            )
        else:
            self.model = model

        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self._log(
                f"Model: {self.config_name} | "
                f"Params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable | "
                f"d_model={args.d_model}, n_layer={args.n_layer}, "
                f"n_experts={args.n_experts}, seq_len={args.max_seq_len}"
            )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def build_dataloaders(self):
        if not self.data_train:
            raise ValueError("--data-train is required")

        train_dataset = PretrainDataset(self.data_train, self.max_seq_len)
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self.data_val:
            val_dataset = PretrainDataset(self.data_val, self.max_seq_len)
            val_sampler = None
            if self.world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_loader = None

        if self.is_main:
            n_tokens = train_dataset.num_tokens
            self._log(
                f"Train data: {self.data_train} | "
                f"{n_tokens/1e6:.0f}M tokens, "
                f"{len(train_dataset):,} samples | "
                f"batch={self.batch_size}, seq_len={self.max_seq_len}"
            )
            if self.val_loader:
                self._log(
                    f"Val data: {self.data_val} | "
                    f"{val_dataset.num_tokens/1e6:.0f}M tokens, "
                    f"{len(val_dataset):,} samples"
                )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch, is_last_micro: bool) -> Dict[str, float]:
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.local_rank, non_blocking=True)
        target_ids = target_ids.to(self.local_rank, non_blocking=True)

        ntp, mtp_list, idx_data = self.model(input_ids)

        ntp_loss = cross_entropy(target_ids, ntp)
        mtp_loss = sum(
            cross_entropy(target_ids[:, i + 1 :], m[:, : -(i + 1)])
            for i, m in enumerate(mtp_list)
        )
        kl_loss = sum(
            indexer_kl_loss(iscore, idx, wc.detach() if wc is not None else None)
            for (iscore, wc, idx) in idx_data
        )

        lm_loss = ntp_loss + 0.3 * mtp_loss
        total_loss = (lm_loss + 0.5 * kl_loss) / self.grad_accum

        if self.world_size > 1 and not is_last_micro:
            with self.model.no_sync():
                total_loss.backward()
        else:
            total_loss.backward()

        return {
            "ntp": ntp_loss.item(),
            "mtp": mtp_loss.item(),
            "kl": kl_loss.item(),
            "lm": lm_loss.item(),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, max_val_batches: int = 100) -> Dict[str, float]:
        if not self.val_loader:
            return {}

        self.model.eval()

        total_ntp = 0.0
        total_mtp = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch in self.val_loader:
            if num_batches >= max_val_batches:
                break

            input_ids, target_ids = batch
            input_ids = input_ids.to(self.local_rank, non_blocking=True)
            target_ids = target_ids.to(self.local_rank, non_blocking=True)

            ntp, mtp_list, idx_data = self.model(input_ids)

            total_ntp += cross_entropy(target_ids, ntp).item()
            total_mtp += sum(
                cross_entropy(target_ids[:, i + 1 :], m[:, : -(i + 1)]).item()
                for i, m in enumerate(mtp_list)
            )
            total_kl += sum(
                indexer_kl_loss(iscore, idx, wc).item()
                for (iscore, wc, idx) in idx_data
            )
            num_batches += 1

        self.model.train()

        avg_ntp = total_ntp / num_batches
        avg_mtp = total_mtp / num_batches
        avg_kl = total_kl / num_batches
        avg_lm = avg_ntp + 0.3 * avg_mtp

        if self.world_size > 1:
            losses_t = torch.tensor(
                [avg_ntp, avg_mtp, avg_kl, avg_lm],
                device=self.local_rank,
            )
            dist.all_reduce(losses_t, op=dist.ReduceOp.SUM)
            losses_t /= self.world_size
            avg_ntp, avg_mtp, avg_kl, avg_lm = losses_t.tolist()

        return {
            "val/ntp_loss": avg_ntp,
            "val/mtp_loss": avg_mtp,
            "val/kl_loss": avg_kl,
            "val/lm_loss": avg_lm,
        }

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def get_lr(self, step: int) -> float:
        return cosine_annealing_lr_schedule(
            step, self.lr, self.lr_min, self.warmup_steps, self.total_steps
        )

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> float:
        gn = grad_clip(self.model.parameters(), self.max_grad_norm)
        for opt in self.optimizers:
            opt.step()
        self.model.zero_grad()
        return gn


class SFTTrainer(Trainer):
    """
    SFT部分基本上和pretrain保持一致
    """

    def __init__(self, args: SFTTrainerArgs):
        super().__init__(args)
        self._model_args: Optional[DSArgs] = None
        self.base_model_ckpt_path = args.base_model_path

    def build_model_and_optimizers(self):
        cfg = MODEL_CONFIGS[self.config_name].copy()
        cfg["max_seq_len"] = self.max_seq_len
        self.max_seq_len = cfg["max_seq_len"]

        base_fields = DSArgs.__dataclass_fields__
        args = DSArgs(**{k: v for k, v in cfg.items() if k in base_fields})

        needed = args.n_layer + args.n_mtp_layer
        if len(args.compress_ratios) < needed:
            args.compress_ratios = args.compress_ratios + tuple(
                [0] * (needed - len(args.compress_ratios))
            )  # MTP layer当时没有写compress_ratios，当时只写了前面block的，打个补丁

        self._model_args = args

        model = DeepSeekV4(args)
        model.train()
        model = model.to(self.local_rank)

        # 和pretrain相比只多出来加载权重
        ckpt = torch.load(
            self.base_model_ckpt_path,
            map_location=f"cuda:{self.local_rank}",
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        muon_p, adamw_p = group_params(model)
        idx_p = get_indexer_params(model)

        muon_opt = Muon(muon_p, lr=self.lr, momentum=0.95, weight_decay=0.1)
        adamw_opt = AdamW(adamw_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)
        idx_opt = AdamW(idx_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)

        self.optimizers = [muon_opt, adamw_opt, idx_opt]

        if self.world_size > 1:
            self.model = DDP(
                model, device_ids=[self.local_rank], find_unused_parameters=True
            )
        else:
            self.model = model

        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self._log(
                f"Model: {self.config_name} | "
                f"Params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable | "
                f"d_model={args.d_model}, n_layer={args.n_layer}, "
                f"n_experts={args.n_experts}, seq_len={args.max_seq_len}"
            )

    def build_dataloaders(self):
        if not self.data_train:
            raise ValueError("--data-train is required")

        if self.data_mix:
            train_dataset = MixedSFTDataset(
                self.data_mix, self.data_train, self.mix_ratio
            )
        else:
            train_dataset = SFTDataset(self.data_train)
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self.data_val:
            val_dataset = SFTDataset(self.data_val)
            val_sampler = None
            if self.world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_loader = None

        if self.is_main:
            n_tokens = train_dataset.num_tokens
            self._log(
                f"Train data: {self.data_train} | "
                f"{n_tokens/1e6:.0f}M tokens, "
                f"{len(train_dataset):,} samples | "
                f"batch={self.batch_size}, seq_len={self.max_seq_len}"
            )
            if self.val_loader:
                self._log(
                    f"Val data: {self.data_val} | "
                    f"{val_dataset.num_tokens/1e6:.0f}M tokens, "
                    f"{len(val_dataset):,} samples"
                )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch, is_last_micro: bool) -> Dict[str, float]:
        input_ids, target_ids, masks = batch
        input_ids = input_ids.to(self.local_rank, non_blocking=True)
        target_ids = target_ids.to(self.local_rank, non_blocking=True)
        masks = masks.to(self.local_rank, non_blocking=True)

        ntp, mtp_list, idx_data = self.model(input_ids)

        ntp_loss = cross_entropy_masked(target_ids, ntp, masks)
        mtp_loss = sum(
            cross_entropy_masked(
                target_ids[:, i + 1 :], m[:, : -(i + 1)], masks[:, : -(i + 1)]
            )
            for i, m in enumerate(mtp_list)
        )
        kl_loss = sum(
            indexer_kl_loss(iscore, idx, wc.detach() if wc is not None else None)
            for (iscore, wc, idx) in idx_data
        )

        lm_loss = ntp_loss + 0.3 * mtp_loss
        total_loss = (lm_loss + 0.5 * kl_loss) / self.grad_accum

        if self.world_size > 1 and not is_last_micro:
            with self.model.no_sync():
                total_loss.backward()
        else:
            total_loss.backward()

        return {
            "ntp": ntp_loss.item(),
            "mtp": mtp_loss.item(),
            "kl": kl_loss.item(),
            "lm": lm_loss.item(),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, max_val_batches: int = 100) -> Dict[str, float]:
        if not self.val_loader:
            return {}

        self.model.eval()

        total_ntp = 0.0
        total_mtp = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch in self.val_loader:
            if num_batches >= max_val_batches:
                break

            input_ids, target_ids, masks = batch
            input_ids = input_ids.to(self.local_rank, non_blocking=True)
            target_ids = target_ids.to(self.local_rank, non_blocking=True)
            masks = masks.to(self.local_rank, non_blocking=True)

            ntp, mtp_list, idx_data = self.model(input_ids)

            total_ntp += cross_entropy_masked(target_ids, ntp, masks).item()
            total_mtp += sum(
                cross_entropy_masked(
                    target_ids[:, i + 1 :], m[:, : -(i + 1)], masks[:, : -(i + 1)]
                ).item()
                for i, m in enumerate(mtp_list)
            )
            total_kl += sum(
                indexer_kl_loss(iscore, idx, wc).item()
                for (iscore, wc, idx) in idx_data
            )
            num_batches += 1

        self.model.train()

        avg_ntp = total_ntp / num_batches
        avg_mtp = total_mtp / num_batches
        avg_kl = total_kl / num_batches
        avg_lm = avg_ntp + 0.3 * avg_mtp

        if self.world_size > 1:
            losses_t = torch.tensor(
                [avg_ntp, avg_mtp, avg_kl, avg_lm],
                device=self.local_rank,
            )
            dist.all_reduce(losses_t, op=dist.ReduceOp.SUM)
            losses_t /= self.world_size
            avg_ntp, avg_mtp, avg_kl, avg_lm = losses_t.tolist()

        return {
            "val/ntp_loss": avg_ntp,
            "val/mtp_loss": avg_mtp,
            "val/kl_loss": avg_kl,
            "val/lm_loss": avg_lm,
        }

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def get_lr(self, step: int) -> float:
        return cosine_annealing_lr_schedule(
            step, self.lr, self.lr_min, self.warmup_steps, self.total_steps
        )

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> float:
        gn = grad_clip(self.model.parameters(), self.max_grad_norm)
        for opt in self.optimizers:
            opt.step()
        self.model.zero_grad()
        return gn


class GRPOOffPolicyTrainer(Trainer):
    def __init__(self, args: GRPOOffPolicyTrainerArgs):
        super().__init__(args)
        self._model_args: Optional[DSArgs] = None
        self.base_model_ckpt_path = args.base_model_path
        self.ref_model_ckpt_path = args.ref_model_ckpt_path
        self.group_size = args.group_size

    def build_model_and_optimizers(self):
        cfg = MODEL_CONFIGS[self.config_name].copy()
        cfg["max_seq_len"] = self.max_seq_len
        self.max_seq_len = cfg["max_seq_len"]

        base_fields = DSArgs.__dataclass_fields__
        args = DSArgs(**{k: v for k, v in cfg.items() if k in base_fields})

        needed = args.n_layer + args.n_mtp_layer
        if len(args.compress_ratios) < needed:
            args.compress_ratios = args.compress_ratios + tuple(
                [0] * (needed - len(args.compress_ratios))
            )  # MTP layer当时没有写compress_ratios，当时只写了前面block的，打个补丁

        self._model_args = args

        model = DeepSeekV4(args)
        model.train()
        model = model.to(self.local_rank)

        ref_model = DeepSeekV4(args)
        ref_model = ref_model.to(self.local_rank)

        # 和pretrain相比只多出来加载权重
        ckpt = torch.load(
            self.base_model_ckpt_path,
            map_location=f"cuda:{self.local_rank}",
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        ref_model.load_state_dict(ckpt["model_state_dict"])
        ref_model.eval()
        ref_model.requires_grad_(False)

        muon_p, adamw_p = group_params(model)
        idx_p = get_indexer_params(model)

        muon_opt = Muon(muon_p, lr=self.lr, momentum=0.95, weight_decay=0.1)
        adamw_opt = AdamW(adamw_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)
        idx_opt = AdamW(idx_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)

        self.optimizers = [muon_opt, adamw_opt, idx_opt]

        if self.world_size > 1:
            self.model = DDP(
                model, device_ids=[self.local_rank], find_unused_parameters=True
            )
        else:
            self.model = model
        self.ref_model = ref_model

        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self._log(
                f"Model: {self.config_name} | "
                f"Params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable | "
                f"d_model={args.d_model}, n_layer={args.n_layer}, "
                f"n_experts={args.n_experts}, seq_len={args.max_seq_len}"
            )

    def build_dataloaders(self):
        if not self.data_train:
            raise ValueError("--data-train is required")

        train_dataset = GRPOOffPolicyDataset(
            self.data_train, group_size=self.group_size
        )
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self.data_val:
            val_dataset = GRPOOffPolicyDataset(
                self.data_val, group_size=self.group_size
            )
            val_sampler = None
            if self.world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_loader = None

        if self.is_main:
            self._log(
                f"Train data: {self.data_train} | "
                f"{len(train_dataset):,} samples | "
                f"batch={self.batch_size}, group_size={self.group_size}, "
                f"seq_len={self.max_seq_len}"
            )
            if self.val_loader:
                self._log(
                    f"Val data: {self.data_val} | " f"{len(val_dataset):,} samples"
                )

    def train_step(self, batch, is_last_micro):
        # input_ids: (batch_size, group_size, seq_len)
        # 和SFT & Pretrain的数据格式不同，前面两个直接返回了input_ids 和 target ids
        # 其长度都是1023
        # 这里input_ids就是1024，所以要按需取用
        input_ids, masks, scores = batch
        bs, gs, _ = input_ids.shape
        input_ids = input_ids.reshape(bs * gs, -1)

        input_ids = input_ids.to(self.local_rank, non_blocking=True)

        ntp_working, _, _ = self.model(input_ids[..., :-1].contiguous())
        with torch.no_grad():
            ntp_ref, _, _ = self.ref_model(input_ids[..., :-1].contiguous())

        ntp_working = ntp_working.reshape(bs, gs, -1, ntp_working.shape[-1])
        ntp_ref = ntp_ref.reshape(bs, gs, -1, ntp_ref.shape[-1])
        input_ids = input_ids.reshape(bs, gs, -1)  # (bs, gs, 1024)
        masks = masks.to(self.local_rank, non_blocking=True)
        scores = scores.to(self.local_rank, non_blocking=True).reshape(bs, gs)
        metrics = grpo_loss(
            logits_working=ntp_working,
            logits_ref=ntp_ref,
            token_ids=input_ids[..., 1:],
            scores=scores,
            masks=masks[..., :-1],
            beta=self.kl_penalty,
            eps=self.clip_eps,
        )

        # ---- additional monitoring ----
        log_probs = torch.log_softmax(ntp_working, dim=-1)
        probs = torch.exp(log_probs)
        entropy_per_tok = -(probs * log_probs).sum(dim=-1)
        mask_f = masks[..., :-1].float()
        entropy = (entropy_per_tok * mask_f).sum() / mask_f.sum().clamp(min=1)

        reward_margin = (scores.max(dim=-1).values - scores.min(dim=-1).values).mean()
        adv = metrics["advantage"]

        total_loss = metrics["total_loss"] / self.grad_accum

        if self.world_size > 1 and not is_last_micro:
            with self.model.no_sync():
                total_loss.backward()
        else:
            total_loss.backward()

        return {
            "ntp": metrics["total_loss"].item(),
            "lm": metrics["total_loss"].item(),
            "policy_loss": metrics["policy_loss"].item(),
            "kl": metrics["kl"].item(),
            "ratio_mean": metrics["ratio_mean"].item(),
            "ratio_std": metrics["ratio_std"].item(),
            "ratio_max": metrics["ratio_max"].item(),
            "clip_frac": metrics["clip_fraction"].item(),
            "entropy": entropy.item(),
            "reward_margin": reward_margin.item(),
            "adv_mean": adv.mean().item(),
            "adv_std": adv.std().item(),
        }

    @torch.no_grad()
    def validate(self, max_val_batches: int = 100) -> Dict[str, float]:
        if not self.val_loader:
            return {}

        self.model.eval()

        accum: Dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            if num_batches >= max_val_batches:
                break

            input_ids, masks, scores = batch
            bs, gs, _ = input_ids.shape
            input_ids = input_ids.reshape(bs * gs, -1)

            input_ids = input_ids.to(self.local_rank, non_blocking=True)

            ntp_working, _, _ = self.model(input_ids[..., :-1].contiguous())
            ntp_ref, _, _ = self.ref_model(input_ids[..., :-1].contiguous())

            ntp_working = ntp_working.reshape(bs, gs, -1, ntp_working.shape[-1])
            ntp_ref = ntp_ref.reshape(bs, gs, -1, ntp_ref.shape[-1])
            input_ids = input_ids.reshape(bs, gs, -1)
            masks = masks.to(self.local_rank, non_blocking=True)
            scores = scores.to(self.local_rank, non_blocking=True).reshape(bs, gs)
            metrics = grpo_loss(
                logits_working=ntp_working,
                logits_ref=ntp_ref,
                token_ids=input_ids[..., 1:],
                scores=scores,
                masks=masks[..., :-1],
                beta=self.kl_penalty,
                eps=self.clip_eps,
            )

            mask_f = masks[..., :-1].float()
            response_logp = (metrics["logp_w"] * mask_f).sum(dim=-1)  # (bs, gs)
            response_len = mask_f.sum(dim=-1).clamp(min=1)
            response_score = response_logp / response_len  # length-normalized

            # ranking metrics
            pred = response_score.argmax(dim=-1)  # (bs,)
            gt = scores.argmax(dim=-1)  # (bs,)
            top1 = (pred == gt).float().mean()

            score_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)  # (bs, gs, gs)
            model_diff = response_score.unsqueeze(-1) - response_score.unsqueeze(-2)
            pair_mask = score_diff > 0
            pair_correct = ((model_diff > 0) & pair_mask).float().sum(dim=(-1, -2))
            pair_total = pair_mask.float().sum(dim=(-1, -2))
            pairwise_acc = (pair_correct / pair_total.clamp(min=1)).mean()

            def _spearman(x, y):
                rk = lambda t: t.argsort(dim=-1).argsort(dim=-1).float()
                xr, yr = rk(x), rk(y)
                mx, my = xr.mean(dim=-1, keepdim=True), yr.mean(dim=-1, keepdim=True)
                num = ((xr - mx) * (yr - my)).sum(dim=-1)
                den = ((xr - mx) ** 2).sum(dim=-1).sqrt() * ((yr - my) ** 2).sum(
                    dim=-1
                ).sqrt() + 1e-8
                return (num / den).mean()

            spearman = _spearman(response_score, scores)

            rw_score = (response_score * scores).mean()

            for k in [
                "total_loss",
                "policy_loss",
                "kl",
                "ratio_mean",
                "ratio_std",
                "ratio_max",
                "clip_fraction",
            ]:
                accum[k] = accum.get(k, 0.0) + metrics[k].item()
            accum["response_logp"] = (
                accum.get("response_logp", 0.0) + response_logp.mean().item()
            )
            accum["top1"] = accum.get("top1", 0.0) + top1.item()
            accum["pairwise"] = accum.get("pairwise", 0.0) + pairwise_acc.item()
            accum["spearman"] = accum.get("spearman", 0.0) + spearman.item()
            accum["rw_score"] = accum.get("rw_score", 0.0) + rw_score.item()
            num_batches += 1

        self.model.train()

        for k in accum:
            accum[k] /= num_batches

        if self.world_size > 1:
            keys = sorted(accum.keys())
            vals = [accum[k] for k in keys]
            t = torch.tensor(vals, device=self.local_rank)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= self.world_size
            for k, v in zip(keys, t.tolist()):
                accum[k] = v

        result = {f"val/{k}": v for k, v in accum.items()}
        result["val/lm_loss"] = accum.get("total_loss", 0)
        result["val/ntp_loss"] = accum.get("total_loss", 0)
        result["val/kl_loss"] = accum.get("kl", 0)

        self._log(
            f"  VAL: top1={accum.get('top1', 0):.3f} "
            f"pairwise={accum.get('pairwise', 0):.3f} "
            f"spearman={accum.get('spearman', 0):.3f} | "
            f"rw_score={accum.get('rw_score', 0):.4f} "
            f"resp_logp={accum.get('response_logp', 0):.2f} | "
            f"kl={accum.get('kl', 0):.4f} "
            f"ratio(mean/max)={accum.get('ratio_mean', 0):.3f}/{accum.get('ratio_max', 0):.2f} "
            f"clip={accum.get('clip_fraction', 0):.3f}"
        )
        return result

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def get_lr(self, step: int) -> float:
        return cosine_annealing_lr_schedule(
            step, self.lr, self.lr_min, self.warmup_steps, self.total_steps
        )

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> float:
        gn = grad_clip(self.model.parameters(), self.max_grad_norm)
        for opt in self.optimizers:
            opt.step()
        self.model.zero_grad()
        return gn


class DPOTrainer(Trainer):
    def __init__(self, args: DPOTrainerArgs):
        super().__init__(args)
        self._model_args: Optional[DSArgs] = None
        self.base_model_ckpt_path = args.base_model_path
        self.ref_model_ckpt_path = args.ref_model_ckpt_path
        self.group_size = args.group_size

    def build_model_and_optimizers(self):
        cfg = MODEL_CONFIGS[self.config_name].copy()
        cfg["max_seq_len"] = self.max_seq_len
        self.max_seq_len = cfg["max_seq_len"]

        base_fields = DSArgs.__dataclass_fields__
        args = DSArgs(**{k: v for k, v in cfg.items() if k in base_fields})

        needed = args.n_layer + args.n_mtp_layer
        if len(args.compress_ratios) < needed:
            args.compress_ratios = args.compress_ratios + tuple(
                [0] * (needed - len(args.compress_ratios))
            )  # MTP layer当时没有写compress_ratios，当时只写了前面block的，打个补丁

        self._model_args = args

        model = DeepSeekV4(args)
        model.train()
        model = model.to(self.local_rank)

        ref_model = DeepSeekV4(args)
        ref_model = ref_model.to(self.local_rank)

        # 和pretrain相比只多出来加载权重
        ckpt = torch.load(
            self.base_model_ckpt_path,
            map_location=f"cuda:{self.local_rank}",
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        ref_model.load_state_dict(ckpt["model_state_dict"])
        ref_model.eval()
        ref_model.requires_grad_(False)

        muon_p, adamw_p = group_params(model)
        idx_p = get_indexer_params(model)

        muon_opt = Muon(muon_p, lr=self.lr, momentum=0.95, weight_decay=0.1)
        adamw_opt = AdamW(adamw_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)
        idx_opt = AdamW(idx_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)

        self.optimizers = [muon_opt, adamw_opt, idx_opt]

        if self.world_size > 1:
            self.model = DDP(
                model, device_ids=[self.local_rank], find_unused_parameters=True
            )
        else:
            self.model = model
        self.ref_model = ref_model

        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self._log(
                f"Model: {self.config_name} | "
                f"Params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable | "
                f"d_model={args.d_model}, n_layer={args.n_layer}, "
                f"n_experts={args.n_experts}, seq_len={args.max_seq_len}"
            )

    def build_dataloaders(self):
        if not self.data_train:
            raise ValueError("--data-train is required")

        train_dataset = DPODataset(self.data_train, group_size=self.group_size)
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self.data_val:
            val_dataset = DPODataset(self.data_val, group_size=self.group_size)
            val_sampler = None
            if self.world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_loader = None

        if self.is_main:
            self._log(
                f"Train data: {self.data_train} | "
                f"{len(train_dataset):,} samples | "
                f"batch={self.batch_size}, group_size={self.group_size}, "
                f"seq_len={self.max_seq_len}"
            )
            if self.val_loader:
                self._log(
                    f"Val data: {self.data_val} | " f"{len(val_dataset):,} samples"
                )

    def train_step(self, batch, is_last_micro):
        # input_ids: (batch_size, group_size, seq_len)
        # 和SFT & Pretrain的数据格式不同，前面两个直接返回了input_ids 和 target ids
        # 其长度都是1023
        # 这里input_ids就是1024，所以要按需取用
        input_ids, masks, scores = batch
        bs, gs, _ = input_ids.shape
        input_ids = input_ids.reshape(bs * gs, -1)

        input_ids = input_ids.to(self.local_rank, non_blocking=True)

        ntp_working, _, _ = self.model(input_ids[..., :-1].contiguous())
        with torch.no_grad():
            ntp_ref, _, _ = self.ref_model(input_ids[..., :-1].contiguous())

        ntp_working = ntp_working.reshape(bs, gs, -1, ntp_working.shape[-1])
        ntp_ref = ntp_ref.reshape(bs, gs, -1, ntp_ref.shape[-1])
        input_ids = input_ids.reshape(bs, gs, -1)  # (bs, gs, 1024)
        masks = masks.to(self.local_rank, non_blocking=True)
        scores = scores.to(self.local_rank, non_blocking=True).reshape(bs, gs)
        metrics = dpo_loss(
            logits_working=ntp_working,
            logits_ref=ntp_ref,
            token_ids=input_ids[..., 1:],
            scores=scores,
            masks=masks[..., :-1],
            beta=self.beta,
        )

        # ---- additional monitoring ----
        reward_margin = (scores.max(dim=-1).values - scores.min(dim=-1).values).mean()

        total_loss = metrics["total_loss"] / self.grad_accum

        if self.world_size > 1 and not is_last_micro:
            with self.model.no_sync():
                total_loss.backward()
        else:
            total_loss.backward()

        return {
            "loss": metrics["total_loss"].item(),
            "acc": metrics["accuracy"].item(),
            "margin_raw": metrics["raw_margin_mean"].item(),
            "margin_beta": metrics["scaled_margin_mean"].item(),
            "conf": metrics["pair_confidence_mean"].item(),
            "chosen_lr": metrics["chosen_log_ratio_mean"].item(),
            "rejected_lr": metrics["rejected_log_ratio_mean"].item(),
            "lr_mean": metrics["log_ratio_mean"].item(),
            "lr_abs": metrics["log_ratio_abs_mean"].item(),
            "reward_margin": reward_margin.item(),
        }

    @torch.no_grad()
    def validate(self, max_val_batches: int = 100) -> Dict[str, float]:
        if not self.val_loader:
            return {}

        self.model.eval()

        accum: Dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            if num_batches >= max_val_batches:
                break

            input_ids, masks, scores = batch
            bs, gs, _ = input_ids.shape
            input_ids = input_ids.reshape(bs * gs, -1)

            input_ids = input_ids.to(self.local_rank, non_blocking=True)

            ntp_working, _, _ = self.model(input_ids[..., :-1].contiguous())
            ntp_ref, _, _ = self.ref_model(input_ids[..., :-1].contiguous())

            ntp_working = ntp_working.reshape(bs, gs, -1, ntp_working.shape[-1])
            ntp_ref = ntp_ref.reshape(bs, gs, -1, ntp_ref.shape[-1])
            input_ids = input_ids.reshape(bs, gs, -1)
            masks = masks.to(self.local_rank, non_blocking=True)
            scores = scores.to(self.local_rank, non_blocking=True).reshape(bs, gs)
            metrics = dpo_loss(
                logits_working=ntp_working,
                logits_ref=ntp_ref,
                token_ids=input_ids[..., 1:],
                scores=scores,
                masks=masks[..., :-1],
                beta=self.beta,
            )

            mask_f = masks[..., :-1].float()
            response_logp = (metrics["logp_w"] * mask_f).sum(dim=-1)  # (bs, gs)
            response_len = mask_f.sum(dim=-1).clamp(min=1)
            response_score = response_logp / response_len  # length-normalized

            # ranking metrics
            pred = response_score.argmax(dim=-1)  # (bs,)
            gt = scores.argmax(dim=-1)  # (bs,)
            top1 = (pred == gt).float().mean()

            score_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)  # (bs, gs, gs)
            model_diff = response_score.unsqueeze(-1) - response_score.unsqueeze(-2)
            pair_mask = score_diff > 0
            pair_correct = ((model_diff > 0) & pair_mask).float().sum(dim=(-1, -2))
            pair_total = pair_mask.float().sum(dim=(-1, -2))
            pairwise_acc = (pair_correct / pair_total.clamp(min=1)).mean()

            def _spearman(x, y):
                rk = lambda t: t.argsort(dim=-1).argsort(dim=-1).float()
                xr, yr = rk(x), rk(y)
                mx, my = xr.mean(dim=-1, keepdim=True), yr.mean(dim=-1, keepdim=True)
                num = ((xr - mx) * (yr - my)).sum(dim=-1)
                den = ((xr - mx) ** 2).sum(dim=-1).sqrt() * ((yr - my) ** 2).sum(
                    dim=-1
                ).sqrt() + 1e-8
                return (num / den).mean()

            spearman = _spearman(response_score, scores)

            for k in [
                "total_loss",
                "accuracy",
                "raw_margin_mean",
                "scaled_margin_mean",
                "pair_confidence_mean",
                "chosen_log_ratio_mean",
                "rejected_log_ratio_mean",
                "log_ratio_mean",
                "log_ratio_std",
                "log_ratio_abs_mean",
            ]:
                accum[k] = accum.get(k, 0.0) + metrics[k].item()
            accum["response_logp"] = (
                accum.get("response_logp", 0.0) + response_score.mean().item()
            )
            accum["top1"] = accum.get("top1", 0.0) + top1.item()
            accum["pairwise"] = accum.get("pairwise", 0.0) + pairwise_acc.item()
            accum["spearman"] = accum.get("spearman", 0.0) + spearman.item()
            num_batches += 1

        self.model.train()

        for k in accum:
            accum[k] /= num_batches

        if self.world_size > 1:
            keys = sorted(accum.keys())
            vals = [accum[k] for k in keys]
            t = torch.tensor(vals, device=self.local_rank)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= self.world_size
            for k, v in zip(keys, t.tolist()):
                accum[k] = v

        result = {f"val/{k}": v for k, v in accum.items()}
        result["val/lm_loss"] = accum.get("total_loss", 0)
        result["val/ntp_loss"] = accum.get("total_loss", 0)
        result["val/kl_loss"] = 0.0

        self._log(
            f"  VAL: "
            f"top1: {accum.get('top1', 0):.3f} | "
            f"pairwise: {accum.get('pairwise', 0):.3f} | "
            f"spearman: {accum.get('spearman', 0):.3f} | "
            f"loss: {accum.get('total_loss', 0):.4f} | "
            f"acc: {accum.get('accuracy', 0):.3f} | "
            f"resp: {accum.get('response_logp', 0):.2f} | "
            f"margin(raw): {accum.get('raw_margin_mean', 0):.4f} | "
            f"margin(beta): {accum.get('scaled_margin_mean', 0):.4f} | "
            f"chosen_lr: {accum.get('chosen_log_ratio_mean', 0):.4f} | "
            f"rejected_lr: {accum.get('rejected_log_ratio_mean', 0):.4f} | "
            f"conf: {accum.get('pair_confidence_mean', 0):.3f} | "
            f"lr_abs: {accum.get('log_ratio_abs_mean', 0):.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def get_lr(self, step: int) -> float:
        return cosine_annealing_lr_schedule(
            step, self.lr, self.lr_min, self.warmup_steps, self.total_steps
        )

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> float:
        gn = grad_clip(self.model.parameters(), self.max_grad_norm)
        for opt in self.optimizers:
            opt.step()
        self.model.zero_grad()
        return gn


class SimPOTrainer(Trainer):

    def __init__(self, args: SimPOTrainerArgs):
        super().__init__(args)
        self._model_args: Optional[DSArgs] = None
        self.base_model_ckpt_path = args.base_model_path

    def build_model_and_optimizers(self):
        cfg = MODEL_CONFIGS[self.config_name].copy()
        cfg["max_seq_len"] = self.max_seq_len
        self.max_seq_len = cfg["max_seq_len"]

        base_fields = DSArgs.__dataclass_fields__
        args = DSArgs(**{k: v for k, v in cfg.items() if k in base_fields})

        needed = args.n_layer + args.n_mtp_layer
        if len(args.compress_ratios) < needed:
            args.compress_ratios = args.compress_ratios + tuple(
                [0] * (needed - len(args.compress_ratios))
            )

        self._model_args = args

        model = DeepSeekV4(args)
        model.train()
        model = model.to(self.local_rank)

        ckpt = torch.load(
            self.base_model_ckpt_path,
            map_location=f"cuda:{self.local_rank}",
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        muon_p, adamw_p = group_params(model)
        idx_p = get_indexer_params(model)

        muon_opt = Muon(muon_p, lr=self.lr, momentum=0.95, weight_decay=0.1)
        adamw_opt = AdamW(adamw_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)
        idx_opt = AdamW(idx_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)

        self.optimizers = [muon_opt, adamw_opt, idx_opt]

        if self.world_size > 1:
            self.model = DDP(
                model, device_ids=[self.local_rank], find_unused_parameters=True
            )
        else:
            self.model = model

        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self._log(
                f"Model: {self.config_name} | "
                f"Params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable | "
                f"d_model={args.d_model}, n_layer={args.n_layer}, "
                f"n_experts={args.n_experts}, seq_len={args.max_seq_len}"
            )

    def build_dataloaders(self):
        if not self.data_train:
            raise ValueError("--data-train is required")

        train_dataset = SimPODataset(self.data_train)
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self.data_val:
            val_dataset = SimPODataset(self.data_val)
            val_sampler = None
            if self.world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_loader = None

        if self.is_main:
            self._log(
                f"Train data: {self.data_train} | "
                f"{len(train_dataset):,} samples | "
                f"batch={self.batch_size}, seq_len={self.max_seq_len}"
            )
            if self.val_loader:
                self._log(f"Val data: {self.data_val} | {len(val_dataset):,} samples")

    def train_step(self, batch, is_last_micro):
        ids, mask = batch  # each (bs, 4, seq_len)
        bs = ids.shape[0]

        # Forward: all 4 candidates together
        all_ids = ids.reshape(bs * 4, -1)
        all_ids = all_ids.to(self.local_rank, non_blocking=True)

        ntp, mtp_list, idx_data = self.model(all_ids[..., :-1].contiguous())
        ntp = ntp.reshape(bs, 4, -1, ntp.shape[-1])  # (bs, 4, seq-1, vocab)

        ids = ids.to(self.local_rank, non_blocking=True)
        mask = mask.to(self.local_rank, non_blocking=True)

        # ---- SFT losses on winner (index 0) ----
        ntp_loss = cross_entropy_masked(ids[:, 0, 1:], ntp[:, 0], mask[:, 0, :-1])
        # MTP on winner: each head produces (bs*4, seq-1, vocab), take winner slice
        mtp_loss = sum(
            cross_entropy_masked(
                ids[:, 0, i + 2 :],
                m.reshape(bs, 4, -1, m.shape[-1])[:, 0, : -(1 + i)],
                mask[:, 0, : -(i + 2)],
            )
            for i, m in enumerate(mtp_list)
        )
        kl_loss = sum(
            indexer_kl_loss(iscore, idx, wc.detach() if wc is not None else None)
            for (iscore, wc, idx) in idx_data
        )

        # ---- SimPO loss: winner vs all 3 losers ----
        spo = simpo_loss(
            logits=ntp,
            ids=ids[..., 1:],
            mask=mask[..., :-1],
            beta=self.beta,
            gamma=self.gamma,
        )

        lm_loss = ntp_loss + 0.3 * mtp_loss
        total_loss = (
            lm_loss + 0.5 * kl_loss + self.lambda_simpo * spo["simpo_loss"]
        ) / self.grad_accum

        if self.world_size > 1 and not is_last_micro:
            with self.model.no_sync():
                total_loss.backward()
        else:
            total_loss.backward()

        return {
            "ntp": ntp_loss.item(),
            "lm": lm_loss.item(),
            "mtp": mtp_loss.item(),
            "kl": kl_loss.item(),
            "simpo": spo["simpo_loss"].item(),
            "pair_acc": spo["pair_acc"].item(),
            "m1": spo["margins"][0].item(),
            "m2": spo["margins"][1].item(),
            "m3": spo["margins"][2].item(),
            "a1": spo["accs"][0].item(),
            "a2": spo["accs"][1].item(),
            "a3": spo["accs"][2].item(),
        }

    @torch.no_grad()
    def validate(self, max_val_batches: int = 100) -> Dict[str, float]:
        if not self.val_loader:
            return {}

        self.model.eval()

        accum: Dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            if num_batches >= max_val_batches:
                break

            ids, mask = batch  # (bs, 4, seq_len)
            bs = ids.shape[0]

            all_ids = ids.reshape(bs * 4, -1)
            all_ids = all_ids.to(self.local_rank, non_blocking=True)

            ntp, mtp_list, idx_data = self.model(all_ids[..., :-1].contiguous())
            ntp = ntp.reshape(bs, 4, -1, ntp.shape[-1])

            ids = ids.to(self.local_rank, non_blocking=True)
            mask = mask.to(self.local_rank, non_blocking=True)

            ntp_loss = cross_entropy_masked(ids[:, 0, 1:], ntp[:, 0], mask[:, 0, :-1])
            mtp_loss = sum(
                cross_entropy_masked(
                    ids[:, 0, i + 2 :],
                    m.reshape(bs, 4, -1, m.shape[-1])[:, 0, : -(1 + i)],
                    mask[:, 0, : -(i + 2)],
                )
                for i, m in enumerate(mtp_list)
            )
            kl_loss = sum(
                indexer_kl_loss(iscore, idx, wc) for (iscore, wc, idx) in idx_data
            )

            spo = simpo_loss(
                logits=ntp,
                ids=ids[..., 1:],
                mask=mask[..., :-1],
                beta=self.beta,
                gamma=self.gamma,
            )

            for k, v in [
                ("ntp", ntp_loss),
                ("mtp", mtp_loss),
                ("kl", kl_loss),
                ("simpo", spo["simpo_loss"]),
                ("pair_acc", spo["pair_acc"]),
            ]:
                accum[k] = accum.get(k, 0.0) + v.item()
            for j in range(3):
                accum[f"m{j+1}"] = accum.get(f"m{j+1}", 0.0) + spo["margins"][j].item()
                accum[f"a{j+1}"] = accum.get(f"a{j+1}", 0.0) + spo["accs"][j].item()
            for j in range(4):
                accum[f"logp{j}"] = accum.get(f"logp{j}", 0.0) + spo["logp"][j].item()
            num_batches += 1

        self.model.train()

        for k in accum:
            accum[k] /= num_batches

        if self.world_size > 1:
            keys = sorted(accum.keys())
            vals = [accum[k] for k in keys]
            t = torch.tensor(vals, device=self.local_rank)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= self.world_size
            for k, v in zip(keys, t.tolist()):
                accum[k] = v

        lm = accum.get("ntp", 0) + 0.3 * accum.get("mtp", 0)
        result = {f"val/{k}": v for k, v in accum.items()}
        result["val/lm_loss"] = lm
        result["val/ntp_loss"] = accum.get("ntp", 0)

        self._log(
            f"  VAL: ntp={accum.get('ntp', 0):.4f} "
            f"simpo={accum.get('simpo', 0):.4f} "
            f"pair_acc={accum.get('pair_acc', 0):.3f} | "
            f"acc={accum.get('a1', 0):.3f}/{accum.get('a2', 0):.3f}/{accum.get('a3', 0):.3f} | "
            f"m={accum.get('m1', 0):.3f}/{accum.get('m2', 0):.3f}/{accum.get('m3', 0):.3f} | "
            f"logp={accum.get('logp0', 0):.2f}/{accum.get('logp1', 0):.2f}/{accum.get('logp2', 0):.2f}/{accum.get('logp3', 0):.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def get_lr(self, step: int) -> float:
        return cosine_annealing_lr_schedule(
            step, self.lr, self.lr_min, self.warmup_steps, self.total_steps
        )

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> float:
        gn = grad_clip(self.model.parameters(), self.max_grad_norm)
        for opt in self.optimizers:
            opt.step()
        self.model.zero_grad()
        return gn


class WeightedSFTTrainer(Trainer):

    def __init__(self, args: WeightedSFTTrainerArgs):
        super().__init__(args)
        self._model_args: Optional[DSArgs] = None
        self.base_model_ckpt_path = args.base_model_path
        self.weight_min = args.weight_min_start  # init before first train_step

    def build_model_and_optimizers(self):
        cfg = MODEL_CONFIGS[self.config_name].copy()
        cfg["max_seq_len"] = self.max_seq_len
        self.max_seq_len = cfg["max_seq_len"]

        base_fields = DSArgs.__dataclass_fields__
        args = DSArgs(**{k: v for k, v in cfg.items() if k in base_fields})

        needed = args.n_layer + args.n_mtp_layer
        if len(args.compress_ratios) < needed:
            args.compress_ratios = args.compress_ratios + tuple(
                [0] * (needed - len(args.compress_ratios))
            )

        self._model_args = args

        model = DeepSeekV4(args)
        model.train()
        model = model.to(self.local_rank)

        ckpt = torch.load(
            self.base_model_ckpt_path,
            map_location=f"cuda:{self.local_rank}",
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        muon_p, adamw_p = group_params(model)
        idx_p = get_indexer_params(model)

        muon_opt = Muon(muon_p, lr=self.lr, momentum=0.95, weight_decay=0.1)
        adamw_opt = AdamW(adamw_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)
        idx_opt = AdamW(idx_p, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)

        self.optimizers = [muon_opt, adamw_opt, idx_opt]

        if self.world_size > 1:
            self.model = DDP(
                model, device_ids=[self.local_rank], find_unused_parameters=True
            )
        else:
            self.model = model

        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self._log(
                f"Model: {self.config_name} | "
                f"Params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable | "
                f"d_model={args.d_model}, n_layer={args.n_layer}, "
                f"n_experts={args.n_experts}, seq_len={args.max_seq_len}"
            )

    def build_dataloaders(self):
        if not self.data_train:
            raise ValueError("--data-train is required")

        train_dataset = WeightedSFTDataset(self.data_train)
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self.data_val:
            val_dataset = WeightedSFTDataset(self.data_val)
            val_sampler = None
            if self.world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_loader = None

        if self.is_main:
            self._log(
                f"Train data: {self.data_train} | "
                f"{len(train_dataset):,} samples | "
                f"batch={self.batch_size}, seq_len={self.max_seq_len}"
            )
            if self.val_loader:
                self._log(f"Val data: {self.data_val} | {len(val_dataset):,} samples")

    def train_step(self, batch, is_last_micro):
        ids, mask, scores = batch  # each (bs, 4, seq_len) / (bs, 4, seq_len) / (bs, 4)
        bs = ids.shape[0]

        # Forward: all 4 candidates together
        all_ids = ids.reshape(bs * 4, -1)
        all_ids = all_ids.to(self.local_rank, non_blocking=True)

        ntp, mtp_list, idx_data = self.model(all_ids[..., :-1].contiguous())
        ntp = ntp.reshape(bs, 4, -1, ntp.shape[-1])  # (bs, 4, seq-1, vocab)

        ids = ids.to(self.local_rank, non_blocking=True)
        mask = mask.to(self.local_rank, non_blocking=True)
        scores = scores.to(self.local_rank, non_blocking=True)

        # ---- Weighted SFT on all 4 responses ----
        self._set_weight_min(self._current_step)
        wsft = weighted_sft_loss(
            logits_ntp=ntp,
            logits_mtp=mtp_list,
            ids=ids,
            mask=mask,
            scores=scores,
            weight_min=self.weight_min,
            weight_max=1.0,
        )
        kl_loss = sum(
            indexer_kl_loss(iscore, idx, wc.detach() if wc is not None else None)
            for (iscore, wc, idx) in idx_data
        )

        total_loss = (wsft["total_loss"] + 0.5 * kl_loss) / self.grad_accum

        if self.world_size > 1 and not is_last_micro:
            with self.model.no_sync():
                total_loss.backward()
        else:
            total_loss.backward()

        return {
            "ntp": wsft["ntp_loss"].item(),
            "mtp": wsft["mtp_loss"].item(),
            "kl": kl_loss.item(),
            "lm": wsft["total_loss"].item(),
            "pair_acc": wsft["pair_acc"].item(),
            "margin_raw": wsft["raw_margin_mean"].item(),
            "chosen_lp": wsft["chosen_lr_mean"].item(),
            "rejected_lp": wsft["rejected_lr_mean"].item(),
            "w_min": self.weight_min,
        }

    @torch.no_grad()
    def validate(self, max_val_batches: int = 100) -> Dict[str, float]:
        if not self.val_loader:
            return {}

        self.model.eval()

        accum: Dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            if num_batches >= max_val_batches:
                break

            ids, mask, scores = batch  # (bs, 4, seq_len) / (bs, 4, seq_len) / (bs, 4)
            bs = ids.shape[0]

            all_ids = ids.reshape(bs * 4, -1)
            all_ids = all_ids.to(self.local_rank, non_blocking=True)

            ntp, mtp_list, idx_data = self.model(all_ids[..., :-1].contiguous())
            ntp = ntp.reshape(bs, 4, -1, ntp.shape[-1])

            ids = ids.to(self.local_rank, non_blocking=True)
            mask = mask.to(self.local_rank, non_blocking=True)
            scores = scores.to(self.local_rank, non_blocking=True)

            wsft = weighted_sft_loss(
                logits_ntp=ntp,
                logits_mtp=mtp_list,
                ids=ids,
                mask=mask,
                scores=scores,
                weight_min=self.weight_min,
                weight_max=1.0,
            )
            kl_loss = sum(
                indexer_kl_loss(iscore, idx, wc) for (iscore, wc, idx) in idx_data
            )

            for k in [
                "ntp_loss",
                "mtp_loss",
                "total_loss",
                "pair_acc",
                "raw_margin_mean",
                "chosen_lr_mean",
                "rejected_lr_mean",
            ]:
                accum[k] = accum.get(k, 0.0) + wsft[k].item()
            accum["kl"] = accum.get("kl", 0.0) + kl_loss.item()

            # per-response log probs for ranking
            logp = wsft["logp"]  # (bs, 4)
            pred = logp.argmax(dim=-1)
            gt = scores.argmax(dim=-1)
            top1 = (pred == gt).float().mean()
            accum["top1"] = accum.get("top1", 0.0) + top1.item()
            for j in range(4):
                accum[f"logp{j}"] = (
                    accum.get(f"logp{j}", 0.0) + logp[:, j].mean().item()
                )
            num_batches += 1

        self.model.train()

        for k in accum:
            accum[k] /= num_batches

        if self.world_size > 1:
            keys = sorted(accum.keys())
            vals = [accum[k] for k in keys]
            t = torch.tensor(vals, device=self.local_rank)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= self.world_size
            for k, v in zip(keys, t.tolist()):
                accum[k] = v

        lm = accum.get("ntp_loss", 0) + 0.3 * accum.get("mtp_loss", 0)
        result = {f"val/{k}": v for k, v in accum.items()}
        result["val/lm_loss"] = lm
        result["val/ntp_loss"] = accum.get("ntp_loss", 0)

        self._log(
            f"  VAL: ntp={accum.get('ntp_loss', 0):.4f} "
            f"mtp={accum.get('mtp_loss', 0):.4f} "
            f"kl={accum.get('kl', 0):.4f} | "
            f"top1={accum.get('top1', 0):.3f} "
            f"pair_acc={accum.get('pair_acc', 0):.3f} | "
            f"margin={accum.get('raw_margin_mean', 0):.4f} | "
            f"chosen_lp={accum.get('chosen_lr_mean', 0):.3f} "
            f"rejected_lp={accum.get('rejected_lr_mean', 0):.3f} | "
            f"logp={accum.get('logp0', 0):.2f}/{accum.get('logp1', 0):.2f}/{accum.get('logp2', 0):.2f}/{accum.get('logp3', 0):.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def get_lr(self, step: int) -> float:
        return cosine_annealing_lr_schedule(
            step, self.lr, self.lr_min, self.warmup_steps, self.total_steps
        )

    def get_weight_min(self, step: int) -> float:
        progress = min(step / self.total_steps, 1.0)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 → 0.0
        return (
            self.weight_min_end
            + (self.weight_min_start - self.weight_min_end) * cos_val
        )

    def _set_weight_min(self, step: int):
        self.weight_min = self.get_weight_min(step)

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> float:
        gn = grad_clip(self.model.parameters(), self.max_grad_norm)
        for opt in self.optimizers:
            opt.step()
        self.model.zero_grad()
        return gn
