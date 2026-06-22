import os
import time
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import MODEL_CONFIGS
from src.deepseek import DeepSeekV4, DSArgs
from src.dataset import PretrainDataset
from src.loss import cross_entropy, indexer_kl_loss
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

    def load_checkpoint(self, path: str) -> Tuple[int, Dict]:
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

        return state["step"] + 1, extra

    def _cleanup_checkpoints(self):
        if not self.is_main:
            return
        ckpts = sorted(Path(self.output_dir).glob("ckpt_*.pt"))
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

        self._log(
            f"Step {step:>6d}/{self.total_steps} | "
            f"loss: {losses.get('lm', 0):.4f} | "
            f"ntp: {losses.get('ntp', 0):.4f} | "
            f"mtp: {losses.get('mtp', 0):.4f} | "
            f"kl: {losses.get('kl', 0):.4f} | "
            f"lr: {lr:.2e} | "
            f"grad_norm: {grad_norm:.4f} | "
            f"tok/s: {tok_per_sec:,.0f} | "
            f"mem: {mem_mb:.0f}MB | "
            f"ETA: {eta_days:.1f}d"
        )

        if self.wandb_run:
            self.wandb_run.log(
                {
                    "train/lm_loss": losses.get("lm", 0),
                    "train/ntp_loss": losses.get("ntp", 0),
                    "train/mtp_loss": losses.get("mtp", 0),
                    "train/kl_loss": losses.get("kl", 0),
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mem_mb": mem_mb,
                    "train/step": step,
                },
                step=step,
            )

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
            self.start_step, extra = self.load_checkpoint(self.resume)
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

            lm = accum_losses.get("lm", 0.0)
            if not torch.isfinite(torch.tensor(lm)):
                self._log(
                    f"Step {step}: NaN/Inf loss detected! Saving emergency checkpoint."
                )
                self.save_checkpoint(
                    os.path.join(self.output_dir, f"ckpt_nan_{step:07d}.pt"),
                    step,
                    extra={"best_val_loss": self.best_val_loss},
                )
                raise RuntimeError(f"Loss is NaN/Inf at step {step}")

            grad_norm = self._optimizer_step()

            step_time = time.time() - self._step_start_time

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
                if val_lm < self.best_val_loss:
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
        self.idx_p: list = []
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
        idx_p = get_indexer_params(model)  # 偷懒了，理应这里也分Muon & Adam的
        self.idx_p = idx_p

        # 直接 fix 为论文里面的优化器超参
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
        mtp_loss = sum(cross_entropy(target_ids, m) for m in mtp_list)
        kl_loss = sum(
            indexer_kl_loss(iscore, idx, wc) for (iscore, wc, idx) in idx_data
        )

        lm_loss = ntp_loss + 0.3 * mtp_loss
        lm_loss_scaled = lm_loss / self.grad_accum
        kl_loss_scaled = 0.5 * kl_loss / self.grad_accum

        # KL grad: manual, bypasses DDP hooks
        idx_p = self.idx_p
        if idx_p and kl_loss_scaled.requires_grad:
            idx_grads = torch.autograd.grad(
                kl_loss_scaled, idx_p, retain_graph=True, allow_unused=True
            )
            for p, g in zip(idx_p, idx_grads):
                if g is not None:
                    p.grad = g if p.grad is None else p.grad.add_(g)

        # LM backward — no_sync for non-last micro-batches in DDP
        if self.world_size > 1 and not is_last_micro:
            with self.model.no_sync():
                lm_loss_scaled.backward()
        else:
            lm_loss_scaled.backward()

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
            total_mtp += sum(cross_entropy(target_ids, m).item() for m in mtp_list)
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
