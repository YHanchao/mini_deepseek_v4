import torch
from torch import optim
import math

from collections.abc import Iterable


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid negative learning rate: {lr}")
        if len(betas) != 2:
            raise ValueError(f"Invalid moment hyperparameters: {betas}")

        defaults = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "eps": eps,
            "lambda": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
            wd = group["lambda"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                if "t" not in state:
                    state["t"] = 0

                state["t"] += 1
                t = state["t"]

                p.data *= 1 - lr * wd

                state["m"].data = beta_1 * state["m"].data + (1 - beta_1) * p.grad.data
                state["v"].data = (
                    beta_2 * state["v"].data + (1 - beta_2) * p.grad.data**2
                )

                alpha_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)

                p.data -= (
                    alpha_t * state["m"].data / (torch.sqrt(state["v"].data) + eps)
                )

        return loss


class Muon(optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, momentum=0.95, weight_decay=0.1, update_rescale=0.18
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "update_rescale": update_rescale,
        }
        self.abc_list = [(3.4445, -4.7750, 2.0315)] * 8 + [(2, -1.5, 0.5)] * 2
        super().__init__(params, defaults)

    def hybrid_newton_schulz(
        self, X: torch.Tensor, abc_list: list[tuple[float]], eps=1e-20
    ):
        # Step 1: normalize matrix M as M_0 = M / \Vert M\Vert_F
        assert X.ndim >= 2
        y, dtype, need_transpose = X.bfloat16(), X.dtype, X.shape[-2] > X.shape[-1]
        # Recall: 矩阵计算的FLOPs (m, n), (n, m) -> m^2n
        # 所以最好n大m小
        y = y.mT if need_transpose else y
        y /= ((y**2).sum(axis=(-2, -1), keepdims=True) + eps) ** 0.5

        for _, abc in enumerate(abc_list):
            a, b, c = abc
            # Follow the implementation of KellerJordan
            A = y @ y.mT
            B = b * A + c * A @ A
            y = a * y + B @ y

        y = y.mT if need_transpose else y
        return y.to(dtype)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            update_rescale = group["update_rescale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                _n, _m = p.shape[-2], p.shape[-1]

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)

                state["m"].data = state["m"].data * momentum + p.grad.data
                O = self.hybrid_newton_schulz(
                    state["m"].data * momentum + p.grad.data, self.abc_list
                )
                O *= max(_n, _m) * update_rescale
                p.data *= 1 - lr * weight_decay
                p.data -= lr * O

        return loss


def cosine_annealing_lr_schedule(t, lr_max, lr_min, warm_iter, anneal_iter):
    if t < warm_iter:
        # Warm-up stage
        return t * lr_max / warm_iter
    elif t <= anneal_iter:
        return lr_min + 0.5 * (
            1 + math.cos((t - warm_iter) / (anneal_iter - warm_iter) * math.pi)
        ) * (lr_max - lr_min)
    else:
        return lr_min


def grad_clip(params: Iterable[torch.nn.Parameter], max_norm, eps=1e-6):
    total_norm = torch.sqrt(
        sum(torch.norm(p.grad.data) ** 2 for p in params if p.grad is not None)
    )
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data *= scale


def group_params(model):
    """Split model parameters into Muon (matrix) and AdamW (vector/special) groups.

    Indexer params are excluded — they are trained separately via KL loss.

    Layer 1 (hard constraint): params with ndim < 2 → AdamW
    Layer 2 (paper exceptions): ≥2D but specified as AdamW by name keywords
    Layer 3 (remainder): all other ≥2D params → Muon
    """
    muon_p, adamw_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "indexer" in name:
            continue  # trained separately via KL loss
        if p.ndim < 2:
            adamw_p.append(p)
        elif any(
            kw in name
            for kw in ["embedding", "prediction", "phi_pre", "phi_post", "phi_res"]
        ):
            adamw_p.append(p)
        else:
            muon_p.append(p)
    return muon_p, adamw_p


def get_indexer_params(model):
    """Return list of all Indexer parameters (trained via KL loss)."""
    return [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and "indexer" in name
    ]
