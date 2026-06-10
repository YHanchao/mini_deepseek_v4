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
