"""Generate scaling law plot from training log."""

import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Parse val loss from train.log ──────────────────────────────────
steps, losses, tokens_seen = [], [], []
with open("checkpoints/pretrain/train.log") as f:
    for line in f:
        if "Validation step" not in line:
            continue
        parts = line.strip().split()
        step_str = parts[4].rstrip(":")
        lm_str = parts[5]
        step = int(step_str)
        lm = float(lm_str.split("=")[1])
        steps.append(step)
        losses.append(lm)
        tokens_seen.append(step * 16384)  # effective tokens per step

n = len(steps)
tokens_seen_M = np.array(tokens_seen) / 1e6
losses = np.array(losses)
step_arr = np.array(steps)

# only use pre-overfitting data (step < 213500)
data_mask = step_arr < 213500
tokens_M = tokens_seen_M[data_mask]
losses_val = losses[data_mask]

# ── Power-law fit on (5000, 200000] ────────────────────────────────
fit_mask = (step_arr >= 5000) & (step_arr <= 200000)
log_t = np.log(tokens_seen_M[fit_mask])
log_l = np.log(losses[fit_mask])
coeff = np.polyfit(log_t, log_l, 1)
alpha = coeff[0]
c = coeff[1]
r2 = np.corrcoef(log_t, log_l)[0, 1] ** 2

# smooth fit line
t_smooth = np.logspace(math.log10(7), math.log10(11000), 200)
l_smooth = np.exp(alpha * np.log(t_smooth) + c)

# extrapolation
params_M = 305
chinchilla_opt = params_M * 20
extra_tok = [5000, 7000, 10000]
extra_pred = [np.exp(alpha * np.log(t) + c) for t in extra_tok]

# ── Plot ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: log-log scaling plot
ax1.loglog(
    tokens_M,
    losses_val,
    "o",
    markersize=1.8,
    color="#2171b5",
    alpha=0.35,
    rasterized=True,
    label=f"val loss ({np.sum(data_mask)} checkpoints)",
)
ax1.loglog(
    t_smooth,
    l_smooth,
    "-",
    color="#d73027",
    linewidth=2,
    label=f"power-law fit  $\\propto T^{{{alpha:.4f}}}$,  $R^2={r2:.4f}$",
)

# Chinchilla reference
ax1.axvline(x=chinchilla_opt, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.annotate(
    f"Chinchilla optimal\n(≈{chinchilla_opt}M tokens)",
    xy=(chinchilla_opt, 4.8),
    fontsize=8,
    ha="center",
    color="gray",
)

# Extrapolation markers
for t, p in zip(extra_tok, extra_pred):
    ax1.plot(t, p, "v", color="#d73027", markersize=5, alpha=0.6)
    ax1.annotate(f" {p:.2f}", (t * 1.08, p), fontsize=7.5, va="center", color="#d73027")

# Best checkpoint
best_step = 191000
best_tok = best_step * 16384 / 1e6
best_loss = losses[steps.index(best_step)] if best_step in steps else 4.75
ax1.plot(best_tok, best_loss, "*", color="#f59e0b", markersize=14, zorder=5)
ax1.annotate(
    f"  ckpt_best\n  step {best_step//1000}K, lm={best_loss:.2f}",
    xy=(best_tok, best_loss),
    fontsize=9,
    va="bottom",
    color="#92400e",
)

ax1.set_xlabel("Training tokens (M)", fontsize=11)
ax1.set_ylabel("Validation LM loss", fontsize=11)
ax1.set_title("Scaling Law: DeepSeekV4-305M", fontsize=13, fontweight="bold")
ax1.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(7, 11000)

# ── Right: linear-scale zoom on best region ────────────────────────
ax2.plot(tokens_M, losses_val, "-o", markersize=1.2, linewidth=1, color="#2171b5")
ax2.plot(
    best_tok,
    best_loss,
    "*",
    color="#f59e0b",
    markersize=16,
    zorder=5,
    label=f"ckpt_best (step {best_step//1000}K)\nval lm={best_loss:.2f}",
)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax2.set_xlabel("Training tokens (M)", fontsize=11)
ax2.set_ylabel("Validation LM loss", fontsize=11)
ax2.set_title("Loss Curve (linear scale)", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.2)

fig.tight_layout(pad=2.5)
fig.savefig("docs/scaling_law.png", dpi=180, bbox_inches="tight")
print(f"Saved docs/scaling_law.png")
print(f"  α = {alpha:.4f},  R² = {r2:.4f}")
print(f"  Chinchilla ratio at best: {best_tok:.0f}M / 305M = {best_tok/305:.1f}")
print(
    f"  Extrapolation: 5B→{extra_pred[0]:.2f}, 7B→{extra_pred[1]:.2f}, 10B→{extra_pred[2]:.2f}"
)
