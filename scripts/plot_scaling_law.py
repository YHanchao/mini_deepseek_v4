"""Generate scaling law plots for pretraining and SFT from training logs."""

import math
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Parse helpers ────────────────────────────────────────────────────

def parse_validation_entries(log_path):
    """Parse (step, lm_loss) from validation lines. Dedup: keep smaller loss."""
    entries = []
    with open(log_path) as f:
        for line in f:
            if "Validation step" not in line:
                continue
            parts = line.strip().split()
            step = int(parts[4].rstrip(":"))
            lm = float(parts[5].split("=")[1])
            entries.append((step, lm))

    best = {}
    for step, lm in entries:
        if step not in best or lm < best[step]:
            best[step] = lm

    steps = sorted(best.keys())
    losses = [best[s] for s in steps]
    return np.array(steps), np.array(losses)


def parse_world_size_changes(log_path):
    """Return list of (start_step, world_size) sorted by start_step."""
    init_entries = []   # (timestamp, world_size)
    train_entries = []  # (timestamp, start_step)

    with open(log_path) as f:
        for line in f:
            if "Trainer init:" in line:
                m = re.search(r"world_size=(\d+)", line)
                ts = line.split("]")[0] + "]"
                if m:
                    init_entries.append((ts, int(m.group(1))))
            elif "Training from step" in line:
                m = re.search(r"Training from step (\d+)", line)
                ts = line.split("]")[0] + "]"
                if m:
                    train_entries.append((ts, int(m.group(1))))

    changes = [(0, init_entries[0][1])]
    ti = 0
    for train_ts, start_step in train_entries:
        while ti + 1 < len(init_entries) and init_entries[ti + 1][0] <= train_ts:
            ti += 1
        ws = init_entries[ti][1]
        if not changes or changes[-1][0] != start_step:
            changes.append((start_step, ws))
        elif changes[-1][1] != ws:
            changes[-1] = (start_step, ws)

    return changes


def compute_cumulative_tokens(steps, world_size_changes, batch_size=4, seq_len=1024):
    """Compute cumulative tokens at each validation step."""
    tokens = np.zeros(len(steps))
    changes = sorted(world_size_changes)

    for i, step in enumerate(steps):
        total = 0
        for j in range(len(changes)):
            seg_start = changes[j][0]
            seg_ws = changes[j][1]
            seg_end = changes[j + 1][0] if j + 1 < len(changes) else step
            seg_end = min(seg_end, step)
            if seg_end > seg_start:
                total += (seg_end - seg_start) * seg_ws * batch_size * seq_len
        tokens[i] = total

    return tokens


# ── Power-law fit ────────────────────────────────────────────────────

def power_law_fit(tokens, losses, step_arr, fit_step_min=5000, fit_step_max=None):
    """Fit power law L ∝ T^α on steps in [fit_step_min, fit_step_max]."""
    if fit_step_max is None:
        fit_mask = step_arr >= fit_step_min
    else:
        fit_mask = (step_arr >= fit_step_min) & (step_arr <= fit_step_max)
    log_t = np.log(tokens[fit_mask])
    log_l = np.log(losses[fit_mask])
    coeff = np.polyfit(log_t, log_l, 1)
    alpha = coeff[0]
    c = coeff[1]
    r2 = np.corrcoef(log_t, log_l)[0, 1] ** 2
    return alpha, c, r2


# ── Pretraining plot ─────────────────────────────────────────────────

def plot_pretrain(steps, tokens, losses, save_path):
    tokens_M = tokens / 1e6
    step_arr = np.array(steps)
    losses = np.array(losses)

    # Fit on power-law region only (before plateau)
    fit_step_min, fit_step_max = 5000, 250000
    alpha, c, r2 = power_law_fit(tokens, losses, step_arr,
                                 fit_step_min=fit_step_min,
                                 fit_step_max=fit_step_max)

    # Smooth fit line: cover full data range
    t_smooth_M = np.logspace(math.log10(tokens_M[0]), math.log10(tokens_M[-1]), 200)
    l_smooth = np.exp(alpha * np.log(t_smooth_M * 1e6) + c)

    # Final checkpoint: step 375000
    final_step = 375000
    final_idx = np.argmin(np.abs(step_arr - final_step))
    final_tok = tokens_M[final_idx]
    final_loss = losses[final_idx]
    final_step_actual = steps[final_idx]

    params_M = 305
    chinchilla_opt = params_M * 20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: log-log scaling plot
    ax1.loglog(tokens_M, losses, "o", markersize=1.5, color="#2171b5", alpha=0.3,
               rasterized=True, label=f"val loss ({len(steps)} checkpoints)")
    ax1.loglog(t_smooth_M, l_smooth, "-", color="#d73027", linewidth=2,
               label=f"power-law fit  $\\propto T^{{{alpha:.4f}}}$,  $R^2={r2:.4f}$")

    ax1.axvline(x=chinchilla_opt, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.annotate(f"Chinchilla optimal\n(≈{chinchilla_opt}M tokens)",
                 xy=(chinchilla_opt, losses[0] * 1.1), fontsize=8, ha="center", color="gray")

    ax1.plot(final_tok, final_loss, "*", color="#f59e0b", markersize=14, zorder=5)
    ax1.annotate(f"  ckpt_final\n  step {final_step_actual//1000}K, lm={final_loss:.2f}",
                 xy=(final_tok, final_loss), fontsize=9, va="bottom", color="#92400e")

    ax1.set_xlabel("Training tokens (M)", fontsize=11)
    ax1.set_ylabel("Validation LM loss", fontsize=11)
    ax1.set_title("Scaling Law: DeepSeekV4-305M (Pretrain)", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # Right: linear-scale
    ax2.plot(tokens_M, losses, "-o", markersize=1.2, linewidth=1, color="#2171b5")
    ax2.plot(final_tok, final_loss, "*", color="#f59e0b", markersize=16, zorder=5,
             label=f"ckpt_final (step {final_step_actual//1000}K)\nval lm={final_loss:.2f}")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.set_xlabel("Training tokens (M)", fontsize=11)
    ax2.set_ylabel("Validation LM loss", fontsize=11)
    ax2.set_title("Loss Curve (linear scale)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout(pad=2.5)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Saved {save_path}")
    print(f"  Pretrain: α={alpha:.4f}, R²={r2:.4f}  (fit on steps [{fit_step_min}, {fit_step_max}])")
    print(f"  Total tokens: {tokens_M[-1]:.0f}M")
    print(f"  Final ckpt: step={final_step_actual}, lm={final_loss:.4f}, tokens={final_tok:.0f}M")
    print(f"  Chinchilla ratio at final: {final_tok:.0f}M / {params_M}M = {final_tok / params_M:.1f}")
    plt.close(fig)


# ── SFT plot ─────────────────────────────────────────────────────────

def plot_sft(steps, tokens, losses, save_path):
    tokens_M = tokens / 1e6
    step_arr = np.array(steps)
    losses = np.array(losses)

    # Final checkpoint (ckpt_final = step 6378)
    final_step = 6378
    final_idx = np.argmin(np.abs(step_arr - final_step))
    final_tok = tokens_M[final_idx]
    final_loss = losses[final_idx]
    final_step_actual = steps[final_idx]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(tokens_M, losses, "-o", markersize=2, linewidth=1.2, color="#2171b5",
            label=f"val lm loss ({len(steps)} checkpoints)")

    ax.plot(final_tok, final_loss, "*", color="#f59e0b", markersize=16, zorder=5)
    ax.annotate(f"  ckpt_final\n  step {final_step_actual}, lm={final_loss:.4f}",
                xy=(final_tok, final_loss), fontsize=9, va="bottom", color="#92400e")

    ax.set_xlabel("Training tokens (M)", fontsize=11)
    ax.set_ylabel("Validation LM loss", fontsize=11)
    ax.set_title("SFT Loss Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(pad=2.5)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Saved {save_path}")
    print(f"  SFT: total tokens={tokens_M[-1]:.1f}M, steps={step_arr[-1]}")
    print(f"  Final ckpt: step={final_step_actual}, lm={final_loss:.4f}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # ── Pretraining ──────────────────────────────────────────────────
    pretrain_log = "checkpoints/pretrain_20260701/train.log"
    steps_pt, losses_pt = parse_validation_entries(pretrain_log)

    ws_changes = parse_world_size_changes(pretrain_log)
    print(f"World size changes (step, ws): {ws_changes}")

    tokens_pt = compute_cumulative_tokens(steps_pt, ws_changes)

    # Truncate at step 375000
    mask = steps_pt <= 375000
    steps_pt = steps_pt[mask]
    losses_pt = losses_pt[mask]
    tokens_pt = tokens_pt[mask]

    plot_pretrain(steps_pt, tokens_pt, losses_pt, "docs/scaling_law_pretrain.png")

    # ── SFT ──────────────────────────────────────────────────────────
    sft_log = "checkpoints/sft/train.log"
    steps_sft, losses_sft = parse_validation_entries(sft_log)

    # SFT: constant world_size=5, batch_size=4, seq_len=1024
    tokens_sft = steps_sft.astype(np.float64) * 5 * 4 * 1024

    plot_sft(steps_sft, tokens_sft, losses_sft, "docs/scaling_law_sft.png")


if __name__ == "__main__":
    main()
