import torch


def cross_entropy(y_true, y_pred):
    log_probs = torch.log_softmax(y_pred, dim=-1)
    nll = -log_probs.gather(dim=-1, index=y_true.unsqueeze(-1)).squeeze(-1)
    return torch.mean(nll)


def cross_entropy_masked(y_true, y_pred, mask):
    # 防止除零
    if mask.sum() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    log_probs = torch.log_softmax(y_pred, dim=-1)
    nll = -log_probs.gather(dim=-1, index=y_true.unsqueeze(-1)).squeeze(-1)
    masked_nll = nll * mask.float()
    return masked_nll.sum() / mask.sum().float()


def cross_entropy_chunked(y_true, y_pred, chunk_size=8192):
    vocab_size = y_pred.shape[-1]
    y_pred_flat = y_pred.reshape(-1, vocab_size)
    y_true_flat = y_true.reshape(-1)

    max_val = y_pred_flat.max(dim=-1, keepdim=True).values

    log_sum_exp = torch.zeros_like(max_val)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = y_pred_flat[:, start:end]
        log_sum_exp += (chunk - max_val).exp().sum(dim=-1, keepdim=True)
    log_sum_exp = log_sum_exp.log()

    total_loss = 0.0
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        mask = (y_true_flat >= start) & (y_true_flat < end)
        if not mask.any():
            continue
        log_probs_chunk = (
            y_pred_flat[mask, start:end] - max_val[mask] - log_sum_exp[mask]
        )
        local_indices = (y_true_flat[mask] - start).long().unsqueeze(-1)
        nll = -log_probs_chunk.gather(dim=-1, index=local_indices).squeeze(-1)
        total_loss += nll.sum()

    return total_loss / y_true_flat.numel()


def indexer_kl_loss(index_score, compress_topk_idxs, weight_compress):
    """KL divergence loss for training the lightning indexer.

    Eq. (3)-(4): L^I = Σ_t KL(p_{t,S_t} || Softmax(I_{t,S_t}))

    Target distribution p is derived from main attention weights (detached).
    Predicted distribution is softmax of indexer scores gathered at selected blocks.

    Args:
        index_score: (b, s, n_blocks) — indexer scores after causal masking, before topk
        compress_topk_idxs: (b, s, topk) — indices of selected compressed blocks
        weight_compress: (b, s, n_heads, topk) — attention weights for compressed blocks (detached)

    Returns:
        scalar KL loss, or 0.0 if no indexer data is available
    """
    if index_score is None or compress_topk_idxs is None or weight_compress is None:
        return torch.tensor(
            0.0,
            device=(
                index_score.device
                if index_score is not None
                else (
                    weight_compress.device
                    if weight_compress is not None
                    else torch.cuda.current_device()
                )
            ),
        )

    b, s, n_heads, topk = weight_compress.shape

    # Target: sum attention weights across heads, L1 normalize → p_{t,S_t}
    target = weight_compress.sum(dim=2)  # (b, s, topk)
    target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)

    # Predicted: gather index_score at selected indices, softmax
    mask = compress_topk_idxs == -1
    idx = compress_topk_idxs.clamp(0)
    pred = index_score.gather(-1, idx)  # (b, s, topk)
    pred = pred.masked_fill(mask, float("-inf"))

    # Positions where ALL blocks are causally masked produce log_softmax(-inf) = NaN.
    # These positions have target = 0 anyway, so set log_pred to 0 to avoid NaN
    # in both forward and backward (LogSoftmaxBackward chokes on NaN input).
    all_masked = mask.all(dim=-1, keepdim=True)  # (b, s, 1)
    pred_safe = torch.where(all_masked, 0.0, pred)

    log_pred = torch.log_softmax(pred_safe, dim=-1)  # (b, s, topk)
    kl = target * (torch.log(target + 1e-8) - log_pred)  # (b, s, topk)
    kl = kl.masked_fill(mask | all_masked, 0.0)
    kl = kl.sum(dim=-1)  # (b, s)

    # Only count positions that attend to at least one compressed block
    valid = target.sum(dim=-1) > 1e-8  # (b, s)
    return (
        kl[valid].mean()
        if valid.any()
        else torch.tensor(0.0, device=index_score.device)
    )


def grpo_loss(
    logits_working: torch.Tensor,
    logits_ref: torch.Tensor,
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 0.2,
    beta: float = 0.05,
):
    """GRPO loss with token-level importance ratio and monitoring metrics.

    Args:
        logits_working: (bs, gs, seq_len, vocab) — policy model logits
        logits_ref:     (bs, gs, seq_len, vocab) — reference model logits
        token_ids:      (bs, gs, seq_len)        — ground-truth token ids
        scores:         (bs, gs)                 — per-response reward scores
        masks:          (bs, gs, seq_len)        — 1 on completion tokens, 0 elsewhere
        eps:            PPO clip epsilon
        beta:           KL penalty coefficient

    Returns:
        dict with keys: total_loss, policy_loss, kl, ratio_mean, ratio_std,
                        ratio_max, clip_fraction, advantage
    """
    # 1. Per-token log probs: (bs, gs, seq_len)
    logp_w = (
        torch.log_softmax(logits_working, dim=-1)
        .gather(dim=-1, index=token_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    logp_r = (
        torch.log_softmax(logits_ref, dim=-1)
        .gather(dim=-1, index=token_ids.unsqueeze(-1))
        .squeeze(-1)
    )

    # 2. Per-token log-ratio and ratio: (bs, gs, seq_len)
    log_ratio = logp_w - logp_r
    ratio = torch.exp(log_ratio)

    # 3. Group-normalized advantage: (bs, gs, 1)
    adv_2d = (scores - scores.mean(dim=-1, keepdim=True)) / (
        scores.std(dim=-1, keepdim=True) + 1e-8
    )
    adv = adv_2d.unsqueeze(-1)

    # 4. PPO-style clipped objective (per-token)
    ratio_clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    gain = torch.min(ratio * adv, ratio_clipped * adv)

    # 5. Policy loss: average over masked tokens
    mask_f = masks.float()
    valid = mask_f.sum().clamp(min=1)
    policy_loss = -(gain * mask_f).sum() / valid

    # 6. KL penalty (per-token, masked)
    kl_per_tok = ratio - log_ratio - 1
    kl = (kl_per_tok * mask_f).sum() / valid

    # 7. Monitoring: ratio statistics over valid tokens
    ratio_valid = ratio[masks.bool()]
    ratio_mean = ratio_valid.mean()
    ratio_std = ratio_valid.std()
    ratio_max = ratio_valid.max()

    # 8. Monitoring: PPO clip fraction
    clip_mask = (ratio > 1 + eps) | (ratio < 1 - eps)
    clip_fraction = (clip_mask.float() * mask_f).sum() / valid

    return {
        "total_loss": policy_loss + beta * kl,
        "policy_loss": policy_loss,
        "kl": kl,
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "ratio_max": ratio_max,
        "clip_fraction": clip_fraction,
        "advantage": adv_2d,
        "logp_w": logp_w,
        "logp_r": logp_r,
    }
