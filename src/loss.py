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


def dpo_loss(
    logits_working: torch.Tensor,
    logits_ref: torch.Tensor,
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    masks: torch.Tensor,
    beta: float = 0.1,
):
    """DPO loss with Bradley-Terry model over all pairs within each group.

    For each group of 4 responses, constructs all C(4,2)=6 pairs. Each pair where
    scores differ contributes a binary cross-entropy loss that pushes the model to
    increase log-ratio for the chosen response relative to the rejected one.

    Args:
        logits_working: (bs, gs, seq_len, vocab) — policy model logits
        logits_ref:     (bs, gs, seq_len, vocab) — reference model logits
        token_ids:      (bs, gs, seq_len)        — ground-truth token ids
        scores:         (bs, gs)                 — per-response scalar rewards
        masks:          (bs, gs, seq_len)        — 1 on response tokens, 0 elsewhere
        beta:           temperature for the Bradley-Terry implicit reward

    Returns:
        dict with keys: total_loss, accuracy, log_ratio_mean, log_ratio_std,
                        log_ratio_diff_mean, num_pairs
    """
    # 1. Per-token log probs: (bs, gs, seq_len)
    logp_w = (
        torch.log_softmax(logits_working, dim=-1)
        .gather(dim=-1, index=token_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    with torch.no_grad():
        logp_r = (
            torch.log_softmax(logits_ref, dim=-1)
            .gather(dim=-1, index=token_ids.unsqueeze(-1))
            .squeeze(-1)
        )

    # 2. Sequence-level mean log-ratio: (bs, gs)
    resp_len = masks.sum(dim=-1).clamp(min=1)
    log_ratio = ((logp_w - logp_r) * masks).sum(dim=-1) / resp_len

    # 3. Build all C(gs,2) pairs
    gs = scores.shape[1]
    idx_i, idx_j = torch.triu_indices(gs, gs, offset=1, device=scores.device).unbind()

    score_i = scores[:, idx_i]  # (bs, 6)
    score_j = scores[:, idx_j]  # (bs, 6)

    log_ratio_i = log_ratio[:, idx_i]  # (bs, 6)
    log_ratio_j = log_ratio[:, idx_j]  # (bs, 6)

    # chosen = higher score, rejected = lower score
    pref_mask = score_i > score_j
    tie_mask = score_i == score_j

    chosen_lr = torch.where(pref_mask, log_ratio_i, log_ratio_j)
    rejected_lr = torch.where(pref_mask, log_ratio_j, log_ratio_i)

    # 4. DPO loss: -log(sigmoid(beta * (chosen - rejected)))
    raw_margin = chosen_lr - rejected_lr
    diff = beta * raw_margin
    pair_loss = -torch.nn.functional.logsigmoid(diff)

    valid = ~tie_mask
    valid_count = valid.sum().clamp(min=1)
    total_loss = (pair_loss * valid).sum() / valid_count

    # 5. Accuracy: fraction of valid pairs correctly ordered (raw_margin > 0)
    correct = (raw_margin > 0) & valid
    accuracy = correct.float().sum() / valid_count

    # 6. Monitoring metrics over valid pairs
    def _valid_mean(x):
        return x[valid].mean() if valid.any() else torch.tensor(0.0, device=scores.device)

    return {
        "total_loss": total_loss,
        "accuracy": accuracy,
        # DPO signal
        "raw_margin_mean": _valid_mean(raw_margin),
        "scaled_margin_mean": _valid_mean(diff),
        "pair_confidence_mean": _valid_mean(torch.sigmoid(diff)),
        # chosen/rejected separation
        "chosen_log_ratio_mean": _valid_mean(chosen_lr),
        "rejected_log_ratio_mean": _valid_mean(rejected_lr),
        # policy drift
        "log_ratio_mean": log_ratio.mean(),
        "log_ratio_std": log_ratio.std(),
        "log_ratio_abs_mean": log_ratio.abs().mean(),
        # for downstream ranking metrics
        "logp_w": logp_w,
        "logp_r": logp_r,
    }


def logp_from_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Extract per-token log-probabilities from logits."""
    return (
        torch.log_softmax(logits, dim=-1)
        .gather(dim=-1, index=token_ids.unsqueeze(-1))
        .squeeze(-1)
    )


def simpo_loss(
    logits: torch.Tensor,
    ids: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 0.1,
):
    """SimPO loss: winner vs all 3 losers, vectorized.

    Args:
        logits: (bs, 4, seq_len, vocab)
        ids:    (bs, 4, seq_len)   — already shifted by trainer
        mask:   (bs, 4, seq_len)   — already shifted by trainer
        beta:   SimPO temperature
        gamma:  target margin

    Returns:
        dict with simpo_loss, pair_acc, margins(3,), accs(3,), logp(4,)
    """
    logp = logp_from_logits(logits, ids) * mask.float()  # (bs, 4, seq_len)
    length = mask.float().sum(dim=-1).clamp(min=1)
    r = logp.sum(dim=-1) / length  # (bs, 4)  length-normalized scores

    diff = r[:, 0:1] - r[:, 1:4]  # (bs, 3)  winner minus each loser
    losses = -torch.nn.functional.logsigmoid(beta * diff - gamma)  # (bs, 3)

    return {
        "simpo_loss": losses.mean(),
        "pair_acc": (diff > 0).float().mean(),
        "margins": diff.mean(dim=0),     # (3,)  per-pair average margin
        "accs": (diff > 0).float().mean(dim=0),  # (3,)  per-pair accuracy
        "logp": r.mean(dim=0),           # (4,)  avg logp for each position
    }


def weighted_sft_loss(
    logits_ntp: torch.Tensor,
    logits_mtp: list,
    ids: torch.Tensor,
    mask: torch.Tensor,
    scores: torch.Tensor,
):
    """Weighted SFT: all 4 responses contribute, weighted by score / max_score.

    Args:
        logits_ntp: (bs, 4, seq-1, vocab) — NTP logits
        logits_mtp: list of (bs*4, seq_i, vocab) — MTP logits per head
        ids:        (bs, 4, seq_len)          — full token ids (unshifted)
        mask:       (bs, 4, seq_len)          — 1 on response tokens
        scores:     (bs, 4)                   — sorted [winner, loser1, loser2, loser3]

    Returns:
        dict with total_loss, ntp_loss, mtp_loss, and DPO-style monitoring metrics.
    """
    bs, gs = ids.shape[:2]

    # ---- weights: linear map to [0.1, 1.0] ----
    s_min = scores.min(dim=-1, keepdim=True).values
    s_max = scores.max(dim=-1, keepdim=True).values
    s_range = (s_max - s_min).clamp(min=1e-8)
    w = 0.1 + 0.9 * (scores - s_min) / s_range  # (bs, 4)

    # ---- NTP weighted CE ----
    logp_ntp = torch.log_softmax(logits_ntp, dim=-1)  # (bs, 4, seq-1, vocab)
    nll_ntp = -logp_ntp.gather(dim=-1, index=ids[..., 1:].unsqueeze(-1)).squeeze(-1)  # (bs, 4, seq-1)
    mask_ntp = mask[..., :-1].float()
    w_ntp = w.unsqueeze(-1)  # (bs, 4, 1)
    ntp_loss = (nll_ntp * mask_ntp * w_ntp).sum() / (mask_ntp * w_ntp).sum().clamp(min=1)

    # ---- MTP weighted CE ----
    mtp_loss = torch.tensor(0.0, device=ids.device)
    for i, m in enumerate(logits_mtp):
        m_r = m.reshape(bs, gs, -1, m.shape[-1])  # (bs, 4, seq_i, vocab)
        offset = 1 + i + 1  # shift: 1 (ntp) + i (mtp head has already lost i tokens)
        target = ids[..., offset:]  # (bs, 4, seq_len - offset)
        m_r = m_r[:, :, : target.shape[-1]]  # align lengths
        mask_mtp_i = mask[..., offset:].float()  # (bs, 4, seq_len - offset)

        logp_m = torch.log_softmax(m_r, dim=-1)
        nll_m = -logp_m.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        mtp_loss = mtp_loss + (nll_m * mask_mtp_i * w_ntp).sum() / (mask_mtp_i * w_ntp).sum().clamp(min=1)

    total_loss = ntp_loss + 0.3 * mtp_loss

    # ---- Monitoring (no_grad) ----
    with torch.no_grad():
        # per-response length-normalized log prob
        resp_len = mask_ntp.sum(dim=-1).clamp(min=1)  # (bs, 4)
        logp_resp = (logp_ntp.gather(dim=-1, index=ids[..., 1:].unsqueeze(-1)).squeeze(-1)
                     * mask_ntp).sum(dim=-1) / resp_len  # (bs, 4)

        # 6 pairs + monitoring (same as DPO)
        idx_i, idx_j = torch.triu_indices(gs, gs, offset=1, device=ids.device).unbind()
        lp_i = logp_resp[:, idx_i]
        lp_j = logp_resp[:, idx_j]

        # chosen = higher score
        si = scores[:, idx_i]
        sj = scores[:, idx_j]
        pref_mask = si > sj
        tie_mask = si == sj

        chosen_lp = torch.where(pref_mask, lp_i, lp_j)
        rejected_lp = torch.where(pref_mask, lp_j, lp_i)
        raw_margin = chosen_lp - rejected_lp

        valid = ~tie_mask
        vc = valid.sum().clamp(min=1)

    return {
        "total_loss": total_loss,
        "ntp_loss": ntp_loss,
        "mtp_loss": mtp_loss,
        "pair_acc": ((raw_margin > 0) & valid).float().sum() / vc,
        "raw_margin_mean": (raw_margin * valid).sum() / vc,
        "chosen_lr_mean": (chosen_lp * valid).sum() / vc,
        "rejected_lr_mean": (rejected_lp * valid).sum() / vc,
        "logp": logp_resp,  # (bs, 4)
    }
