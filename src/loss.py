import torch


def cross_entropy(y_true, y_pred):
    log_probs = torch.log_softmax(y_pred, dim=-1)
    nll = -log_probs.gather(dim=-1, index=y_true.unsqueeze(-1)).squeeze(-1)
    return torch.mean(nll)


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
        log_probs_chunk = y_pred_flat[mask, start:end] - max_val[mask] - log_sum_exp[mask]
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
        return torch.tensor(0.0, device=index_score.device if index_score is not None else
                           (weight_compress.device if weight_compress is not None else
                            torch.cuda.current_device()))

    b, s, n_heads, topk = weight_compress.shape

    # Target: sum attention weights across heads, L1 normalize → p_{t,S_t}
    target = weight_compress.sum(dim=2)  # (b, s, topk)
    target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)

    # Predicted: gather index_score at selected indices, softmax
    mask = compress_topk_idxs == -1
    idx = compress_topk_idxs.clamp(0)
    pred = index_score.gather(-1, idx)  # (b, s, topk)
    pred = pred.masked_fill(mask, float("-inf"))

    # KL(target || softmax(pred))
    log_pred = torch.log_softmax(pred, dim=-1)  # (b, s, topk)
    kl = target * (torch.log(target + 1e-8) - log_pred)  # (b, s, topk)
    kl = kl.masked_fill(mask | torch.isnan(kl), 0.0)
    kl = kl.sum(dim=-1)  # (b, s)

    # Only count positions that attend to at least one compressed block
    valid = target.sum(dim=-1) > 1e-8  # (b, s)
    return kl[valid].mean() if valid.any() else torch.tensor(0.0)
