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
