import torch


def cross_entropy(y_true, y_pred):
    log_probs = torch.log_softmax(y_pred, dim=-1)
    nll = -log_probs.gather(dim=-1, index=y_true.unsqueeze(-1)).squeeze(-1)
    return torch.mean(nll)
