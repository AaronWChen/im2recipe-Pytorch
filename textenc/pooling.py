import torch

POOLINGS = ("mean", "last", "first")


def pool(hidden, mask, mode):
    """Reduce token states to one vector per sequence.

    hidden (B, T, H); mask (B, T) with 1 for real tokens, right-padded, at least one real token per row.
      mean  average over real tokens          (a good default for any model)
      last  state of the last real token      (natural for causal models such as GPT-2)
      first state of the first token          (the [CLS] position of BERT-style models)
    """
    hidden = hidden.float()
    if mode == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    if mode == "first":
        return hidden[:, 0]
    if mode == "last":
        last = mask.long().sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]
    raise ValueError(f"unknown pooling {mode!r}; choose from {POOLINGS}")
