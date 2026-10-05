"""Skip-thought model (UniSkip) for current PyTorch.

Ported from sanyam5/skip-thoughts (PyTorch 0.3 era). Differences:
  * no Variable / .cuda(id) / hard-coded device; works on any device
  * the encoder uses packed sequences, so a sentence's vector does not depend
    on padding or on what else is in the batch (the original reversed padded
    sequences as a workaround)
  * the loss mask is applied to the per-token loss, not to the logits
  * (previous, next) pairs that cross a recipe boundary can be masked out
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, word_size=620, thought_size=1024):
        super().__init__()
        self.thought_size = thought_size
        self.embedding = nn.Embedding(vocab_size, word_size)
        self.lstm = nn.LSTM(word_size, thought_size, batch_first=True)

    def forward(self, tokens, lengths):
        """tokens (B, T) long, lengths (B,) long >= 1 -> thoughts (B, H), embeddings (B, T, W)."""
        emb = torch.tanh(self.embedding(tokens))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return h[-1], emb


class DuoDecoder(nn.Module):
    """Predicts the previous and the next sentence from one sentence's thought."""

    def __init__(self, vocab_size, word_size=620, thought_size=1024):
        super().__init__()
        self.prev_lstm = nn.LSTM(thought_size + word_size, word_size, batch_first=True)
        self.next_lstm = nn.LSTM(thought_size + word_size, word_size, batch_first=True)
        self.out = nn.Linear(word_size, vocab_size)

    def _run(self, lstm, thoughts, target_emb):
        # Teacher forcing: the input at step t is the embedding of target word t-1
        # (zeros at step 0); the thought is supplied at every step.
        n, t, _ = target_emb.shape
        delayed = torch.cat([torch.zeros_like(target_emb[:, :1]), target_emb[:, :-1]], dim=1)
        context = thoughts.unsqueeze(1).expand(n, t, thoughts.size(-1))
        hidden, _ = lstm(torch.cat([context, delayed], dim=2))
        return self.out(hidden)  # (n, t, vocab)

    def forward(self, thoughts, emb):
        # sentence i+1's thought predicts sentence i, and sentence i's predicts i+1
        prev_logits = self._run(self.prev_lstm, thoughts[1:], emb[:-1])
        next_logits = self._run(self.next_lstm, thoughts[:-1], emb[1:])
        return prev_logits, next_logits


def _masked_ce(logits, targets, lengths, pair_valid):
    n, t, v = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, v), targets.reshape(-1), reduction="none").view(n, t)
    steps = torch.arange(t, device=lengths.device)
    mask = ((steps[None, :] < lengths[:, None]) & pair_valid[:, None]).to(ce.dtype)
    return (ce * mask).sum() / mask.sum().clamp(min=1.0)


class UniSkip(nn.Module):
    def __init__(self, vocab_size, word_size=620, thought_size=1024):
        super().__init__()
        self.encoder = Encoder(vocab_size, word_size, thought_size)
        self.decoder = DuoDecoder(vocab_size, word_size, thought_size)

    def encode(self, tokens, lengths):
        thoughts, _ = self.encoder(tokens, lengths)
        return thoughts

    def forward(self, tokens, lengths, pair_valid=None):
        """Training loss for a window of consecutive sentences (B >= 2)."""
        if pair_valid is None:
            pair_valid = torch.ones(tokens.size(0) - 1, dtype=torch.bool, device=tokens.device)
        thoughts, emb = self.encoder(tokens, lengths)
        prev_logits, next_logits = self.decoder(thoughts, emb)
        prev_loss = _masked_ce(prev_logits, tokens[:-1], lengths[:-1], pair_valid)
        next_loss = _masked_ce(next_logits, tokens[1:], lengths[1:], pair_valid)
        return prev_loss + next_loss
