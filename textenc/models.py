import torch


def load_encoder(name_or_path, device="cpu", allow_pickle=False, half=False):
    """Load (model, tokenizer) for a Hugging Face model id or a local folder.

    Weights must be safetensors unless allow_pickle=True: a pickled checkpoint can run code when it is
    loaded, so the default is to refuse it. Convert an old model once with `python -m textenc.to_safetensors`.
    Language-model heads are dropped (GPT2LMHeadModel -> GPT2Model); encoder-decoder models (T5, BART) use
    their encoder.
    """
    from transformers import AutoModel, AutoTokenizer

    try:
        model = AutoModel.from_pretrained(name_or_path, use_safetensors=None if allow_pickle else True)
    except OSError as e:
        if not allow_pickle and "safetensors" in str(e).lower():
            raise OSError(
                f"{name_or_path} has no safetensors weights, and pickled weights are refused by default. Convert it "
                f"once with  python -m textenc.to_safetensors {name_or_path} --out models/<name> --allow-pickle  "
                "(only for models you trust) and use that folder, or pass --allow-pickle.") from e
        raise
    if getattr(model.config, "is_encoder_decoder", False):
        model = model.get_encoder()
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = model.eval().to(device)
    return (model.half() if half else model), tokenizer


def max_positions(model):
    cfg = model.config
    return getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)


def pad_token_id(tokenizer):
    for tid in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if tid is not None:
            return tid
    return 0


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device="cpu", pooling="mean", max_length=None, batch_size=64, layer=-1,
                 l2_normalize=False, prefix="", suffix=""):
    """Encode a list of strings -> float32 array (len(texts), hidden_size), in the order given.

    Texts are tokenized by the model's own tokenizer, truncated to max_length tokens, sorted by length so
    batches carry little padding, and padded on the right with an explicit attention mask (so a text's vector
    does not depend on the other texts in its batch). An empty text becomes a single end-of-text token.
    max_length defaults to 128, or fewer if the model has fewer positions. prefix / suffix are added to every text first
    (some embedding models expect one, e.g. "passage: " for e5; check the model card).
    """
    from .pooling import pool

    limit = max_positions(model)
    if max_length is None:
        max_length = min(128, limit or 128)
    if limit is not None and max_length > limit:
        raise ValueError(f"max_length {max_length} exceeds the model's {limit} positions")
    texts = [prefix + t + suffix for t in texts] if (prefix or suffix) else list(texts)
    ids = tokenizer(list(texts), add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    fallback = pad_token_id(tokenizer)
    ids = [x if len(x) else [fallback] for x in ids]
    pad = pad_token_id(tokenizer)

    out = None
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        width = max(len(ids[i]) for i in idx)
        batch = torch.full((len(idx), width), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            batch[r, :len(ids[i])] = torch.tensor(ids[i])
            mask[r, :len(ids[i])] = 1
        batch, mask = batch.to(device), mask.to(device)
        if layer == -1:
            hidden = model(input_ids=batch, attention_mask=mask).last_hidden_state
        else:
            hidden = model(input_ids=batch, attention_mask=mask, output_hidden_states=True).hidden_states[layer]
        vec = pool(hidden, mask, pooling)
        if l2_normalize:
            vec = torch.nn.functional.normalize(vec, dim=-1)
        if out is None:
            out = torch.empty((len(ids), vec.size(1)), dtype=torch.float32)
        out[torch.tensor(idx)] = vec.cpu()
    return out.numpy() if out is not None else None
