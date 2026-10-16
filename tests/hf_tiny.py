"""Tiny randomly initialised Hugging Face models with locally trained tokenizers (no downloads needed)."""
import os
import shutil

import torch

TEXTS = ["heat the olive oil in a large pan .", "add garlic and onion , stir for two minutes .",
         "season with salt and serve warm .", "chop the onion .", "bake for twenty minutes at 350 degrees ."] * 12


def _gpt2_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(TEXTS, trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", bos_token="<|endoftext|>")


def make_gpt2(path, seed=0, n_positions=64):
    """A GPT2LMHeadModel (n_embd 16) saved as safetensors. GPT-2 has no pad token, like the real one."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    tok = _gpt2_tokenizer()
    cfg = GPT2Config(vocab_size=len(tok), n_positions=n_positions, n_embd=16, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=0)
    GPT2LMHeadModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def make_pickle_only(src, dst):
    """Copy a model folder but keep only a pickled pytorch_model.bin, like an old Hub checkpoint."""
    from safetensors.torch import load_file

    os.makedirs(dst, exist_ok=True)
    torch.save(load_file(os.path.join(src, "model.safetensors")), os.path.join(dst, "pytorch_model.bin"))
    for f in os.listdir(src):
        if f != "model.safetensors":
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    return dst


def make_bert(path, seed=0):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(TEXTS, trainers.WordPieceTrainer(vocab_size=120, special_tokens=specials))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]")
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=len(fast), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64)
    BertModel(cfg).save_pretrained(path)
    fast.save_pretrained(path)
    return path


def make_llama(path, seed=0, n_positions=256):
    """A tiny modern-architecture causal LM (RoPE, RMSNorm, grouped-query attention, tied embeddings), as in the Llama /
    Qwen / SmolLM families, whose tokenizer adds a beginning-of-text token and has no pad token."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(TEXTS, trainers.BpeTrainer(vocab_size=300, special_tokens=["<|begin_of_text|>", "<|end_of_text|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    bos = tok.token_to_id("<|begin_of_text|>")
    tok.post_processor = processors.TemplateProcessing(single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", bos)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>")
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=n_positions, tie_word_embeddings=True,
                      bos_token_id=bos, eos_token_id=tok.token_to_id("<|end_of_text|>"))
    LlamaForCausalLM(cfg).save_pretrained(path)
    fast.save_pretrained(path)
    return path
