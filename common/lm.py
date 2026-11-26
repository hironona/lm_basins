"""
TransformerLens model loading, tokenization and residual-stream hook names.

Models are loaded with `from_pretrained_no_processing` (no LayerNorm folding, no weight centering), so the residual
stream is exactly the one of the Hugging Face model: `blocks.{k}.hook_resid_pre` equals HF's `hidden_states[k]`.
Set HF_HUB_CACHE (and HF_HUB_OFFLINE=1 on nodes without internet) to load from a local cache.
"""

from typing import List

import torch

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


def load_model(name: str, device: str, dtype: str = 'float32'):
    """HookedTransformer in eval mode without gradients (float32 on CPU)."""
    from transformer_lens import HookedTransformer
    if device == 'cpu':
        dtype = 'float32'
    model = HookedTransformer.from_pretrained_no_processing(name, device=device, dtype=DTYPES[dtype])
    model.eval()
    model.requires_grad_(False)
    return model


def resid_pre(layer: int) -> str:
    """Hook name of the residual stream entering block `layer` (layer 0: the embeddings)."""
    return f'blocks.{layer}.hook_resid_pre'


def encode(model, text: str, prepend_bos: bool = False) -> torch.Tensor:
    """Token ids [n] of `text`, without special tokens except an optional BOS."""
    ids = model.tokenizer(text, add_special_tokens=False)['input_ids']
    if prepend_bos:
        ids = [model.tokenizer.bos_token_id] + ids
    return torch.tensor(ids, device=model.cfg.device)


def chat_prompt(model, text: str) -> str:
    """A single user turn with the assistant generation prompt (Qwen3: thinking disabled)."""
    messages = [{'role': 'user', 'content': text}]
    try:
        return model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
    except TypeError:  # templates without the enable_thinking switch
        return model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def eos_ids(model) -> List[int]:
    """Tokens that end a response (Qwen3: <|im_end|> and <|endoftext|>)."""
    tok = model.tokenizer
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for name in ('<|im_end|>', '<|endoftext|>'):
        tid = tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(ids)


def ones_mask(tokens: torch.Tensor) -> torch.Tensor:
    """
    All-ones attention mask. Passing it explicitly keeps TransformerLens from inferring padding from pad-token ids
    (Qwen3's pad token is also an EOS token, which can appear in continuations).
    """
    return torch.ones(tokens.shape, dtype=torch.int, device=tokens.device)


@torch.no_grad()
def block_input(model, tokens: torch.Tensor, layer: int) -> torch.Tensor:
    """Residual stream entering block `layer` for tokens [B, n] -> [B, n, d] (right padding is harmless)."""
    return model(tokens, stop_at_layer=layer, attention_mask=ones_mask(tokens))
