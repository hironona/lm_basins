"""Tiny randomly initialised Qwen3 models (HF and its TransformerLens conversion) for CPU tests."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

VOCAB, D_MODEL, EOS = 24, 32, 23


def tiny_qwen3(seed: int = 0):
    """(TransformerLens model, HF model) with the same weights; sharp next-token distributions so pruning matters."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.pretrained.weight_conversions import convert_qwen3_weights
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    hf_cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=D_MODEL, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    hf = Qwen3ForCausalLM(hf_cfg).eval()
    with torch.no_grad():
        hf.lm_head.weight.mul_(8.0)
    cfg = HookedTransformerConfig(
        d_model=D_MODEL, d_head=8, n_heads=4, n_key_value_heads=2, d_mlp=64, n_layers=3, n_ctx=64, d_vocab=VOCAB,
        act_fn='silu', normalization_type='RMS', positional_embedding_type='rotary', rotary_dim=8,
        rotary_base=int(hf_cfg.rope_theta), rotary_adjacent_pairs=False, final_rms=True, gated_mlp=True,
        use_qk_norm=True, eps=hf_cfg.rms_norm_eps, use_attn_scale=True, default_prepend_bos=False,
        original_architecture='Qwen3ForCausalLM', device='cpu')
    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(convert_qwen3_weights(hf, cfg), fold_ln=False, center_writing_weights=False,
                                      center_unembed=False, fold_value_biases=False,
                                      refactor_factored_attn_matrices=False)
    model.eval().requires_grad_(False)
    return model, hf


@pytest.fixture(scope='session')
def tiny():
    return tiny_qwen3()
