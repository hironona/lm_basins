"""
Last-token forward passes from a source layer, for perturbing only the last position of a fixed context.

Layer convention (as everywhere in this project): layer L is the residual stream entering block L (0 = embeddings,
n_layers = output of the last block). By the causal mask the context positions do not depend on the last token, so
their activations are fixed and the last token can be pushed through block by block.
"""

from typing import Dict

import torch

from common.lm import ones_mask, resid_pre

JACOBIAN_CHUNK = 256  # output dims per vmap chunk in jacrev (lower = less memory)


@torch.no_grad()
def anchor_activations(model, tokens: torch.Tensor, layer: int):
    """
    Sentences tokens [A, T] that share all but the last token. Returns the last-token activations at `layer` [A, d],
    the shared context activations {k: [T-1, d]} entering every block k >= layer, and the last-position logits [A, V].
    The last-token forward pass is checked against the full model.
    """
    n_layers = model.cfg.n_layers
    names = {k: resid_pre(k) for k in range(layer, n_layers)}
    logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names.values(),
                                         attention_mask=ones_mask(tokens))
    contexts = {}
    for k, name in names.items():
        ctx = cache[name][:, :-1]
        assert torch.allclose(ctx, ctx[:1].expand_as(ctx), atol=1e-4), f"contexts differ at layer {k}"
        contexts[k] = ctx[0].clone()
    if layer < n_layers:
        anchors = cache[names[layer]][:, -1].clone()
    else:
        anchors = model(tokens, stop_at_layer=n_layers, attention_mask=ones_mask(tokens))[:, -1].clone()
    logits = logits[:, -1]
    err = (forward_from_layer(model, anchors, layer, contexts) - logits).abs().max().item()
    assert err < 1e-3 * max(1.0, logits.abs().max().item()), f"last-token forward differs from the model: {err}"
    return anchors, contexts, logits


def forward_from_layer(model, x: torch.Tensor, layer: int, contexts: Dict[int, torch.Tensor],
                       return_resid: bool = False) -> torch.Tensor:
    """Last-token activations x [B, d] at `layer` -> logits [B, V] (or the final residual [B, d])."""
    for k in range(layer, model.cfg.n_layers):
        seq = torch.cat([contexts[k][None].expand(x.shape[0], -1, -1), x[:, None]], dim=1)
        x = model.blocks[k](seq)[:, -1]
    return x if return_resid else model.unembed(model.ln_final(x[:, None]))[:, 0]


def unembed_gram_factor(model) -> torch.Tensor:
    """C [d, d] with C^T C = W_U W_U^T, so ||W_U^T M||_F = ||C M||_F without a [V, d] Jacobian per point."""
    w = model.W_U.detach().double()
    eigvals, eigvecs = torch.linalg.eigh(w @ w.T)
    return (eigvecs * eigvals.clamp(min=0).sqrt()).T


def jacobian_norm_to_logits(model, x: torch.Tensor, layer: int, contexts: Dict[int, torch.Tensor],
                            gram: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of d logits / d x (last token, `layer` -> logits) at each point x [B, d] -> [B]."""
    gram = gram.to(x.device, x.dtype)

    def f(z):  # samples are independent, so the Jacobian of the batch sum is the per-sample Jacobian
        resid = forward_from_layer(model, z, layer, contexts, return_resid=True)
        return (model.ln_final(resid[:, None])[:, 0] @ gram.T).sum(0)

    jac = torch.func.jacrev(f, chunk_size=JACOBIAN_CHUNK)(x)  # [d, B, d]
    return torch.linalg.matrix_norm(jac.permute(1, 0, 2).double(), ord='fro').detach()
