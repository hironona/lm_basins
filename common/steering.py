"""
Steered forward passes with a reusable KV cache (TransformerLens).

Steering adds an offset to the residual stream entering block L (`blocks.L.hook_resid_pre`) at the context positions
of a scope: 'context' = every context position 0 .. n_ctx-1, 'last' = only n_ctx-1. Tokens after the context
(decoded or teacher-forced) are never steered: steering is applied only in passes that start at position 0, and
continuations run on the cache without hooks.

A ContextState holds the KV cache and next-token log-probs of B (possibly steered) copies of one token sequence.
Continuations of K token paths are evaluated on its cache expanded to B*K rows (row b*K + k: copy b, path k).
"""

from typing import Dict, List, Optional, Tuple

import torch
from transformer_lens.past_key_value_caching import (HookedTransformerKeyValueCache,
                                                      HookedTransformerKeyValueCacheEntry)

from common.lm import ones_mask, resid_pre

SCOPES = ('context', 'last')


class ContextState:
    def __init__(self, kv: List[Tuple[torch.Tensor, torch.Tensor]], log_probs: torch.Tensor):
        self.kv = kv  # per layer (keys, values), each [B, n, heads, d_head]
        self.log_probs = log_probs  # [B, V] float32, next-token log-probs after the last position

    @property
    def batch(self) -> int:
        return self.log_probs.shape[0]

    @property
    def length(self) -> int:
        return self.kv[0][0].shape[1]

    def select(self, rows) -> 'ContextState':
        return ContextState([(k[rows], v[rows]) for k, v in self.kv], self.log_probs[rows])

    def cache(self, k: int = 1) -> HookedTransformerKeyValueCache:
        """A fresh cache with B * k rows (each copy repeated k times; no copy when B = 1)."""
        def rep(t):
            return t.expand(k, -1, -1, -1) if self.batch == 1 else t.repeat_interleave(k, 0)
        entries = [HookedTransformerKeyValueCacheEntry(rep(kk), rep(vv)) for kk, vv in self.kv]
        mask = torch.ones(self.batch * k, self.length, dtype=torch.int, device=self.log_probs.device)
        return HookedTransformerKeyValueCache(entries, mask)


def offset_positions(n_ctx: int, scope: str) -> slice:
    assert scope in SCOPES, scope
    return slice(0, n_ctx) if scope == 'context' else slice(n_ctx - 1, n_ctx)


@torch.no_grad()
def prefix_states(model, ids: torch.Tensor, n_ctx: int, layer: int, offsets: Optional[torch.Tensor] = None,
                  scope: str = 'context', capture: bool = False):
    """
    One pass over ids [T] = a context of n_ctx tokens followed by fixed tokens y_1..y_J, with offsets [B, d] (B steered
    copies; None = one unsteered copy) added at block `layer` over the context positions of `scope`. Returns
        states: [state_0, ..., state_J], state_j = the context followed by y_1..y_j (B copies);
        acts:   {'src': residual entering block `layer` after steering, 'final': residual leaving the last block},
                each [B, T, d] float32, when `capture` (else None).
    Steering acts only at positions < n_ctx and attention is causal, so state_j equals a pass over ids[:n_ctx + j].
    """
    b = 1 if offsets is None else offsets.shape[0]
    tokens = ids[None].expand(b, -1)
    pos = offset_positions(n_ctx, scope)
    acts: Dict[str, torch.Tensor] = {}

    def steer(resid, hook):
        if offsets is not None:
            resid = resid.clone()
            resid[:, pos] += offsets.to(resid.dtype)[:, None]
        if capture:
            acts['src'] = resid.float()
        return resid

    cache = HookedTransformerKeyValueCache.init_cache(model.cfg, model.cfg.device, b)
    with model.hooks(fwd_hooks=[(resid_pre(layer), steer)]):
        final = model(tokens, stop_at_layer=model.cfg.n_layers, past_kv_cache=cache,
                      attention_mask=ones_mask(tokens))  # [B, T, d], residual leaving the last block
    if capture:
        acts['final'] = final.float()
    # Unembed only the positions whose next-token distribution is needed (n_ctx-1 .. T-1)
    log_probs = torch.log_softmax(model.unembed(model.ln_final(final[:, n_ctx - 1:])).float(), dim=-1)
    kv = [(e.past_keys, e.past_values) for e in cache.entries]
    states = [ContextState([(k[:, :n_ctx + j], v[:, :n_ctx + j]) for k, v in kv], log_probs[:, j])
              for j in range(ids.shape[0] - n_ctx + 1)]
    return states, (acts if capture else None)


def context_state(model, ids: torch.Tensor, layer: int, offsets: Optional[torch.Tensor] = None,
                  scope: str = 'context') -> ContextState:
    """State of the context ids [n] alone (B steered copies, or one unsteered copy)."""
    return prefix_states(model, ids, ids.shape[0], layer, offsets, scope)[0][0]


@torch.no_grad()
def continuation_log_probs(model, state: ContextState, paths: torch.Tensor) -> torch.Tensor:
    """
    Teacher-forced next-token log-probs after each token of paths [K, l] following the state: [B * K, l, V] float32
    (row b*K + k: copy b, path k); [:, t] is the distribution after paths[:, :t+1].
    """
    k = paths.shape[0]
    tokens = paths if state.batch == 1 else paths.repeat(state.batch, 1)
    logits = model(tokens, past_kv_cache=state.cache(k), attention_mask=ones_mask(tokens))
    return torch.log_softmax(logits.float(), dim=-1)


@torch.no_grad()
def greedy_decode(model, state: ContextState, n_tokens: int, eos: List[int]) -> List[List[int]]:
    """Greedy continuation of every copy (unsteered continuation), each cut after its first EOS."""
    log_probs, cache = state.log_probs, state.cache(1)
    eos_t = torch.tensor(eos, device=log_probs.device)
    done = torch.zeros(state.batch, dtype=torch.bool, device=log_probs.device)
    steps = []
    for step in range(n_tokens):
        tok = torch.where(done, eos_t[0], log_probs.argmax(-1))
        steps.append(tok)
        done |= torch.isin(tok, eos_t)
        if step == n_tokens - 1 or bool(done.all()):
            break
        logits = model(tok[:, None], past_kv_cache=cache, attention_mask=ones_mask(tok[:, None]))
        log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1)
    out = []
    for row in torch.stack(steps, dim=1).tolist():
        cut = next((j + 1 for j, t in enumerate(row) if t in eos), len(row))
        out.append(row[:cut])
    return out


def orthogonal_directions(v: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """k random directions [k, d], orthonormal to each other and orthogonal to v, each scaled to ||v||."""
    gen = torch.Generator().manual_seed(seed)
    v64 = v.detach().double().cpu()
    v_hat = v64 / v64.norm()
    r = torch.randn(v64.shape[0], k, generator=gen, dtype=torch.float64)
    r -= v_hat[:, None] * (v_hat @ r)[None]
    q, _ = torch.linalg.qr(r)
    q -= v_hat[:, None] * (v_hat @ q)[None]
    q /= q.norm(dim=0, keepdim=True)
    return (q.T * v64.norm()).to(v.device, v.dtype)
