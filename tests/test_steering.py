"""Steered cached passes (common/steering.py), the norm ratio and the TransformerLens conversion of Qwen3."""

import pytest
import torch
from conftest import D_MODEL, EOS, VOCAB

from common.joint_kl import JointKL
from common.lm import ones_mask, resid_pre
from common.metrics import NormRatio, project_onto
from common.steering import continuation_log_probs, offset_positions, prefix_states

LAYER, N_CTX = 1, 6


def data(seed):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB - 1, (N_CTX + 3,), generator=gen)
    return ids, torch.randn(3, D_MODEL, generator=gen), gen


def steered_logits(model, tokens, offsets, scope):
    """Full forward (no cache) with the offset added at block LAYER over the scope's context positions."""
    pos = offset_positions(N_CTX, scope)

    def hook(resid, hook):
        resid = resid.clone()
        resid[:, pos] += offsets[:, None]
        return resid

    with model.hooks(fwd_hooks=[(resid_pre(LAYER), hook)]):
        return model(tokens, attention_mask=ones_mask(tokens))


def test_matches_hf(tiny):
    model, hf = tiny
    ids, _, _ = data(0)
    with torch.no_grad():
        assert torch.allclose(model(ids[None]), hf(ids[None]).logits, atol=1e-4)


@pytest.mark.parametrize('scope', ['context', 'last'])
def test_scope(tiny, scope):
    model, _ = tiny
    ids, offsets, _ = data(1)
    _, plain = prefix_states(model, ids, N_CTX, LAYER, capture=True)
    _, steered = prefix_states(model, ids, N_CTX, LAYER, offsets, scope, capture=True)
    diff = steered['src'] - plain['src']
    steered_pos = range(N_CTX) if scope == 'context' else [N_CTX - 1]
    for t in range(ids.shape[0]):
        want = offsets if t in steered_pos else torch.zeros_like(offsets)
        assert torch.allclose(diff[:, t], want, atol=1e-5), t


@pytest.mark.parametrize('scope', ['context', 'last'])
def test_prefix_states(tiny, scope):
    model, _ = tiny
    ids, offsets, gen = data(2)
    states, acts = prefix_states(model, ids, N_CTX, LAYER, offsets, scope, capture=True)
    assert len(states) == 4 and acts['final'].shape == (3, N_CTX + 3, D_MODEL)
    paths = torch.randint(0, VOCAB, (5, 2), generator=gen)
    for j, st in enumerate(states):
        assert st.length == N_CTX + j and st.batch == 3
        direct = steered_logits(model, ids[:N_CTX + j][None].expand(3, -1), offsets, scope)
        assert torch.allclose(st.log_probs, torch.log_softmax(direct[:, -1], -1), atol=1e-5)
        # Continuations on the sliced cache == teacher-forced full forward
        cont = continuation_log_probs(model, st, paths).view(3, 5, 2, VOCAB)
        for b in range(3):
            tokens = torch.cat([ids[:N_CTX + j][None].expand(5, -1), paths], 1)
            full = steered_logits(model, tokens, offsets[b:b + 1].expand(5, -1), scope)
            assert torch.allclose(cont[b], torch.log_softmax(full[:, N_CTX + j:], -1), atol=1e-4)
    # The captured final residual gives the model's logits
    logits = model.unembed(model.ln_final(acts['final']))
    assert torch.allclose(logits, steered_logits(model, ids[None].expand(3, -1), offsets, scope), atol=1e-4)

    # Joint KL on batched prefix states == single states; 0 at zero offset
    unsteered, _ = prefix_states(model, ids, N_CTX, LAYER)
    for j in (1, 3):
        jkl = JointKL(model, unsteered[j], 3, [EOS], max_nodes=30, n_mc=16)
        kl = jkl(states[j])['kl']
        for b in range(3):
            single, _ = prefix_states(model, ids, N_CTX, LAYER, offsets[b:b + 1], scope)
            assert torch.allclose(kl[b], jkl(single[j])['kl'][0], rtol=1e-4, atol=1e-7)
        zero, _ = prefix_states(model, ids, N_CTX, LAYER, torch.zeros(1, D_MODEL), scope)
        assert jkl(zero[j])['kl'].abs().max() < 1e-6


def test_norm_ratio():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(8, generator=gen) * 3 + 5
    h = x + torch.randn(50, 8, generator=gen)
    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=gen, dtype=torch.float64))

    def f(a):  # two affine maps (rotation + scale, plain scale), one per output position
        return torch.stack([7.0 * (a.double() @ q.T) + 1.0, 0.1 * a.double() - 2.0], dim=-2).float()

    out, x_out = f(h), f(x)
    assert NormRatio(x, x_out, h, out)(h, out).abs().max() < 1e-5  # affine maps keep relative sizes
    out[0] = x_out + 2 * (out[0] - x_out)
    ratio = NormRatio(x, x_out, h, out)
    assert (ratio(h, out)[0] > 0.1).all()  # pushed twice as far: expansion
    assert torch.isnan(ratio(x[None], x_out[None])).all()


def test_project_onto():
    u, v = torch.tensor([2.0, 0, 0]), torch.tensor([0, 3.0, 0])
    s = 0.5 * u - 2 * v + torch.tensor([0, 0, 7.0])
    assert torch.allclose(project_onto(s, torch.stack([u, v])).float(), torch.tensor([0.5, -2.0]))
