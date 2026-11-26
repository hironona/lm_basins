"""common/joint_kl.py against brute-force enumeration of the |V|^m joint on a tiny random Qwen3."""

import itertools

import torch
from conftest import D_MODEL, EOS, VOCAB

from common.joint_kl import JointKL
from common.steering import context_state, continuation_log_probs, greedy_decode

IDS = torch.tensor([1, 5, 7, 2, 9])
LAYER = 1


def joint_log_probs(model, state, m):
    """log P(x_1..x_m) for every sequence in V^m with EOS absorbing (tokens after EOS must be EOS)."""
    seqs = list(itertools.product(range(VOCAB), repeat=m))
    lp_all = continuation_log_probs(model, state, torch.tensor([s[:-1] for s in seqs])) if m > 1 else None
    out = {}
    for n, s in enumerate(seqs):
        logp, ended = 0.0, False
        for t in range(m):
            if ended:
                logp = logp if s[t] == EOS else -float('inf')
                continue
            lp = state.log_probs[0] if t == 0 else lp_all[n, t - 1]
            logp += float(lp[s[t]])
            ended = s[t] == EOS
        out[s] = logp
    return out


def exact_kl(model, state_p, state_q, m):
    lp, lq = joint_log_probs(model, state_p, m), joint_log_probs(model, state_q, m)
    return sum(float(torch.tensor(a).exp()) * (a - lq[s]) for s, a in lp.items() if a > -float('inf'))


def test_joint_kl(tiny):
    model, _ = tiny
    m = 3
    gen = torch.Generator().manual_seed(0)
    offset = 2.0 * torch.randn(1, D_MODEL, generator=gen)
    state_p = context_state(model, IDS, LAYER)
    state_q = context_state(model, IDS, LAYER, offset)
    exact = torch.tensor([exact_kl(model, state_p, state_q, mm) for mm in range(1, m + 1)])

    # Complete tree: exact, no tail
    import common.joint_kl as jk
    threshold, jk.PROB_THRESHOLD = jk.PROB_THRESHOLD, 0.0
    try:
        full = JointKL(model, state_p, m, [EOS], max_nodes=10 ** 6, n_mc=64)
    finally:
        jk.PROB_THRESHOLD = threshold
    res = full(state_q)
    assert torch.allclose(res['kl'][0], exact, rtol=1e-4, atol=1e-6)
    assert torch.allclose(res['kl'], res['head'])
    assert full(context_state(model, IDS, LAYER, torch.zeros(1, D_MODEL)))['kl'].abs().max() == 0

    # Pruned tree: head is a lower bound, head + MC tail is within a few standard errors
    pruned = JointKL(model, state_p, m, [EOS], max_nodes=8, n_mc=4000, seed=1)
    res = pruned(state_q)
    assert (res['head'][0] <= exact + 1e-6).all()
    assert ((res['kl'][0] - exact).abs() <= 4 * res['tail_se'][0] + 1e-5).all()

    # Batched steered copies == one at a time (also for greedy decoding)
    offsets = torch.cat([offset, 0.5 * offset, torch.zeros(1, D_MODEL)])
    batch = context_state(model, IDS, LAYER, offsets)
    batched = pruned(batch)
    greedy = greedy_decode(model, batch, 5, [EOS])
    for b in range(3):
        single = context_state(model, IDS, LAYER, offsets[b:b + 1])
        assert torch.allclose(batched['kl'][b], pruned(single)['kl'][0], rtol=1e-4, atol=1e-6)
        assert greedy[b] == greedy_decode(model, single, 5, [EOS])[0]
    assert batched['kl'][2].abs().max() < 1e-6
