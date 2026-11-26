"""
KL divergence between the m-token joint continuations of a reference context P and steered contexts Q,

    D_m = KL( P(X_1..X_m) || Q(X_1..X_m) ),

without enumerating |V|^m sequences.

1. Chain rule: D_m = sum_{t=1..m} E_{x<t ~ P}[ KL(P(.|x<t) || Q(.|x<t)) ]. The inner KL over the vocabulary is exact,
   so only the expectation over prefixes needs approximating. D_1 is exact, and every D_m' (m' <= m) comes out of
   the same computation.
2. Head: a probability-pruned prefix tree of P (each depth keeps prefixes with P(prefix) >= PROB_THRESHOLD, among the
   TOP_K children of their parent and the `max_nodes` most probable of the depth). EOS-ended prefixes are absorbing.
   The head sum over the tree is exact and a lower bound on D_m.
3. Tail: Monte Carlo over the mass outside the tree with `n_mc` ancestral samples of P (fixed seed); head + tail is
   unbiased for D_m, and the tail's standard error is reported.
4. Evaluation: every prefix needing a distribution ("slot") is covered by a small set of maximal paths, run
   teacher-forced on the context's KV cache. P's log-probs go through the same code path as Q's, so D_m = 0 exactly at
   zero steering.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from common.steering import ContextState, continuation_log_probs

PROB_THRESHOLD = 1e-4
TOP_K = 64
CHUNK = 256  # sequences per forward pass


@dataclass
class PrefixTree:
    """Prefixes of P's continuation; node 0 is the empty prefix."""
    prefixes: List[Tuple[int, ...]]
    terminal: List[bool]  # ends with EOS
    index: Dict[Tuple[int, ...], int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {p: i for i, p in enumerate(self.prefixes)}


@torch.no_grad()
def build_prefix_tree(model, state: ContextState, m: int, eos: List[int], max_nodes: int) -> PrefixTree:
    """Breadth-first, probability-pruned prefix tree of P (a single-copy state) up to depth m-1."""
    log_threshold = float(torch.log(torch.tensor(PROB_THRESHOLD)))
    prefixes, terminal, log_joint = [()], [False], [0.0]
    frontier, frontier_lp = [0], state.log_probs  # [F, V]
    for depth in range(1, m):
        joint = torch.tensor([log_joint[i] for i in frontier], device=frontier_lp.device)[:, None] + frontier_lp
        vals, toks = joint.topk(min(TOP_K, joint.shape[1]), dim=1)
        keep = vals >= log_threshold
        rows = torch.arange(joint.shape[0], device=joint.device)[:, None].expand_as(toks)
        rows, toks, vals = rows[keep], toks[keep], vals[keep]
        order = vals.argsort(descending=True)[:max_nodes]
        new_frontier = []
        for r, tok, v in zip(rows[order].tolist(), toks[order].tolist(), vals[order].tolist()):
            prefixes.append(prefixes[frontier[r]] + (tok,))
            terminal.append(tok in eos)
            log_joint.append(v)
            if tok not in eos:
                new_frontier.append(len(prefixes) - 1)
        frontier = new_frontier
        if depth == m - 1 or not frontier:
            break
        paths = torch.tensor([prefixes[i] for i in frontier], device=frontier_lp.device)
        frontier_lp = torch.cat([continuation_log_probs(model, state, paths[s:s + CHUNK])[:, -1]
                                 for s in range(0, len(frontier), CHUNK)])
    return PrefixTree(prefixes, terminal)


@torch.no_grad()
def sample_paths(model, state: ContextState, length: int, n: int, eos: List[int], seed: int) -> torch.Tensor:
    """n ancestral samples [n, length] from a single-copy state (Gumbel-max, noise fixed by seed and step)."""
    device, vocab = state.log_probs.device, state.log_probs.shape[1]
    if length == 0:
        return torch.zeros(n, 0, dtype=torch.long, device=device)
    eos_t = torch.tensor(eos, device=device)
    cache, log_probs = state.cache(n), state.log_probs.expand(n, -1)
    tokens, done = [], torch.zeros(n, dtype=torch.bool, device=device)
    for step in range(length):
        gen = torch.Generator(device=device).manual_seed(seed * 1_000_003 + step)
        u = torch.rand(n, vocab, generator=gen, device=device).clamp_(1e-12, 1 - 1e-7)
        tok = torch.where(done, eos_t[0], (log_probs - torch.log(-torch.log(u))).argmax(-1))
        tokens.append(tok)
        done |= torch.isin(tok, eos_t)
        if step < length - 1:
            logits = model(tok[:, None], past_kv_cache=cache, attention_mask=torch.ones_like(tok[:, None], dtype=torch.int))
            log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1)
    return torch.stack(tokens, dim=1)


def _until_eos(path: Tuple[int, ...], eos: List[int]) -> Tuple[int, ...]:
    for j, t in enumerate(path):
        if t in eos:
            return path[:j + 1]
    return path


class JointKL:
    """
    Reference side: P's prefix tree, MC samples, the evaluation plan and P's log-probs at every slot.
    Call it on a (batched) steered state to get D_m' for every m' <= m.
    """

    def __init__(self, model, state_p: ContextState, m: int, eos: List[int], max_nodes: int = 256, n_mc: int = 256,
                 seed: int = 0, reference: Dict = None):
        """reference: {'tree', 'mc_paths'} from `reference()`, to rebuild the same estimator without resampling."""
        assert state_p.batch == 1
        self.model, self.m = model, m
        device = state_p.log_probs.device
        if reference is None:
            self.tree = build_prefix_tree(model, state_p, m, eos, max_nodes)
            self.mc_paths = sample_paths(model, state_p, m - 1, n_mc, eos, seed)
        else:
            self.tree, self.mc_paths = reference['tree'], reference['mc_paths'].to(device)
        self._make_plan(eos, device)

        self.lp_p = torch.empty(len(self.slots), state_p.log_probs.shape[1], device=device)
        for ids, lp in self._slot_log_probs(state_p):
            self.lp_p[ids] = lp[0]
        self.p_p = self.lp_p.exp()
        self.neg_entropy = (self.p_p * self.lp_p).sum(-1)  # [S]

        # Head weights P(prefix) for every non-terminal tree prefix, from the evaluated log-probs
        head = [(self.slot_index[p], len(p), sum(float(self.lp_p[self.slot_index[p[:j]], p[j]]) for j in range(len(p))))
                for p, term in zip(self.tree.prefixes, self.tree.terminal) if not term and len(p) <= m - 1]
        self.head_slot = torch.tensor([h[0] for h in head], device=device)
        self.head_depth = torch.tensor([h[1] for h in head], device=device)
        self.head_weight = torch.tensor([h[2] for h in head], device=device).exp()

    def _make_plan(self, eos: List[int], device):
        """Slots (tree prefixes + MC tail prefixes) and the maximal paths, grouped by length, that produce them."""
        m = self.m
        slots = [p for p, term in zip(self.tree.prefixes, self.tree.terminal) if not term and len(p) <= m - 1]
        slot_index = {p: i for i, p in enumerate(slots)}
        tail_slot = torch.full((self.mc_paths.shape[0], m), -1, dtype=torch.long)  # step t uses the length-t prefix
        for s, path in enumerate(self.mc_paths.tolist()):
            path = _until_eos(tuple(path), eos)
            for t in range(1, m):
                prefix = path[:t]
                if len(prefix) < t or prefix[-1] in eos:
                    break  # the sample ended: contributes 0
                if prefix in self.tree.index:
                    continue  # counted in the head
                if prefix not in slot_index:
                    slot_index[prefix] = len(slots)
                    slots.append(prefix)
                tail_slot[s, t] = slot_index[prefix]

        inner = {p[:j] for p in slots for j in range(1, len(p))}
        by_len: Dict[int, List[Tuple[int, ...]]] = {}
        for p in slots:
            if p and p not in inner:
                by_len.setdefault(len(p), []).append(p)
        assigned, groups = {()}, []
        for length in sorted(by_len):
            rows, pos, ids = [], [], []
            for r, p in enumerate(by_len[length]):
                for j in range(length):
                    if p[:j + 1] in slot_index and p[:j + 1] not in assigned:
                        assigned.add(p[:j + 1])
                        rows.append(r), pos.append(j), ids.append(slot_index[p[:j + 1]])
            groups.append(tuple(torch.tensor(a, device=device) for a in (by_len[length], rows, pos, ids)))
        assert len(assigned) == len(slots)
        self.slots, self.slot_index, self.groups = slots, slot_index, groups
        self.tail_slot = tail_slot.to(device)

    def _slot_log_probs(self, state: ContextState):
        """Yield (slot ids [S_c], log-probs [B, S_c, V]) covering every slot once."""
        b = state.batch
        yield torch.tensor([0], device=state.log_probs.device), state.log_probs[:, None]
        per = max(1, CHUNK // b)  # paths per forward
        for paths, rows, pos, ids in self.groups:
            for start in range(0, paths.shape[0], per):
                sel = (rows >= start) & (rows < start + per)
                if sel.any():
                    sub = paths[start:start + per]
                    lp = continuation_log_probs(self.model, state, sub).view(b, sub.shape[0], sub.shape[1], -1)
                    yield ids[sel], lp[:, rows[sel] - start, pos[sel]]

    def reference(self) -> Dict:
        return {'tree': PrefixTree(self.tree.prefixes, self.tree.terminal), 'mc_paths': self.mc_paths.cpu()}

    def covered_mass(self) -> List[float]:
        """P-mass of the non-terminal tree prefixes at each step t = 1..m (how much the exact head covers)."""
        return [float(self.head_weight[self.head_depth == t].sum()) for t in range(self.m)]

    @torch.no_grad()
    def __call__(self, state_q: ContextState) -> Dict[str, torch.Tensor]:
        """{'kl': D_m' for m' = 1..m, 'head': its exact lower bound, 'tail_se': std. error}, each [B, m] on CPU."""
        b = state_q.batch
        kl_slot = torch.empty(b, len(self.slots), device=self.lp_p.device)
        for ids, lq in self._slot_log_probs(state_q):
            kl_slot[:, ids] = (self.neg_entropy[ids][None] - (self.p_p[ids][None] * lq).sum(-1)).clamp_min(0.0)
        head = torch.zeros(b, self.m, device=kl_slot.device)
        head.index_add_(1, self.head_depth, self.head_weight[None] * kl_slot[:, self.head_slot])
        tail = torch.where(self.tail_slot[None] >= 0, kl_slot[:, self.tail_slot.clamp_min(0)],
                           torch.zeros((), device=kl_slot.device))  # [B, N, m]
        n = tail.shape[1]
        tail_se = tail.cumsum(2).std(1) / n ** 0.5 if n > 1 else torch.zeros(b, self.m, device=kl_slot.device)
        out = {'kl': (head + tail.mean(1)).cumsum(1), 'head': head.cumsum(1), 'tail_se': tail_se}
        return {k: v.cpu() for k, v in out.items()}
