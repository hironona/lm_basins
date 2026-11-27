#!/usr/bin/env python3
"""
Steering basins (GPU). For n_contexts contexts per experiment, drawn from held-out data:
    vectors  steering vectors (en_zh, refusal) and the context draw -> vectors.pt, contexts.json
    basin    per context: basin check, the unsteered greedy tokens y_1..y_J, and one joint-KL reference after the
             context and after each y_j -> <exp>/<ctx>/basin.pt
    grid     per context and plane k: offsets a v + b u_k at the context positions (y_j teacher-forced, never steered);
             at j = 0..J the m-token joint KL, the greedy label, the log norm ratio (block-L input -> last block output)
             and the in-plane residual shift -> plane_XX.pt
    diff     per context, steered point p (diff_points) and plane k: the steered greedy tokens z_1..z_J (steering
             alpha_p v) and the unsteered y_1..y_J give, at each j, d_j = h_j - x_j (layer-L input at the last position
             of ctx + z_<=j steered minus ctx + y_<=j unsteered) and cos(d_j, v). Offsets a d_j + b u_k (u_k random,
             ⟂ d_j, ||u_k|| = ||d_j||) replace the unsteered x_j at that position only (scope 'last' on the prefix
             ctx + y_<=j; (a, b) = (1, 0) is h_j); the same measures as the grid -> diff_<p>_plane_XX.pt
             (run in the grid stage; d_j comes from the basin stage)
    merge    plane means -> result.pt, summary.json
`--stage all` runs everything. For several GPUs: --stage vectors, then one process per GPU with
--stage basin and then --stage grid (--shard i --num-shards n), then --stage merge. Run from the project root:
    python experiments/steering_basins/run.py --data-dir data --output-dir outputs/steering_basins [--quick]
"""

import argparse
import glob
import itertools
import json
import os
import secrets
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..')), HERE]
from basins import PROBE_DEPTH, basin_check  # noqa: E402
from common.config import get_device, load_config  # noqa: E402
from common.joint_kl import JointKL  # noqa: E402
from common.lm import block_input, chat_prompt, encode, eos_ids, load_model  # noqa: E402
from common.metrics import NormRatio, project_onto  # noqa: E402
from common.steering import context_state, greedy_decode, orthogonal_directions, prefix_states  # noqa: E402
from labels import CLASSIFIER_OF_TARGET, classify  # noqa: E402
from vectors import steering_vectors  # noqa: E402

QUICK = dict(n_docs=4, n_prompts=4, alpha_max=4, resolution=4, n_planes=2, n_decoded=2, m=4, tree_nodes=16,
             mc_samples=8)
EXTENT_FACTOR = 1.5  # grid half-width = 1.5 x the farthest steered point / end of the target run on the ray
LABEL_TOKENS = 8  # greedy tokens decoded at every grid point for its behavior label
BATCH_POINTS = 50  # grid points per batched pass
DIFF_EXTENT = 1.5  # diff grid half-width in units of ||d_j|| (the steered activation h_j sits at a = 1)
MIN_DIFF_NORM = 1e-4  # relative to ||v||: positions j with a smaller ||d_j|| end the diff trajectory


# ---------------------------------------------------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------------------------------------------------

def context_pool(config, exp, data_dir):
    """Held-out items of each pool source (items the steering vectors use come first and are skipped)."""
    skip = {'wiki_en': config['n_docs'], 'harmless': config['n_prompts']}
    pool = []
    for entry in exp['pool']:
        with open(os.path.join(data_dir, f"{entry['source']}.jsonl"), encoding='utf-8') as f:
            rows = [json.loads(line)['text'].strip() for line in f]
        for i in range(skip.get(entry['source'], 0), len(rows)):
            text = rows[i]
            if entry.get('prefix_words'):
                text = ' '.join(text.split()[:entry['prefix_words']])
            pool.append({'id': f"{entry['source']}_{i:04d}", 'chat': bool(entry['chat']), 'text': text})
    assert pool, f"empty context pool for {exp}"
    return pool


def sample_contexts(config, data_dir):
    seed = config['seed'] if config['seed'] is not None else secrets.randbits(31)
    rng = np.random.default_rng(seed)
    out = {}
    for name, exp in config['experiments'].items():
        pool = context_pool(config, exp, data_dir)
        pick = sorted(rng.choice(len(pool), size=min(config['n_contexts'], len(pool)), replace=False).tolist())
        out[name] = [pool[i] for i in pick]
    return {'seed': seed, 'experiments': out}


def setup_context(model, config, ctx):
    prompt = chat_prompt(model, ctx['text']) if ctx['chat'] else ctx['text']
    ids = encode(model, prompt)
    x = block_input(model, ids[None], config['layer'])[0, -1].float()
    return ids, x


def make_jkl(model, state, config, eos, reference=None):
    return JointKL(model, state, config['m'], eos, config['tree_nodes'], config['mc_samples'], reference=reference)


# ---------------------------------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------------------------------

def run_basin(model, config, name, ctx, v, eos, out_dir):
    layer, scope, target = config['layer'], config['scope'], config['experiments'][name]['target']
    classifier = CLASSIFIER_OF_TARGET[target]
    ids, x = setup_context(model, config, ctx)
    n = ids.shape[0]
    print(f"\n=== {name} / {ctx['id']}: {n} tokens, ||x|| = {x.norm():.1f}, ||v|| = {v.norm():.1f}\n{ctx['text']!r}")
    basin = basin_check(model, ids, x, v, layer, scope, config['alpha_max'], target, classifier, eos)
    reach = [abs(p['alpha']) for p in basin['points'].values()]
    if basin['target_run_alpha'] is not None:
        reach.append(basin['target_run_alpha'][1])
    extent = EXTENT_FACTOR * max(reach + [1.0])

    # Unsteered greedy tokens y_1..y_J (cut before EOS) and one KL reference per prefix (context + y_<=j)
    state_ctx = prefix_states(model, ids, n, layer)[0][0]
    y = list(itertools.takewhile(lambda t: t not in eos, greedy_decode(model, state_ctx, config['n_decoded'], eos)[0]))
    assert y, 'the unsteered model ends right after the context'
    y_text = [model.tokenizer.decode([t]) for t in y]
    ids_full = torch.cat([ids, torch.tensor(y, device=ids.device)])
    states_p, _ = prefix_states(model, ids_full, n, layer)
    jkls = [make_jkl(model, st, config, eos) for st in states_p]
    for j, jkl in enumerate(jkls):
        print(f"[kl] j={j}: P-mass covered by the tree per step {[round(c, 3) for c in jkl.covered_mass()]}")

    # Steered points: KL and greedy labels after the context and after each y_j
    names = list(basin['points'])
    if names:
        offsets = torch.stack([basin['points'][p]['alpha'] * v for p in names])
        states_q, _ = prefix_states(model, ids_full, n, layer, offsets, scope)
        for p in names:
            basin['points'][p].update(kl=[], label_after=[])
        for jkl, st in zip(jkls, states_q):
            kl = jkl(st)['kl']
            texts = model.tokenizer.batch_decode(greedy_decode(model, st, LABEL_TOKENS, eos))
            for r, p in enumerate(names):
                basin['points'][p]['kl'].append(kl[r].tolist())
                basin['points'][p]['label_after'].append(classify(texts[r], classifier))

    diff = steered_trajectories(model, config, ids, ids_full, y, basin, v, eos)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'experiment': name, 'context': ctx, 'n_ctx': n, 'last_token': model.tokenizer.decode(ids[-1:]),
        'decoded_ids': y, 'decoded_text': y_text, 'target': target, 'classifier': classifier, 'layer': layer,
        'scope': scope, 'm': config['m'], 'n_planes': config['n_planes'], 'extent': extent,
        'coords': np.linspace(-extent, extent, config['resolution']), 'basin': basin,
        'references': [jkl.reference() for jkl in jkls], 'diff': diff,
    }, os.path.join(out_dir, 'basin.pt'))


def steered_trajectories(model, config, ids, ids_full, y, basin, v, eos):
    """
    For each steered point p in diff_points: greedy z_1..z_J from the context steered by alpha_p v, and at j = 0..J
    (J = min(|y|, |z|)) d_j = h_j - x_j, the steered minus unsteered input of block L at the last position of
    ctx + z_<=j and ctx + y_<=j. d_0 = alpha_p v when the scope covers the last context token.
    """
    layer, scope, n = config['layer'], config['scope'], ids.shape[0]
    _, acts = prefix_states(model, ids_full, n, layer, capture=True)
    x_traj = acts['src'][0, n - 1:]  # [|y| + 1, d]
    out = {}
    for p in config['diff_points']:
        if p not in basin['points']:
            print(f"[diff] {p}: no such steered point, skipped")
            continue
        alpha = basin['points'][p]['alpha']
        off = (alpha * v)[None]
        z = greedy_decode(model, context_state(model, ids, layer, off, scope), config['n_decoded'], eos)[0]
        z = list(itertools.takewhile(lambda t: t not in eos, z))
        n_meas = min(len(y), len(z)) + 1
        ids_z = torch.cat([ids, torch.tensor(z, device=ids.device, dtype=ids.dtype)])
        _, acts_q = prefix_states(model, ids_z, n, layer, off, scope, capture=True)
        d = acts_q['src'][0, n - 1:n - 1 + n_meas] - x_traj[:n_meas]
        small = (d.norm(dim=-1) < MIN_DIFF_NORM * v.norm()).nonzero()
        n_meas = int(small[0]) if len(small) else n_meas
        if n_meas == 0:
            print(f"[diff] {p}: d_0 = 0, skipped")
            continue
        d = d[:n_meas]
        cos = F.cosine_similarity(d, v[None], dim=-1)
        out[p] = {'alpha': alpha, 'steered_ids': z, 'steered_text': [model.tokenizer.decode([t]) for t in z],
                  'd': d.cpu(), 'cos': cos.tolist(), 'd_norm': (d.norm(dim=-1) / v.norm()).tolist(),
                  'extent': DIFF_EXTENT, 'coords': np.linspace(-DIFF_EXTENT, DIFF_EXTENT, config['resolution'])}
        print(f"[diff] {p}: alpha={alpha:.3f}, z={out[p]['steered_text']!r}, "
              f"cos(d_j, v)={[round(c, 3) for c in out[p]['cos']]}, ||d_j||/||v||="
              f"{[round(r, 3) for r in out[p]['d_norm']]}")
    return out


def run_plane(model, config, ctx, v, eos, out_dir, k):
    info = torch.load(os.path.join(out_dir, 'basin.pt'), weights_only=False)
    layer, scope, classifier = info['layer'], info['scope'], info['classifier']
    ids, x = setup_context(model, config, ctx)
    n, n_meas = ids.shape[0], len(info['decoded_ids']) + 1
    ids_full = torch.cat([ids, torch.tensor(info['decoded_ids'], device=ids.device)])
    pos = slice(n - 1, n - 1 + n_meas)  # positions whose output predicts the token after y_<=j
    u = orthogonal_directions(v, info['n_planes'])[k]
    axes = torch.stack([u, v])  # in-plane coordinates (b, a) in units of ||v||

    states_p, acts = prefix_states(model, ids_full, n, layer, capture=True)
    x_out = acts['final'][0, pos]  # f_j(x)
    shift_x = (acts['final'] - acts['src'])[0, pos]  # unsteered change from block L's input to the last block output
    jkls = [make_jkl(model, st, config, eos, ref) for st, ref in zip(states_p, info['references'])]
    del states_p, acts

    coords = torch.tensor(info['coords'], device=v.device, dtype=torch.float32)
    res = len(coords)
    aa, bb = (g.reshape(-1) for g in torch.meshgrid(coords, coords, indexing='ij'))  # rows over a, columns over b
    offsets_all = aa[:, None] * v + bb[:, None] * u
    n_pts = res * res
    kl = np.zeros((n_pts, n_meas, config['m']), dtype=np.float32)
    labels = np.empty((n_pts, n_meas), dtype=object)
    shift = np.zeros((n_pts, n_meas, 2), dtype=np.float32)
    h_out = torch.empty(n_pts, n_meas, x.shape[0], device=v.device)
    t0 = time.time()
    for start in range(0, n_pts, BATCH_POINTS):
        sl = slice(start, min(n_pts, start + BATCH_POINTS))
        states_q, acts = prefix_states(model, ids_full, n, layer, offsets_all[sl], scope, capture=True)
        h_out[sl] = acts['final'][:, pos]
        # In-plane residual shift: [(f(h) - f(x)) - (h - x)] = steered minus unsteered change from layer L to the end
        shift[sl] = project_onto((acts['final'] - acts['src'])[:, pos] - shift_x, axes).cpu().numpy()
        for j, st in enumerate(states_q):
            kl[sl, j] = jkls[j](st)['kl'].numpy()
            texts = model.tokenizer.batch_decode(greedy_decode(model, st, LABEL_TOKENS, eos))
            labels[sl, j] = [classify(t, classifier) for t in texts]
        del states_q, acts

    h_src = x[None] + offsets_all
    ratio = NormRatio(x, x_out, h_src, h_out)
    log_ratio = ratio(h_src, h_out).cpu().numpy()
    point_ratio = {}
    if info['basin']['points']:
        names = list(info['basin']['points'])
        off = torch.stack([info['basin']['points'][p]['alpha'] * v for p in names])
        _, acts = prefix_states(model, ids_full, n, layer, off, scope, capture=True)
        lr = ratio(x[None] + off, acts['final'][:, pos]).cpu().numpy()
        point_ratio = {p: lr[r].tolist() for r, p in enumerate(names)}
    sec = (time.time() - t0) / n_pts
    print(f"[grid] {info['experiment']} / {ctx['id']} plane {k}: {res}x{res} points, {sec:.2f} s/point")
    grid = lambda a: a.reshape(res, res, *a.shape[1:])  # noqa: E731
    torch.save({'kl': grid(kl), 'labels': grid(labels), 'log_ratio': grid(log_ratio), 'shift': grid(shift),
                'point_log_ratio': point_ratio, 'seconds_per_point': sec},
               os.path.join(out_dir, f'plane_{k:02d}.pt'))


def run_diff_plane(model, config, ctx, eos, out_dir, p, k):
    """Plane k of steered point p: at each j, offsets a d_j + b u_k at the last position of ctx + y_<=j only."""
    info = torch.load(os.path.join(out_dir, 'basin.pt'), weights_only=False)
    if p not in info['diff']:
        return
    dp, layer, classifier = info['diff'][p], info['layer'], info['classifier']
    ids, _ = setup_context(model, config, ctx)
    n, d_all = ids.shape[0], dp['d'].to(ids.device).float()
    n_meas = d_all.shape[0]
    ids_full = torch.cat([ids, torch.tensor(info['decoded_ids'], device=ids.device, dtype=ids.dtype)])
    states_p, acts = prefix_states(model, ids_full, n, layer, capture=True)
    jkls = [make_jkl(model, st, config, eos, ref) for st, ref in zip(states_p[:n_meas], info['references'])]
    x_src, x_out = acts['src'][0], acts['final'][0]
    del states_p, acts

    coords = torch.tensor(dp['coords'], device=ids.device, dtype=torch.float32)
    res = len(coords)
    aa, bb = (g.reshape(-1) for g in torch.meshgrid(coords, coords, indexing='ij'))  # rows over a, columns over b
    n_pts = res * res
    kl = np.zeros((n_pts + 1, n_meas, config['m']), dtype=np.float32)  # last row: the steered activation (1, 0)
    labels = np.empty((n_pts + 1, n_meas), dtype=object)
    shift = np.zeros((n_pts + 1, n_meas, 2), dtype=np.float32)
    log_ratio = np.zeros((n_pts + 1, n_meas), dtype=np.float32)
    t0 = time.time()
    for j in range(n_meas):
        t = n - 1 + j  # the measured position (last of ctx + y_<=j)
        d = d_all[j]
        u = orthogonal_directions(d, info['n_planes'])[k]
        axes = torch.stack([u, d])
        offsets = torch.cat([aa[:, None] * d + bb[:, None] * u, d[None]])
        shift_x = x_out[t] - x_src[t]
        h_out = torch.empty(n_pts + 1, x_src.shape[1], device=ids.device)
        for start in range(0, n_pts + 1, BATCH_POINTS):
            sl = slice(start, min(n_pts + 1, start + BATCH_POINTS))
            states_q, acts = prefix_states(model, ids_full[:t + 1], t + 1, layer, offsets[sl], 'last', capture=True)
            h_out[sl] = acts['final'][:, t]
            shift[sl, j] = project_onto((acts['final'] - acts['src'])[:, t] - shift_x, axes).cpu().numpy()
            kl[sl, j] = jkls[j](states_q[0])['kl'].numpy()
            texts = model.tokenizer.batch_decode(greedy_decode(model, states_q[0], LABEL_TOKENS, eos))
            labels[sl, j] = [classify(txt, classifier) for txt in texts]
            del states_q, acts
        h_src = x_src[t][None] + offsets
        ratio = NormRatio(x_src[t], x_out[t][None], h_src[:n_pts], h_out[:n_pts, None])
        log_ratio[:, j] = ratio(h_src, h_out[:, None])[:, 0].cpu().numpy()
    sec = (time.time() - t0) / (n_pts * n_meas)
    print(f"[diff] {info['experiment']} / {ctx['id']} {p} plane {k}: {res}x{res} points x {n_meas} positions, "
          f"{sec:.2f} s/point")
    grid = lambda a: a[:n_pts].reshape(res, res, *a.shape[1:])  # noqa: E731
    torch.save({'kl': grid(kl), 'labels': grid(labels), 'log_ratio': grid(log_ratio), 'shift': grid(shift),
                'point_kl': kl[-1], 'point_label': labels[-1], 'point_log_ratio': log_ratio[-1],
                'seconds_per_point': sec}, os.path.join(out_dir, f'diff_{p}_plane_{k:02d}.pt'))


def merge(out_dir):
    info = torch.load(os.path.join(out_dir, 'basin.pt'), weights_only=False)
    info.pop('references')
    planes = [torch.load(p, weights_only=False) for p in sorted(glob.glob(os.path.join(out_dir, 'plane_*.pt')))]
    assert len(planes) == info['n_planes'], f"{out_dir}: {len(planes)} of {info['n_planes']} planes"
    res = dict(info)
    for key in ('kl', 'log_ratio', 'shift'):  # plane means: [a, b, j, ...]
        res[key] = np.stack([pl[key] for pl in planes]).mean(0)
    res['target_fraction'] = np.mean([pl['labels'] == info['target'] for pl in planes], axis=0).astype(np.float32)
    for p, point in res['basin']['points'].items():
        point['log_ratio'] = np.mean([pl['point_log_ratio'][p] for pl in planes], axis=0).tolist()
    for p, dp in res.setdefault('diff', {}).items():
        dplanes = [torch.load(f, weights_only=False)
                   for f in sorted(glob.glob(os.path.join(out_dir, f'diff_{p}_plane_*.pt')))]
        assert len(dplanes) == info['n_planes'], f"{out_dir}: {len(dplanes)} of {info['n_planes']} diff planes of {p}"
        dp['d'] = dp['d'].numpy()
        for key in ('kl', 'log_ratio', 'shift', 'point_kl', 'point_log_ratio'):
            dp[key] = np.stack([pl[key] for pl in dplanes]).mean(0)
        dp['target_fraction'] = np.mean([pl['labels'] == info['target'] for pl in dplanes], axis=0).astype(np.float32)
        dp['point_target_fraction'] = np.mean([pl['point_label'] == info['target'] for pl in dplanes], axis=0)
    torch.save(res, os.path.join(out_dir, 'result.pt'))
    summary = {
        'experiment': res['experiment'], 'context': res['context'], 'decoded_text': res['decoded_text'],
        'barrier_deg': res['basin']['barrier_deg'], 'target_deg': res['basin']['target_deg'],
        'points': {p: {k: pt[k] for k in ('alpha', 'label', 'passed', 'kl', 'label_after', 'log_ratio')}
                   for p, pt in res['basin']['points'].items()},
        'max_kl_m_per_j': res['kl'][..., -1].reshape(-1, res['kl'].shape[2]).max(0).tolist(),
        'target_fraction_per_j': res['target_fraction'].reshape(-1, res['kl'].shape[2]).mean(0).tolist(),
        'diff': {p: {'alpha': dp['alpha'], 'steered_text': dp['steered_text'], 'cos': dp['cos'],
                     'd_norm_over_v': dp['d_norm'], 'kl_at_h_j': dp['point_kl'][:, -1].tolist(),
                     'max_kl_m_per_j': dp['kl'][..., -1].reshape(-1, dp['kl'].shape[2]).max(0).tolist(),
                     'target_fraction_per_j': dp['target_fraction'].reshape(-1, dp['kl'].shape[2]).mean(0).tolist()}
                 for p, dp in res['diff'].items()},
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    return summary


# ---------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=os.path.join(HERE, 'config.yaml'))
    parser.add_argument('--data-dir', default='data/steering_basins')
    parser.add_argument('--output-dir', default='outputs/steering_basins')
    parser.add_argument('--quick', action='store_true', help='tiny settings for an end-to-end test')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides, e.g. m=4')
    parser.add_argument('--stage', default='all', choices=['all', 'vectors', 'basin', 'grid', 'merge'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config, args.set, QUICK if args.quick else None)
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda *p: os.path.join(args.output_dir, *p)  # noqa: E731

    if args.stage in ('all', 'vectors'):
        sampled = sample_contexts(config, args.data_dir)
        config['seed'] = sampled['seed']
        with open(out('contexts.json'), 'w', encoding='utf-8') as f:
            json.dump(sampled, f, indent=2, ensure_ascii=False)
        with open(out('config_used.yaml'), 'w') as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    with open(out('contexts.json'), encoding='utf-8') as f:
        sampled = json.load(f)
    jobs = [(name, ctx) for name, ctxs in sampled['experiments'].items() for ctx in ctxs]
    ctx_dir = lambda name, ctx: out(name, ctx['id'])  # noqa: E731

    if args.stage != 'merge':
        device = get_device()
        model = load_model(config['model'], device, config['dtype'])
        assert config['layer'] + PROBE_DEPTH <= model.cfg.n_layers
        print(f"{config['model']} ({model.cfg.n_layers} blocks) on {device}")
        if args.stage in ('all', 'vectors'):
            vectors = steering_vectors(model, config['layer'], args.data_dir, config['n_docs'], config['n_prompts'])
            torch.save({k: v.cpu() for k, v in vectors.items()}, out('vectors.pt'))
            if args.stage == 'vectors':
                return
        vectors = {k: v.to(device).float() for k, v in torch.load(out('vectors.pt')).items()}
        eos = eos_ids(model)
        if args.stage in ('all', 'basin'):
            for i, (name, ctx) in enumerate(jobs):
                if i % args.num_shards == args.shard:
                    run_basin(model, config, name, ctx, vectors[name], eos, ctx_dir(name, ctx))
        if args.stage in ('all', 'grid'):
            units = [(i, p, k) for i in range(len(jobs)) for p in [None] + config['diff_points']
                     for k in range(config['n_planes'])]
            for u, (i, p, k) in enumerate(units):
                if u % args.num_shards == args.shard:
                    name, ctx = jobs[i]
                    if p is None:
                        run_plane(model, config, ctx, vectors[name], eos, ctx_dir(name, ctx), k)
                    else:
                        run_diff_plane(model, config, ctx, eos, ctx_dir(name, ctx), p, k)
    if args.stage in ('all', 'merge'):
        summaries = [merge(ctx_dir(name, ctx)) for name, ctx in jobs]
        with open(out('summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False, default=float)
        for s in summaries:
            print(f"{s['experiment']} / {s['context']['id']}: barrier {s['barrier_deg']}, target {s['target_deg']}, "
                  f"points {[(p, round(d['alpha'], 2), d['label'], d['passed']) for p, d in s['points'].items()]}")
            for p, d in s['diff'].items():
                print(f"  diff {p}: z={d['steered_text']!r}, cos(d_j, v)={[round(c, 3) for c in d['cos']]}")


if __name__ == '__main__':
    main()
