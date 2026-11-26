"""
Basin check for a steering vector v at the activation x entering block L at the last context token
(Finlayson et al. 2026, Secs. 2-4).

1. Circle probe: gamma(theta) = ||x|| (x_hat cos theta + u sin theta), u the unit tangent of v at x, replaces the last
   context position at block L and is pushed through blocks L .. L+PROBE_DEPTH-1. The speed ||d eta / d theta|| shows
   basins (low speed) separated by barriers.
2. The speed is smoothed (SMOOTH_DEG moving average). Barrier = first peak at theta > 0 with prominence >= PROMINENCE x
   the speed range over [0, MAX_STEER_DEG]; target theta* = the lowest speed between it and the next barrier.
3. Two steered points on the ray x + alpha v (both lie on every grid plane span(v, u_k) through x):
     speed_ray    alpha* with theta(alpha*) = theta* (the paper's adaptive steering),
     classifier   the smallest swept alpha whose greedy continuation has the target label, refined by bisection.
4. Behavior: greedy continuations along the ray (offset alpha v at the context positions of the scope), labeled by
   the experiment's classifier. A steered point 'passed' if it is past the barrier and has the target label.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from common.lm import block_input
from common.steering import context_state, greedy_decode
from labels import classify

PROBE_DEPTH = 4  # probe blocks L .. L+3 (paper: Qwen3-8B blocks 18-21)
THETA_STEP_DEG = 1
SMOOTH_DEG = 5
PROMINENCE = 0.1
MAX_STEER_DEG = 120
SWEEP_STEPS = 25
REFINE_STEPS = 4
GREEDY_TOKENS = 16
PROBE_BATCH = 64


def ray_geometry(x: torch.Tensor, v: torch.Tensor):
    """||x||, and v's components along x_hat and perpendicular to it."""
    x_hat = x / x.norm()
    v_par = float(v @ x_hat)
    return float(x.norm()), v_par, float((v - v_par * x_hat).norm())


def theta_of_alpha(x, v, alpha):
    """Angle (rad) of x + alpha v in the plane span(x_hat, u)."""
    r, v_par, v_perp = ray_geometry(x, v)
    alpha = np.asarray(alpha, dtype=float)
    return np.arctan2(alpha * v_perp, r + alpha * v_par)


def alpha_of_theta(x, v, theta: float) -> Optional[float]:
    """Inverse of theta_of_alpha for alpha >= 0 (None if the ray never reaches theta)."""
    r, v_par, v_perp = ray_geometry(x, v)
    denom = v_perp * math.cos(theta) - v_par * math.sin(theta)
    return r * math.sin(theta) / denom if denom > 0 else None


@torch.no_grad()
def speed_profile(model, ids: torch.Tensor, layer: int, x: torch.Tensor, v: torch.Tensor, thetas: np.ndarray):
    x_hat = x / x.norm()
    u = v - (v @ x_hat) * x_hat
    u = u / u.norm()
    th = torch.tensor(thetas, device=x.device, dtype=torch.float32)
    points = x.norm() * (torch.cos(th)[:, None] * x_hat + torch.sin(th)[:, None] * u)
    resid = block_input(model, ids[None], layer)  # [1, n, d]
    etas = []
    for start in range(0, len(thetas), PROBE_BATCH):
        pts = points[start:start + PROBE_BATCH]
        r = resid.expand(pts.shape[0], -1, -1).clone()
        r[:, -1] = pts.to(r.dtype)
        etas.append(model(r, start_at_layer=layer, stop_at_layer=layer + PROBE_DEPTH)[:, -1].float())
    eta = torch.cat(etas)
    return (torch.gradient(eta, spacing=float(thetas[1] - thetas[0]), dim=0)[0]).norm(dim=-1).cpu().numpy()


def find_target_basin(thetas: np.ndarray, speed: np.ndarray) -> Dict:
    """Barrier and target basin angles (rad, None if absent) and the smoothed speed."""
    width = max(1, round(SMOOTH_DEG / THETA_STEP_DEG)) | 1
    smooth = uniform_filter1d(speed, width, mode='wrap')
    max_steer = math.radians(MAX_STEER_DEG)
    window = (thetas >= 0) & (thetas <= max_steer)
    peaks, _ = find_peaks(smooth, prominence=PROMINENCE * float(np.ptp(smooth[window])))
    right = [p for p in peaks if thetas[p] > 0]
    out = {'barrier': None, 'target': None, 'smooth': smooth}
    if right and thetas[right[0]] < max_steer:
        out['barrier'] = float(thetas[right[0]])
        end = right[1] if len(right) > 1 else len(thetas) - 1
        idx = [i for i in range(right[0] + 1, end) if thetas[i] <= max_steer]
        if idx:
            out['target'] = float(thetas[idx[int(np.argmin(smooth[idx]))]])
    return out


@torch.no_grad()
def behavior(model, ids, layer, v, alphas: List[float], scope: str, eos: List[int], classifier: str) -> List[Dict]:
    """Greedy continuation and label for each steering strength alpha (one batched pass)."""
    state = context_state(model, ids, layer, torch.tensor(alphas, device=v.device)[:, None] * v[None], scope)
    texts = model.tokenizer.batch_decode(greedy_decode(model, state, GREEDY_TOKENS, eos))
    return [{'alpha': a, 'text': t, 'label': classify(t, classifier)} for a, t in zip(alphas, texts)]


def basin_check(model, ids, x, v, layer: int, scope: str, alpha_max: float, target: str, classifier: str,
                eos: List[int]) -> Dict:
    half = math.pi
    thetas = np.arange(-half, half + 1e-9, math.radians(THETA_STEP_DEG))
    speed = speed_profile(model, ids, layer, x, v, thetas)
    basin = find_target_basin(thetas, speed)
    deg = lambda r: None if r is None else float(np.degrees(r))  # noqa: E731
    barrier_deg = deg(basin['barrier'])

    def run(alphas):
        out = behavior(model, ids, layer, v, alphas, scope, eos, classifier)
        for b in out:
            b['theta_deg'] = float(np.degrees(theta_of_alpha(x, v, b['alpha'])))
            print(f"  alpha={b['alpha']:6.3f} theta={b['theta_deg']:7.2f} [{b['label']:11s}] {b['text']!r}")
        return out

    sweep = run(np.linspace(0, alpha_max, SWEEP_STEPS).round(4).tolist())
    points = {}

    def add(name, b):
        crossed = barrier_deg is not None and b['theta_deg'] > barrier_deg
        points[name] = dict(b, norm=float((x + b['alpha'] * v).norm()), crossed_barrier=crossed,
                            passed=crossed and b['label'] == target)

    if basin['target'] is not None:
        alpha_star = alpha_of_theta(x, v, basin['target'])
        if alpha_star is not None and alpha_star <= 2 * alpha_max:
            add('speed_ray', run([alpha_star])[0])

    i_flip = next((i for i, s in enumerate(sweep) if s['label'] == target), None)
    target_run = None
    if i_flip is not None:
        hi, lo = sweep[i_flip], sweep[i_flip - 1]['alpha'] if i_flip > 0 else None
        for _ in range(REFINE_STEPS if lo is not None else 0):
            mid = run([0.5 * (lo + hi['alpha'])])[0]
            if mid['label'] == target:
                hi = mid
            else:
                lo = mid['alpha']
        add('classifier', hi)
        j = i_flip
        while j + 1 < len(sweep) and sweep[j + 1]['label'] == target:
            j += 1
        target_run = [sweep[i_flip]['alpha'], sweep[j]['alpha']]

    for name, p in points.items():
        print(f"[basin] {name}: alpha={p['alpha']:.3f} theta={p['theta_deg']:.1f} label={p['label']} "
              f"passed={p['passed']}")
    return {
        'thetas_deg': np.degrees(thetas), 'speed': speed, 'speed_smooth': basin['smooth'],
        'barrier_deg': barrier_deg, 'target_deg': deg(basin['target']),
        'alpha_barrier': alpha_of_theta(x, v, basin['barrier']) if basin['barrier'] is not None else None,
        'sweep': sweep, 'points': points, 'target_run_alpha': target_run, 'probe_depth': PROBE_DEPTH,
    }
