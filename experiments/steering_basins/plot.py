#!/usr/bin/env python3
"""
Plots for every result.pt under a results directory (CPU is fine). Heatmaps are plane means over offsets
a v + b u_k (vertical a along the steering vector v, horizontal b along random u_k ⟂ v, both in units of ||v||);
rows/panels j: right after the context, then after each unsteered greedy token y_j.
    *_kl.png     log10 joint KL of the next m' tokens, and the fraction of planes with the target greedy label
    *_ratio.png  log10 norm ratio R (< 0 contraction, > 0 expansion) with arrows of the in-plane residual shift
    *_basin.png  circle-probe speed with the behavioral sweep, and KL_m along the steering axis
    *_diff_<p>.png  trajectory-difference criterion for steered point p: rows j, planes a d_j + b u_k around the
                 unsteered x_j at the last position of ctx + y_<=j (vertical a along d_j = h_j - x_j, the steered minus
                 unsteered layer-L input with the steered greedy tokens z_<=j; both axes in units of ||d_j||, h_j at
                 (0, 1)); columns: log10 KL_m', fraction with the target greedy label, log10 R with residual-shift
                 arrows. Row labels give y_j, z_j and cos(d_j, v)
    *_diff_cos.png  cos(d_j, v) and ||d_j|| / ||v|| against j for every steered point
Run: python experiments/steering_basins/plot.py outputs/steering_basins
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common.plotting import CATEGORICAL, DIVERGING, GREY, SEQUENTIAL, TEXT, colorbar, draw_arrows, save  # noqa: E402

KL_FLOOR = 1e-4
LABEL_COLORS = {'en': CATEGORICAL[0], 'zh': CATEGORICAL[1], 'other': GREY, 'non-refusal': CATEGORICAL[0],
                'refusal': CATEGORICAL[2]}
POINTS = {'speed_ray': dict(marker='P', mfc=CATEGORICAL[3], ms=8, label='α* (speed basin)'),
          'classifier': dict(marker='*', mfc='white', ms=10, label='α_cls (classifier)')}
EDGE = dict(mec=TEXT, mew=0.8, ls='none')


def report_ms(m):
    return sorted({1, max(1, m // 2), m})


def step_name(res, j):
    return 'after context' if j == 0 else f"after {res['decoded_text'][j - 1]!r}"


def title(res):
    return f"{res['experiment']} · {res['context']['id']} · layer {res['layer']} · steer {res['scope']}"


def mark(ax, res):
    ax.plot(0, 0, marker='o', mfc='white', ms=6, label='unsteered', **EDGE)
    for name, p in res['basin']['points'].items():
        ax.plot(0, p['alpha'], **POINTS[name], **EDGE)


def grid_axes(n_rows, n_cols, res, unit='‖v‖'):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols + 1.2, 2.2 * n_rows + 0.8), sharex=True,
                             sharey=True, squeeze=False, layout='constrained')
    for ax in axes[-1]:
        ax.set_xlabel(f'b (×{unit})')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'a (×{unit})')
    return fig, axes


def show(ax, values, res, **kwargs):
    e = res['extent']
    return ax.imshow(values, origin='lower', extent=[-e, e, -e, e], interpolation='nearest', **kwargs)


def finish(fig, axes, res, path):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(handles))
    fig.suptitle(title(res))
    save(fig, path)


def plot_kl(res, path):
    ms, n_meas = report_ms(res['m']), res['kl'].shape[2]
    log_kl = np.log10(np.maximum(res['kl'], KL_FLOOR))
    vmax = max(log_kl[..., [m - 1 for m in ms]].max(), np.log10(KL_FLOOR) + 1e-3)
    frac_cmap = LinearSegmentedColormap.from_list('frac', ['#f4f4f4', LABEL_COLORS[res['target']]])
    fig, axes = grid_axes(n_meas, len(ms) + 1, res)
    for j in range(n_meas):
        for c, m in enumerate(ms):
            im = show(axes[j, c], log_kl[:, :, j, m - 1], res, cmap=SEQUENTIAL, vmin=np.log10(KL_FLOOR), vmax=vmax)
        im_frac = show(axes[j, -1], res['target_fraction'][:, :, j], res, cmap=frac_cmap, vmin=0, vmax=1)
        for ax in axes[j]:
            mark(ax, res)
        axes[j, 0].set_ylabel(f"{step_name(res, j)}\na (×‖v‖)")
    for c, m in enumerate(ms):
        axes[0, c].set_title(f'KL, m = {m}')
    axes[0, -1].set_title(f"greedy = {res['target']}")
    colorbar(fig, im, axes[:, :len(ms)], 'log₁₀ KL (nats)')
    colorbar(fig, im_frac, axes[:, -1], 'fraction of planes')
    finish(fig, axes, res, path)


def plot_ratio(res, path):
    n_meas, coords = res['log_ratio'].shape[2], res['coords']
    lr = res['log_ratio']
    lim = max(float(np.nanmax(np.abs(lr))), 1e-3)
    norms = np.linalg.norm(res['shift'], axis=-1)
    log_range = (float(np.log10(norms[norms > 0].min())), float(np.log10(norms.max()))) if norms.max() > 0 \
        else (0.0, 1.0)
    fig, axes = grid_axes(1, n_meas, res)
    for j, ax in enumerate(axes[0]):
        im = show(ax, lr[:, :, j], res, cmap=DIVERGING, vmin=-lim, vmax=lim)
        draw_arrows(ax, coords, coords, res['shift'][:, :, j], log_range=log_range)
        mark(ax, res)
        ax.set_title(step_name(res, j))
    colorbar(fig, im, axes, f"log₁₀ R (layer {res['layer']} → last)")
    axes[0, 0].plot([], [], marker=r'$\rightarrow$', color=TEXT, ls='none', ms=10,
                    label=f'residual shift, 10^{log_range[0]:.1f}–10^{log_range[1]:.1f} ‖v‖')
    finish(fig, axes, res, path)


def plot_basin(res, path):
    b, m = res['basin'], res['m']
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    th, speed = np.asarray(b['thetas_deg']), np.asarray(b['speed_smooth'])
    ax0.plot(th, b['speed'], color=CATEGORICAL[0], lw=0.6, alpha=0.3)
    ax0.plot(th, speed, color=CATEGORICAL[0], lw=1.5)
    for key, ls in (('barrier_deg', '--'), ('target_deg', ':')):
        if b[key] is not None:
            ax0.axvline(b[key], color=TEXT, ls=ls, lw=1, label=key.replace('_deg', ''))
    lo, span = speed.min(), np.ptp(speed) or 1.0
    ax0.set_ylim(lo - 0.2 * span, speed.max() + 0.05 * span)
    for label, color in LABEL_COLORS.items():
        sweep = [s['theta_deg'] for s in b['sweep'] if s['label'] == label]
        if sweep:
            ax0.plot(sweep, [lo - 0.1 * span] * len(sweep), 'o', ms=4, color=color, label=f'ray: {label}')
    ax0.set_xlabel('θ toward v (deg)')
    ax0.set_ylabel('speed ‖dη/dθ‖')
    ax0.set_title(f"circle probe, blocks {res['layer']}–{res['layer'] + b['probe_depth'] - 1}")
    ax0.legend(loc='upper left')

    coords = res['coords']
    col = int(np.argmin(np.abs(coords)))  # column nearest b = 0
    n_meas = res['kl'].shape[2]
    for j in range(n_meas):
        ax1.plot(coords, res['kl'][:, col, j, m - 1], color=SEQUENTIAL(0.4 + 0.6 * j / max(n_meas - 1, 1)),
                 label=step_name(res, j))
    for name, p in b['points'].items():
        ax1.plot(p['alpha'], p['kl'][0][m - 1], **POINTS[name], **EDGE)
    ax1.set_yscale('log')
    ax1.set_xlabel('a (×‖v‖), b ≈ 0')
    ax1.set_ylabel(f'KL, m = {m} (nats)')
    ax1.set_title('joint KL along the steering axis')
    ax1.legend()
    fig.suptitle(title(res))
    save(fig, path)


def diff_row_name(res, dp, j):
    head = 'after context' if j == 0 else \
        f"y {res['decoded_text'][j - 1]!r} → z {dp['steered_text'][j - 1]!r}"
    return f"j = {j}: {head}\ncos(d_j, v) = {dp['cos'][j]:+.3f}\n‖d_j‖ = {dp['d_norm'][j]:.2f}‖v‖\n\na (×‖d_j‖)"


def plot_diff(res, p, path):
    dp = res['diff'][p]
    ms, n_meas = report_ms(res['m']), dp['kl'].shape[2]
    log_kl = np.log10(np.maximum(dp['kl'], KL_FLOOR))
    vmax = max(log_kl[..., [m - 1 for m in ms]].max(), np.log10(KL_FLOOR) + 1e-3)
    frac_cmap = LinearSegmentedColormap.from_list('frac', ['#f4f4f4', LABEL_COLORS[res['target']]])
    lr = dp['log_ratio']
    lim = max(float(np.nanmax(np.abs(lr))), 1e-3)
    norms = np.linalg.norm(dp['shift'], axis=-1)
    log_range = (float(np.log10(norms[norms > 0].min())), float(np.log10(norms.max()))) if norms.max() > 0 \
        else (0.0, 1.0)
    fig, axes = grid_axes(n_meas, len(ms) + 2, res, unit='‖d_j‖')
    for j in range(n_meas):
        for c, m in enumerate(ms):
            im = show(axes[j, c], log_kl[:, :, j, m - 1], dp, cmap=SEQUENTIAL, vmin=np.log10(KL_FLOOR), vmax=vmax)
        im_frac = show(axes[j, -2], dp['target_fraction'][:, :, j], dp, cmap=frac_cmap, vmin=0, vmax=1)
        im_lr = show(axes[j, -1], lr[:, :, j], dp, cmap=DIVERGING, vmin=-lim, vmax=lim)
        draw_arrows(axes[j, -1], dp['coords'], dp['coords'], dp['shift'][:, :, j], log_range=log_range)
        for ax in axes[j]:
            ax.plot(0, 0, marker='o', mfc='white', ms=6, label='unsteered x_j', **EDGE)
            ax.plot(0, 1, **dict(POINTS[p], **EDGE, label=f"steered h_j, {POINTS[p]['label']}"))
        axes[j, 0].set_ylabel(diff_row_name(res, dp, j), rotation=0, ha='right', va='center')
    for c, m in enumerate(ms):
        axes[0, c].set_title(f'KL, m = {m}')
    axes[0, -2].set_title(f"greedy = {res['target']}")
    axes[0, -1].set_title('log₁₀ R, residual shift')
    colorbar(fig, im, axes[:, :len(ms)], 'log₁₀ KL (nats)')
    colorbar(fig, im_frac, axes[:, -2], 'fraction of planes')
    colorbar(fig, im_lr, axes[:, -1], f"log₁₀ R (layer {res['layer']} → last)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(handles))
    fig.suptitle(f"{title(res)} · diff criterion, {p} (α = {dp['alpha']:.2f}): vertical a along d_j = h_j − x_j, "
                 f"horizontal b along u_k ⊥ d_j (×‖d_j‖)")
    save(fig, path)


def plot_diff_cos(res, path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.2), layout='constrained')
    for p, dp in res['diff'].items():
        j = np.arange(len(dp['cos']))
        color = TEXT if POINTS[p]['mfc'] == 'white' else POINTS[p]['mfc']
        style = dict(color=color, ls='--' if p == 'classifier' else '-', mfc=POINTS[p]['mfc'], marker=POINTS[p]['marker'], mec=TEXT, mew=0.8, ms=7)
        ax0.plot(j, dp['cos'], label=f"{p} (α = {dp['alpha']:.2f})", **style)
        ax1.plot(j, dp['d_norm'], **style)
    ax0.axhline(0, color=GREY, lw=1)
    ax0.set_ylim(-1.05, 1.05)
    ax0.set_ylabel('cos(d_j, v)')
    ax0.set_title('direction of the steered − unsteered difference')
    ax0.legend()
    ax1.set_ylabel('‖d_j‖ / ‖v‖')
    ax1.set_title('size of the difference')
    for ax in (ax0, ax1):
        ax.set_xlabel('j (tokens after the context)')
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle(title(res))
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    args = parser.parse_args()
    found = 0
    for root, _, files in sorted(os.walk(args.results_dir)):
        if 'result.pt' in files:
            res = torch.load(os.path.join(root, 'result.pt'), weights_only=False)
            stem = os.path.join(args.results_dir, 'plots', f"{res['experiment']}__{res['context']['id']}")
            os.makedirs(os.path.dirname(stem), exist_ok=True)
            plot_kl(res, stem + '_kl.png')
            plot_ratio(res, stem + '_ratio.png')
            plot_basin(res, stem + '_basin.png')
            for p in res.get('diff', {}):
                plot_diff(res, p, stem + f'_diff_{p}.png')
            if res.get('diff'):
                plot_diff_cos(res, stem + '_diff_cos.png')
            found += 1
    assert found, f"no result.pt under {args.results_dir}"


if __name__ == '__main__':
    main()
