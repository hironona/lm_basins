#!/usr/bin/env python3
"""
Plots for a compute.py result (no model needed):
    *_jacobian.png  log10 ||d logits / d x||_F with the argmax-token boundaries
    *_argmax.png    argmax next token (most frequent tokens; the rest grey)
    *_kl.png        log10 KL(P_a || P_x) for each anchor sentence a
Run: python experiments/plateau_plane/plot.py outputs/plateau_plane/gpt2/<file>.pt
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common.plotting import (CATEGORICAL, GREY, TEXT, colorbar, draw_boundaries, heatmap,  # noqa: E402
                             label_regions, save)

N_TOKENS = 7  # most frequent argmax tokens shown by name
KL_FLOOR = 1e-6  # nats; KL is clipped here before log10 (0 at the anchor itself)


def draw_anchors(ax, data, ring=None):
    coords, words = data['anchor_coords'].numpy(), data['config']['words']
    ax.scatter(coords[:, 0], coords[:, 1], s=30, c=TEXT, edgecolors='white', linewidths=1, zorder=5)
    for (u, v), word in zip(coords, words):
        ax.annotate(word, (u, v), xytext=(4, 4), textcoords='offset points', fontsize=8, color=TEXT, zorder=6)
    if ring is not None:
        ax.scatter(*coords[ring], s=160, facecolors='none', edgecolors=CATEGORICAL[1], linewidths=1.5, zorder=6)


def style(ax, data):
    unit = '' if data['config']['surface'] == 'flat' else ' (rad)'
    ax.set_aspect('equal')
    ax.set_xlabel(f"e₁ → {data['config']['words'][1]}{unit}")
    ax.set_ylabel(f"e₂{unit}")


def title(data):
    c = data['config']
    return f"{c['model']} · layer {c['layer']} · “{c['prefix']} ___” · {c['surface']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', help='.pt file written by compute.py')
    args = parser.parse_args()
    data = torch.load(args.result, weights_only=True)
    stem = os.path.splitext(args.result)[0]
    u, v = data['axis_u'].numpy(), data['axis_v'].numpy()
    res = len(u)
    labels = data['labels'].numpy().reshape(res, res)
    ids, counts = np.unique(labels, return_counts=True)
    shown = ids[np.argsort(-counts)][:N_TOKENS].tolist()
    names = {i: repr(data['token_strings'][i]) for i in shown}

    # Jacobian norm
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = heatmap(ax, np.log10(data['jacobian_norm'].numpy().reshape(res, res)), u, v)
    draw_boundaries(ax, u, v, labels)
    label_regions(ax, u, v, labels, names)
    draw_anchors(ax, data)
    style(ax, data)
    colorbar(fig, im, ax, 'log₁₀ ‖J‖_F (to logits)')
    ax.set_title(title(data))
    save(fig, f"{stem}_jacobian.png")

    # Argmax map
    classes = np.full(labels.shape, len(shown))
    for k, label in enumerate(shown):
        classes[labels == label] = k
    colors = CATEGORICAL[:len(shown)] + [GREY]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    heatmap(ax, classes, u, v, cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors) - 0.5)
    draw_anchors(ax, data)
    style(ax, data)
    handles = [Patch(color=colors[k], label=names[label]) for k, label in enumerate(shown)]
    if (classes == len(shown)).any():
        handles.append(Patch(color=GREY, label='other'))
    ax.legend(handles=handles, title='argmax token', loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.set_title(title(data))
    save(fig, f"{stem}_argmax.png")

    # KL from each anchor
    kl = np.log10(np.maximum(data['kl'].numpy(), KL_FLOOR)).reshape(3, res, res)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, layout='constrained')
    for a, ax in enumerate(axes):
        im = heatmap(ax, kl[a], u, v, vmin=kl.min(), vmax=kl.max())
        draw_boundaries(ax, u, v, labels)
        draw_anchors(ax, data, ring=a)
        style(ax, data)
        if a:
            ax.set_ylabel('')
        ax.set_title(f"reference: {data['config']['words'][a]}")
    colorbar(fig, im, axes, 'log₁₀ KL(P_a ‖ P_x) (nats)')
    fig.suptitle(title(data))
    save(fig, f"{stem}_kl.png")


if __name__ == '__main__':
    main()
