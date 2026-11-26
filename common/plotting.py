"""Minimal matplotlib style and grid heat-map helpers shared by all experiments. Importing applies the style."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'figure.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GREY = '#c9c8c3'
TEXT = '#222222'
SEQUENTIAL = LinearSegmentedColormap.from_list('seq', ['#f2f7fd', '#9ec5f4', '#3987e5', '#184f95', '#0b2a55'])
DIVERGING = LinearSegmentedColormap.from_list('div', [CATEGORICAL[0], '#f4f4f4', CATEGORICAL[7]])


def pixel_extent(axis_u, axis_v):
    """imshow extent with pixel centers on the grid points."""
    du, dv = axis_u[1] - axis_u[0], axis_v[1] - axis_v[0]
    return (axis_u[0] - du / 2, axis_u[-1] + du / 2, axis_v[0] - dv / 2, axis_v[-1] + dv / 2)


def heatmap(ax, values, axis_u, axis_v, **kwargs):
    """values [len(v), len(u)] drawn with pixel centers on the grid."""
    kwargs.setdefault('cmap', SEQUENTIAL)
    return ax.imshow(values, origin='lower', extent=pixel_extent(axis_u, axis_v), interpolation='nearest', **kwargs)


def colorbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(label)
    cb.outline.set_visible(False)
    return cb


def draw_boundaries(ax, axis_u, axis_v, labels, color='white'):
    """Boundaries between the regions of an integer label grid [len(v), len(u)]."""
    for label in np.unique(labels):
        mask = (labels == label).astype(float)
        if not mask.all():
            cs = ax.contour(axis_u, axis_v, mask, levels=[0.5], colors=color, linewidths=0.9)
            cs.set(path_effects=[pe.Stroke(linewidth=2.0, foreground=TEXT), pe.Normal()])


def label_regions(ax, axis_u, axis_v, labels, texts: dict, min_fraction=0.02):
    """Write texts[label] at the point of each large region farthest from its boundary."""
    for label, text in texts.items():
        mask = labels == label
        if mask.mean() < min_fraction:
            continue
        dist = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        r, c = np.unravel_index(np.argmax(dist), dist.shape)
        ax.text(axis_u[c], axis_v[r], text, ha='center', va='center', fontsize=8, color=TEXT,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))


def draw_arrows(ax, axis_u, axis_v, vectors, n_per_axis=12, log_range=None, color=TEXT):
    """
    Arrows for a vector field vectors [len(v), len(u), 2] on about n_per_axis grid points per axis. Arrow length is
    affine in log10 of the vector norm over log_range (lo, hi) (default: the field's own range), so fields spanning
    several orders of magnitude stay readable. Returns log_range.
    """
    norms = np.linalg.norm(vectors, axis=-1)
    if log_range is None:
        finite = np.log10(norms[norms > 0])
        log_range = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    lo, hi = log_range
    stride = max(1, len(axis_u) // n_per_axis)
    idx = np.arange(stride // 2, len(axis_u), stride)
    vec, nrm = vectors[np.ix_(idx, idx)], norms[np.ix_(idx, idx)]
    step = stride * (axis_u[1] - axis_u[0])
    length = 0.25 + 0.65 * np.clip((np.log10(np.maximum(nrm, 1e-30)) - lo) / max(hi - lo, 1e-9), 0, 1)
    unit = np.where(nrm[..., None] > 0, vec / np.maximum(nrm[..., None], 1e-30), 0)
    uu, vv = np.meshgrid(axis_u[idx], axis_v[idx])
    ax.quiver(uu, vv, unit[..., 0] * length * step, unit[..., 1] * length * step, angles='xy', scale_units='xy',
              scale=1, color=color, width=0.004, headwidth=3.5, headlength=4)
    return log_range


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
