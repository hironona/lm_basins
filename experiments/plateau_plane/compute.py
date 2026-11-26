#!/usr/bin/env python3
"""
Sample last-token activations on a surface (plane or slerp-like sphere) through the activations of three sentences
at a source layer, and at every grid point compute
    - the argmax next token,
    - the Frobenius norm of the Jacobian d logits / d activation (source layer -> logits),
    - KL(P_a || P_x) between the next-token distribution at each anchor sentence a and at the point x.
Writes one .pt file for plot.py (which needs no model). Run from the project root:
    python experiments/plateau_plane/compute.py [--set layer=6 surface=flat]
"""

import argparse
import os
import re
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common.config import get_device, load_config  # noqa: E402
from common.last_token import (anchor_activations, forward_from_layer, jacobian_norm_to_logits,  # noqa: E402
                               unembed_gram_factor)
from common.lm import encode, load_model  # noqa: E402
from common.metrics import next_token_kl  # noqa: E402
from common.surfaces import build_surface, grid_coords, make_plane_grid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def output_path(config, output_dir):
    slug = re.sub(r'[^A-Za-z0-9]+', '_', '_'.join([config['prefix']] + config['words'])).strip('_')
    name = f"{slug}__layer{config['layer']}_{config['surface']}_res{config['resolution']}.pt"
    return os.path.join(output_dir, config['model'], name)


def sentence_tokens(model, prefix, words):
    """[3, T] tokens of prefix + ' ' + word."""
    prefix_ids = encode(model, prefix)
    rows = []
    for word in words:
        ids = encode(model, ' ' + word)
        assert ids.shape[0] == 1, f"' {word}' is not a single token"
        rows.append(torch.cat([prefix_ids, ids]))
    return torch.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=os.path.join(HERE, 'config.yaml'))
    parser.add_argument('--output-dir', default='outputs/plateau_plane')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    words, layer = config['words'], config['layer']
    assert len(words) == 3, 'three words span the surface'

    device = get_device()
    model = load_model(config['model'], device)
    assert 0 <= layer <= model.cfg.n_layers
    tokens = sentence_tokens(model, config['prefix'], words)
    anchors, contexts, anchor_logits = anchor_activations(model, tokens, layer)
    anchor_log_probs = torch.log_softmax(anchor_logits.double(), dim=-1)

    surface = build_surface(anchors, config['surface'])
    anchor_coords = surface.to_coords(anchors)
    axis_u, axis_v = make_plane_grid(anchor_coords.cpu(), config['resolution'], config['margin'])
    coords = grid_coords(axis_u, axis_v).to(device)
    surface.validate_grid(coords)
    points = surface.to_ambient(coords).float()
    print(f"{config['model']}, layer {layer}, {config['surface']} surface, {points.shape[0]} points on {device}")

    gram = unembed_gram_factor(model).to(device)
    labels, jac_norms, kl = [], [], []
    for start in tqdm(range(0, points.shape[0], config['batch_size']), desc='grid'):
        x = points[start:start + config['batch_size']]
        with torch.no_grad():
            logits = forward_from_layer(model, x, layer, contexts)
        labels.append(logits.argmax(-1).cpu())
        kl.append(next_token_kl(anchor_log_probs, logits).cpu())
        jac_norms.append(jacobian_norm_to_logits(model, x, layer, contexts, gram).cpu())
    labels, kl = torch.cat(labels), torch.cat(kl, dim=1)

    anchor_labels = anchor_logits.argmax(-1).cpu()
    token_ids = torch.unique(torch.cat([labels, anchor_labels])).tolist()
    data = {
        'config': config,
        'axis_u': axis_u, 'axis_v': axis_v,  # grid point (r, c) = (axis_u[c], axis_v[r]), flattened row-major
        'anchor_coords': anchor_coords.cpu().float(),  # [3, 2]
        'anchor_labels': anchor_labels,  # [3] argmax next token of each sentence
        'labels': labels,  # [N] argmax next token
        'token_strings': {i: model.to_string([i]) for i in token_ids},
        'jacobian_norm': torch.cat(jac_norms).float(),  # [N] ||d logits / d x||_F
        'kl': kl.float(),  # [3, N] KL(P_a || P_x) in nats
    }
    path = output_path(config, args.output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)
    print(f"saved {path}")


if __name__ == '__main__':
    main()
