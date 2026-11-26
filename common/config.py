"""Config loading (YAML + `--set key=value` overrides) and device selection."""

import copy
from typing import Dict, List, Optional

import torch
import yaml


def deep_update(base: Dict, override: Optional[Dict]) -> Dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str, overrides: Optional[List[str]] = None, extra: Optional[Dict] = None) -> Dict:
    """
    Load a YAML config, merge `extra` (e.g. quick-test settings) into it, then apply `overrides`, a list of
    'a.b=value' strings whose values are parsed as YAML.
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = deep_update(yaml.safe_load(f), extra)
    for item in overrides or []:
        key, val = item.split('=', 1)
        *path_keys, last = key.split('.')
        node = config
        for k in path_keys:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(val)
    return config


def get_device() -> str:
    """cuda if available, else cpu (TransformerLens is not reliable on mps)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
