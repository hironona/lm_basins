"""
Steering vectors at the residual stream entering block L (Finlayson et al. 2026):
    en_zh   : mean zh Wikipedia token activation - mean en one (per-document means over all tokens but the first,
              which is an attention sink with an outlier norm; then the mean over documents)
    refusal : mean activation at the last chat-prompt token, harmful - harmless instructions (Arditi et al. 2024)
"""

import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from common.lm import block_input, chat_prompt

BATCH = 16
MAX_TOKENS = 128


def read_texts(data_dir: str, source: str, n: int) -> List[str]:
    with open(os.path.join(data_dir, f'{source}.jsonl'), encoding='utf-8') as f:
        texts = [json.loads(line)['text'] for line in f][:n]
    assert len(texts) == n, f"{source}.jsonl has {len(texts)} texts, need {n}"
    return texts


@torch.no_grad()
def mean_activation(model, texts: List[str], layer: int, last_only: bool) -> torch.Tensor:
    """Mean over texts of the activation at the last token (last_only) or of the per-text mean over tokens 1..."""
    out = []
    for start in tqdm(range(0, len(texts), BATCH), desc='activations', leave=False):
        enc = model.tokenizer(texts[start:start + BATCH], add_special_tokens=False, truncation=True,
                              max_length=MAX_TOKENS, padding=True, padding_side='right', return_tensors='pt')
        ids, mask = enc['input_ids'].to(model.cfg.device), enc['attention_mask'].to(model.cfg.device).float()
        hidden = block_input(model, ids, layer).float()  # right padding: real tokens never see the pads
        if last_only:
            out.append(hidden[torch.arange(ids.shape[0]), mask.sum(1).long() - 1])
        else:
            mask[:, 0] = 0
            out.append((hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1))
    return torch.cat(out).mean(0)


def steering_vectors(model, layer: int, data_dir: str, n_docs: int, n_prompts: int) -> Dict[str, torch.Tensor]:
    wiki = {lang: read_texts(data_dir, f'wiki_{lang}', n_docs) for lang in ('en', 'zh')}
    prompts = {name: [chat_prompt(model, t) for t in read_texts(data_dir, name, n_prompts)]
               for name in ('harmful', 'harmless')}
    vectors = {
        'en_zh': mean_activation(model, wiki['zh'], layer, False) - mean_activation(model, wiki['en'], layer, False),
        'refusal': (mean_activation(model, prompts['harmful'], layer, True)
                    - mean_activation(model, prompts['harmless'], layer, True)),
    }
    for name, v in vectors.items():
        print(f"[vectors] {name}: norm {v.norm():.2f} at the input of block {layer}")
    return vectors
