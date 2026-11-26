#!/usr/bin/env python3
"""
Download the text needed for the steering vectors (needs internet; run once):
    wiki_en.jsonl, wiki_zh.jsonl : first --n-docs Wikipedia articles (wikimedia/wikipedia 20231101.{en,zh}),
                                   streamed, truncated to --max-chars characters, skipping very short articles
    harmful.jsonl                : mlabonne/harmful_behaviors (AdvBench-style harmful instructions)
    harmless.jsonl               : mlabonne/harmless_alpaca (Alpaca instructions)
Each line is {"text": ...}. Usage:
    python experiments/steering_basins/prepare_data.py --data-dir data/steering_basins
"""

import argparse
import json
import os

from datasets import load_dataset


def write_jsonl(path, texts):
    with open(path, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')
    print(f"wrote {len(texts)} texts to {path}")


def wikipedia(lang, n_docs, max_chars, min_chars):
    ds = load_dataset('wikimedia/wikipedia', f'20231101.{lang}', split='train', streaming=True)
    texts = []
    for row in ds:
        text = row['text'].strip()
        if len(text) >= min_chars:
            texts.append(text[:max_chars])
        if len(texts) >= n_docs:
            break
    return texts


def instructions(name, n):
    ds = load_dataset(name, split='train')
    return [row['text'].strip() for row in ds.select(range(min(n, len(ds))))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/steering_basins")
    parser.add_argument('--n-docs', type=int, default=1024)
    parser.add_argument('--n-prompts', type=int, default=416)
    parser.add_argument('--max-chars', type=int, default=2000)
    parser.add_argument('--min-chars', type=int, default=500)
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    for lang in ('en', 'zh'):
        write_jsonl(os.path.join(args.data_dir, f'wiki_{lang}.jsonl'),
                    wikipedia(lang, args.n_docs, args.max_chars, args.min_chars))
    write_jsonl(os.path.join(args.data_dir, 'harmful.jsonl'), instructions('mlabonne/harmful_behaviors', args.n_prompts))
    write_jsonl(os.path.join(args.data_dir, 'harmless.jsonl'), instructions('mlabonne/harmless_alpaca', args.n_prompts))


if __name__ == '__main__':
    main()
