"""微调 CrossEncoder（重排序器）的脚本

使用训练集中的 (question, positive_paragraph) 对，并从语料中采样负例，构建带标签的训练样本。
正例标签为 1.0，负例为 0.0。
"""
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List

from sentence_transformers import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_pairs(train_path: str, max_pairs: int = None):
    pairs = []
    p = Path(train_path)

    def extract_from_item(item):
        q = item.get("question") or item.get("query") or ""
        sfs = item.get("supporting_facts") or item.get("support_facts") or []
        for sf in sfs:
            if isinstance(sf, str):
                para = sf
            else:
                para = (
                    sf.get("paragraph") or sf.get("text") or sf.get("context") or sf.get("para")
                    or sf.get("chunk") or sf.get("chunk_text") or sf.get("paragraph_text")
                )
            if para and q:
                pairs.append((q, para))

    if p.is_dir():
        for file in p.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for it in data:
                            extract_from_item(it)
                    elif isinstance(data, dict):
                        extract_from_item(data)
            except Exception as e:
                logger.warning(f"忽略文件 {file}: {e}")
    elif p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for it in data:
                    extract_from_item(it)
            elif isinstance(data, dict):
                extract_from_item(data)

    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def train(train_path: str, model_name: str, output_dir: str, epochs: int = 1, batch_size: int = 16, max_pairs: int = None, neg_per_pos: int = 3):
    pairs = load_training_pairs(train_path, max_pairs=max_pairs)
    if not pairs:
        raise ValueError(f"在 {train_path} 找不到训练对 (question, paragraph)。请检查训练集格式。")

    logger.info(f"加载到 {len(pairs)} 个训练对")

    # 准备负样本池（所有段落）
    all_paras = [p for (_, p) in pairs]

    examples = []
    import random
    for q, pos in pairs:
        # 正例
        examples.append(InputExample(texts=[q, pos], label=1.0))
        # 负例
        negs = random.sample(all_paras, min(len(all_paras), neg_per_pos))
        for n in negs:
            if n == pos:
                continue
            examples.append(InputExample(texts=[q, n], label=0.0))

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    model = CrossEncoder(model_name)
    model.fit(train_dataloader=train_dataloader, epochs=epochs, show_progress_bar=True)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    logger.info(f"CrossEncoder 模型已保存到 {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--model-name', default='/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese')
    parser.add_argument('--output-dir', default='/hdd1/checkpoints/sentence-transformers/zzm_cross_finetuned')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-pairs', type=int, default=None)
    parser.add_argument('--neg-per-pos', type=int, default=3)
    args = parser.parse_args()
    train(args.train_path, args.model_name, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, max_pairs=args.max_pairs, neg_per_pos=args.neg_per_pos)
