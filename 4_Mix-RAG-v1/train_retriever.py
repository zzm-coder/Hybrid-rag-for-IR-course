"""微调 SentenceTransformer 用于向量检索的简单脚本

使用训练集中的 (question, supporting_paragraph) 对进行微调，采用
MultipleNegativesRankingLoss。

输入：训练数据目录（包含若干 JSON 文件或一个 JSON list 文件），输出：保存的模型目录。
"""
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_pairs(train_path: str, max_pairs: int = None):
    """从目录或文件中加载 (question, paragraph) 对。"""
    p = Path(train_path)
    pairs = []

    def extract_from_item(item):
        q = item.get("question") or item.get("query") or ""
        sfs = item.get("supporting_facts") or item.get("support_facts") or []
        # supporting_facts 通常包含 source 与 paragraph/text（兼容多种字段名）
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


def train(train_path: str, model_name: str, output_dir: str, epochs: int = 1, batch_size: int = 16, max_pairs: int = None):
    pairs = load_training_pairs(train_path, max_pairs=max_pairs)
    if not pairs:
        raise ValueError(f"在 {train_path} 找不到训练对 (question, paragraph)。请检查训练集格式。")

    logger.info(f"加载到 {len(pairs)} 个训练对")

    # 构建 SentenceTransformer
    model = SentenceTransformer(model_name)

    # 构建 InputExample 列表：每个 example 的 texts 为 [question, paragraph]
    train_examples = [InputExample(texts=[q, p]) for q, p in pairs]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 训练
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=max(100, int(len(train_examples) * 0.1)),
        show_progress_bar=True
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    logger.info(f"模型已保存到 {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', required=True, help='训练集目录或JSON文件路径')
    parser.add_argument('--model-name', default='/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese', help='预训练基模型')
    parser.add_argument('--output-dir', default='/hdd1/checkpoints/sentence-transformers/zzm_sbert_finetuned', help='保存微调后模型的目录')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-pairs', type=int, default=None)
    args = parser.parse_args()

    train(args.train_path, args.model_name, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, max_pairs=args.max_pairs)
