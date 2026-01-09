"""训练检索器并在测试集上评估

流程：
1) 使用 `train_retriever.py` 微调 SentenceTransformer（保存到 models/sbert_finetuned）
2) 使用微调后的模型更新 SystemConfig.semantic_model_path
3) 在 test 集上运行 `RAGEvaluator` 并保存评估结果

注意：本脚本集中实现训练+评估的调用，训练密集且受硬件影响。"""
import argparse
import logging
from pathlib import Path
import subprocess
import sys
import time

from config import SystemConfig
from hybrid_rag_system import HybridRAGSystem
from evaluator import RAGEvaluator
from config import save_config_to_json
from data_types import SystemConfig as SysCfgClass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def call_train_retriever(train_dir: str, model_name: str, out_dir: str, epochs: int, batch_size: int, max_pairs: int = None):
    cmd = [sys.executable, '/home/zzm/Project_1/kg-hk/4_RAG_method/Mix-RAG-v1/train_retriever.py', '--train-path', train_dir, '--model-name', model_name, '--output-dir', out_dir, '--epochs', str(epochs), '--batch-size', str(batch_size)]
    if max_pairs:
        cmd += ['--max-pairs', str(max_pairs)]
    logger.info('运行命令: ' + ' '.join(cmd))
    subprocess.check_call(cmd)


def call_train_reranker(train_dir: str, cross_model_name: str, out_dir: str, epochs: int, batch_size: int, max_pairs: int = None):
    cmd = [sys.executable, '/home/zzm/Project_1/kg-hk/4_RAG_method/Mix-RAG-v1/train_reranker.py', '--train-path', train_dir, '--model-name', cross_model_name, '--output-dir', out_dir, '--epochs', str(epochs), '--batch-size', str(batch_size)]
    if max_pairs:
        cmd += ['--max-pairs', str(max_pairs)]
    logger.info('运行命令: ' + ' '.join(cmd))
    subprocess.check_call(cmd)


def run(train_dir: str, val_dir: str, test_dir: str, base_model: str, out_model_dir: str, epochs: int, batch_size: int, sample_size: int):
    # 1) 训练检索器
    start = time.time()
    call_train_retriever(train_dir, base_model, out_model_dir, epochs, batch_size)
    logger.info(f"检索器训练完成，用时 {time.time() - start:.1f}s")
    
    # 1.5) 尝试训练 reranker（若有训练数据）
    try:
        cross_out = str(Path(out_model_dir).parent / 'crossencoder_finetuned')
        call_train_reranker(train_dir, '/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese', cross_out, epochs, batch_size)
        logger.info(f"reranker 已训练并保存到 {cross_out}")
    except Exception as e:
        logger.warning(f"训练 reranker 失败或跳过: {e}")

    # 2) 更新配置并初始化系统（使用微调后的语义模型）
    cfg = SystemConfig()
    cfg.semantic_model_path = str(Path(out_model_dir).absolute())
    # 若 reranker 成功训练，则设置 cross_encoder_model 字段
    cross_dir = str(Path(out_model_dir).parent / 'crossencoder_finetuned')
    if Path(cross_dir).exists():
        cfg.cross_encoder_model = cross_dir
    logger.info(f"使用微调语义模型: {cfg.semantic_model_path}")

    rag = HybridRAGSystem(cfg)

    # 3) 在 test 集上评估
    evaluator = RAGEvaluator(test_dir)
    result = evaluator.evaluate_system(rag, sample_size)
    logger.info("评估完成。结果已保存到 results 目录。")
    
    # 可选：将更新后的配置写回到 JSON（如果工作目录下存在 config.json 或用户指定路径）
    try:
        cfg_path = Path('/home/zzm/Project_1/kg-hk/4_RAG_method/Mix-RAG-v1/system_config_trained.json')
        save_config_to_json(cfg, str(cfg_path))
        logger.info(f"训练后配置已保存到 {cfg_path}")
    except Exception as e:
        logger.warning(f"保存训练后配置失败: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='/home/zzm/Project_1/kg-hk/3_QA_creation/3_QA_data/split_datasets/train')
    parser.add_argument('--val-dir', default='/home/zzm/Project_1/kg-hk/3_QA_creation/3_QA_data/split_datasets/dev')
    parser.add_argument('--test-dir', default='/home/zzm/Project_1/kg-hk/3_QA_creation/3_QA_data/split_datasets/test')
    parser.add_argument('--base-model', default='/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese')
    parser.add_argument('--out-model-dir', default='/hdd1/checkpoints/sentence-transformers/zzm_sbert_finetuned')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--sample-size', type=int, default=50)
    parser.add_argument('--save-config', default='', help='可选：将训练后配置保存到指定 JSON 路径')
    args = parser.parse_args()

    # 如果用户指定了保存路径，覆盖默认保存逻辑
    run(args.train_dir, args.val_dir, args.test_dir, args.base_model, args.out_model_dir, args.epochs, args.batch_size, args.sample_size)
