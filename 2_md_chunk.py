import os
import re
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import hashlib

# ========== 配置 ==========
MD_ROOT = Path("/home/zzm/Project_1/kg-hk/0_mineru_pdf/data_md_final")  # 替换为你的 .md 文件根目录
VECTOR_DB_DIR = Path("/home/zzm/Project_1/kg-hk/2_kg_construction/kg_vector_db")
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# MODEL_NAME = "/hdd1/checkpoints/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese"
CHUNK_SIZE = 200  # 字符数
CHUNK_OVERLAP = 50

DEVICE = "cuda"  # 或 "cpu"

# ========== 初始化 ==========
print("🔄 加载语义模型...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
embedding_dim = model.get_sentence_embedding_dimension()

# ========== 工具函数 ==========

def split_text_by_fixed_size(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
    separator: str = "\n"
) -> List[str]:
    """
    按固定字符数分割文本，带有重叠部分
    
    Args:
        text: 输入文本
        chunk_size: 每个块的最大字符数
        overlap: 块之间的重叠字符数
        separator: 用于查找自然边界的分隔符
    
    Returns:
        分割后的文本块列表
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        if end >= text_length:
            # 最后一块
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        
        # 尝试在自然边界处断开（句子、段落等）
        # 先找句号、问号、感叹号
        boundary_chars = ['.', '。', '!', '！', '?', '？', '\n', ';', '；', ' ', '　']
        
        # 从end位置往前找最近的分隔符
        boundary_found = False
        for i in range(end, max(start + chunk_size // 2, start), -1):
            if i < len(text) and text[i] in boundary_chars:
                end = i + 1  # 包含分隔符
                boundary_found = True
                break
        
        # 如果没找到分隔符，就强制在单词边界处断开
        if not boundary_found:
            # 找空格或标点
            for i in range(end, start + chunk_size // 2, -1):
                if i < len(text) and text[i] in [' ', ',', '，', '、']:
                    end = i
                    boundary_found = True
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 移动起始位置，考虑重叠
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks

def chunk_markdown_fixed_size(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50
) -> List[Dict[str, str]]:
    """
    按固定字符数分割Markdown文本
    
    Returns:
        List[Dict[str, str]]: 每个块的信息
    """
    if not text:
        return []
    
    # 清理文本：移除多余的空格和换行
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 移除多余的空行
    text = text.strip()
    
    # 分割文本
    chunks = split_text_by_fixed_size(text, chunk_size, overlap)
    
    # 构建结果
    result = []
    for i, chunk_text in enumerate(chunks):
        # 为每个块添加上下文信息
        context_parts = []
        
        # 添加上一块的部分内容作为上下文
        if i > 0 and len(chunks[i-1]) > 50:
            prev_context = chunks[i-1][-50:]
            context_parts.append(f"[前文] {prev_context}")
        
        context_parts.append(chunk_text)
        
        # 添加下一块的部分内容作为上下文
        if i < len(chunks) - 1 and len(chunks[i+1]) > 50:
            next_context = chunks[i+1][:50]
            context_parts.append(f"[后文] {next_context}")
        
        # 完整上下文（用于向量化）
        full_context = "\n".join(context_parts)
        
        result.append({
            "text": chunk_text,  # 原始块文本
            "full_context": full_context,  # 带上下文的完整文本
            "chunk_index": i,
            "total_chunks": len(chunks)
        })
    
    return result

def extract_md_metadata(text: str, file_path: Path) -> Dict:
    """
    从Markdown文本中提取元数据
    
    Args:
        text: Markdown文本
        file_path: 文件路径
    
    Returns:
        元数据字典
    """
    metadata = {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "file_hash": hashlib.md5(text.encode()).hexdigest()[:16],
        "total_chars": len(text),
        "lines": len(text.splitlines()),
        "extracted_title": "",
        "extracted_headings": []
    }
    
    # 提取标题
    title_patterns = [
        (r'^#\s+(.+)$', 1),  # 一级标题
        (r'^title:\s*(.+)$', 1),  # YAML front matter title
        (r'^# (.+)$', 1),  # 另一种一级标题格式
    ]
    
    for pattern, group_idx in title_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            metadata["extracted_title"] = match.group(group_idx).strip()
            break
    
    # 提取所有标题
    heading_pattern = r'^(#{1,6})\s+(.+)$'
    headings = re.findall(heading_pattern, text, re.MULTILINE)
    metadata["extracted_headings"] = [f"{'#' * len(h[0])} {h[1].strip()}" for h in headings[:10]]  # 最多取前10个
    
    # 提取文档开头部分作为摘要
    first_lines = text.split('\n')[:10]
    summary = ' '.join([line.strip() for line in first_lines if line.strip()])[:200]
    metadata["summary"] = summary + "..." if len(summary) >= 200 else summary
    
    return metadata

def load_and_chunk_md_files(md_root: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    加载并分块所有MD文件
    
    Returns:
        Tuple[List[Dict], List[Dict]]: (块列表, 文件元数据列表)
    """
    chunks_with_meta = []
    file_metadata_list = []
    
    # 查找所有MD文件
    md_files = list(md_root.rglob("*.md"))
    print(f"📂 找到 {len(md_files)} 个 .md 文件，开始分块...")
    
    for md_file in tqdm(md_files, desc="分块处理"):
        try:
            # 读取文件
            text = md_file.read_text(encoding='utf-8', errors='ignore')
            
            # 提取文件元数据
            file_metadata = extract_md_metadata(text, md_file)
            file_metadata_list.append(file_metadata)
            
            # 分块
            chunked = chunk_markdown_fixed_size(
                text,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )
            
            # 为每个块添加元数据
            for i, item in enumerate(chunked):
                if not item["text"].strip():
                    continue
                    
                chunk_metadata = {
                    "file_name": md_file.name,
                    "file_path": str(md_file.resolve()),
                    "file_hash": file_metadata["file_hash"],
                    "chunk_index": i,
                    "total_chunks": len(chunked),
                    "chunk_size": len(item["text"]),
                    "context_size": len(item["full_context"]),
                    "title": file_metadata["extracted_title"],
                    "summary": file_metadata["summary"][:100]  # 只保留前100字符
                }
                
                chunks_with_meta.append({
                    "text": item["full_context"],  # 使用带上下文的文本进行向量化
                    "original_text": item["text"],  # 原始块文本
                    "metadata": chunk_metadata
                })
                
        except Exception as e:
            print(f"⚠️ 处理文件 {md_file.name} 失败: {e}")
            continue
    
    return chunks_with_meta, file_metadata_list

def build_vector_db(chunks_with_meta: List[Dict], output_dir: Path):
    """
    构建向量数据库
    
    Args:
        chunks_with_meta: 带元数据的文本块列表
        output_dir: 输出目录
    """
    if not chunks_with_meta:
        print("❌ 没有可用的文本块，跳过向量数据库构建")
        return
    
    # 准备文本
    texts = [item["text"] for item in chunks_with_meta]
    print(f"🧠 正在对 {len(texts)} 个文本块进行向量化...")
    
    try:
        # 批量编码
        embeddings = model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化以便使用内积
        )
        
        print(f"✅ 向量化完成，维度: {embeddings.shape}")
        
        # 保存向量
        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"💾 向量已保存: {embeddings_path}")
        
        # 保存元数据
        metadata_path = output_dir / "metadata.jsonl"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for item in chunks_with_meta:
                # 只保存metadata部分，减小文件大小
                f.write(json.dumps(item["metadata"], ensure_ascii=False) + "\n")
        print(f"💾 元数据已保存: {metadata_path}")
        
        # 保存完整的块信息（包含原始文本）
        chunks_path = output_dir / "chunks.json"
        chunks_data = []
        for item in chunks_with_meta:
            chunks_data.append({
                "metadata": item["metadata"],
                "original_text": item["original_text"],
                "context_text": item["text"]
            })
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"💾 完整块数据已保存: {chunks_path}")
        
        # 构建FAISS索引
        print("🔧 构建FAISS索引...")
        index = faiss.IndexFlatIP(embedding_dim)  # 内积索引（向量已归一化）
        index.add(embeddings.astype(np.float32))
        
        # 保存索引
        faiss_path = output_dir / "faiss.index"
        faiss.write_index(index, str(faiss_path))
        print(f"💾 FAISS索引已保存: {faiss_path}")
        
        # 保存配置信息
        config = {
            "model_name": MODEL_NAME,
            "embedding_dim": embedding_dim,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "total_chunks": len(chunks_with_meta),
            "total_files": len(set([item["metadata"]["file_hash"] for item in chunks_with_meta])),
            "build_time": np.datetime64('now').astype(str)
        }
        
        config_path = output_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"💾 配置已保存: {config_path}")
        
        print(f"✅ 向量数据库构建完成！")
        print(f"   总块数: {len(chunks_with_meta)}")
        print(f"   向量维度: {embedding_dim}")
        print(f"   输出目录: {output_dir}")
        
    except Exception as e:
        print(f"❌ 构建向量数据库失败: {e}")
        raise

def build_statistics(chunks_with_meta: List[Dict], file_metadata_list: List[Dict], output_dir: Path):
    """
    构建统计信息
    
    Args:
        chunks_with_meta: 块列表
        file_metadata_list: 文件元数据列表
        output_dir: 输出目录
    """
    if not chunks_with_meta:
        return
    
    # 基本统计
    total_chunks = len(chunks_with_meta)
    total_files = len(file_metadata_list)
    
    # 块大小统计
    chunk_sizes = [len(item["original_text"]) for item in chunks_with_meta]
    avg_chunk_size = np.mean(chunk_sizes) if chunk_sizes else 0
    max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
    min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
    
    # 文件大小统计
    file_chars = [meta["total_chars"] for meta in file_metadata_list]
    avg_file_size = np.mean(file_chars) if file_chars else 0
    
    # 每个文件的块数统计
    file_chunk_counts = {}
    for item in chunks_with_meta:
        file_hash = item["metadata"]["file_hash"]
        file_chunk_counts[file_hash] = file_chunk_counts.get(file_hash, 0) + 1
    
    avg_chunks_per_file = np.mean(list(file_chunk_counts.values())) if file_chunk_counts else 0
    
    # 构建统计信息
    stats = {
        "total_files": total_files,
        "total_chunks": total_chunks,
        "chunk_size_stats": {
            "average": float(avg_chunk_size),
            "maximum": int(max_chunk_size),
            "minimum": int(min_chunk_size),
            "target_size": CHUNK_SIZE
        },
        "file_size_stats": {
            "average_chars": float(avg_file_size),
            "total_files": total_files
        },
        "chunk_distribution": {
            "average_per_file": float(avg_chunks_per_file),
            "files_with_chunks": len(file_chunk_counts)
        },
        "processing_summary": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "model_used": MODEL_NAME
        }
    }
    
    # 保存统计信息
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"📊 统计信息:")
    print(f"   总文件数: {total_files}")
    print(f"   总块数: {total_chunks}")
    print(f"   平均块大小: {avg_chunk_size:.1f} 字符")
    print(f"   最大块大小: {max_chunk_size} 字符")
    print(f"   最小块大小: {min_chunk_size} 字符")
    print(f"   平均每个文件块数: {avg_chunks_per_file:.1f}")

class VectorDatabase:
    """向量数据库查询类"""
    
    def __init__(self, db_dir: Path):
        """初始化向量数据库"""
        self.db_dir = db_dir
        
        # 加载索引
        index_path = db_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS索引文件不存在: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # 加载元数据
        metadata_path = db_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        self.metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line.strip()))
        
        # 加载配置
        config_path = db_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索相似文本
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            相似文本列表
        """
        # 编码查询文本
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        # 搜索
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # 构建结果
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                metadata = self.metadata[idx]
                results.append({
                    "rank": i + 1,
                    "score": float(dist),
                    "metadata": metadata,
                    "chunk_index": metadata.get("chunk_index", 0),
                    "file_name": metadata.get("file_name", ""),
                    "title": metadata.get("title", "")
                })
        
        return results

# ========== 主程序 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("Markdown文本向量数据库构建工具")
    print("=" * 60)
    print(f"输入目录: {MD_ROOT}")
    print(f"输出目录: {VECTOR_DB_DIR}")
    print(f"块大小: {CHUNK_SIZE} 字符")
    print(f"重叠大小: {CHUNK_OVERLAP} 字符")
    print(f"模型: {MODEL_NAME}")
    print("-" * 60)
    
    # 1. 加载并分块
    print("\n📄 步骤1: 加载并分块Markdown文件...")
    chunks_with_meta, file_metadata_list = load_and_chunk_md_files(MD_ROOT)
    
    if not chunks_with_meta:
        print("❌ 未找到有效文本块，退出。")
        exit(1)
    
    print(f"✅ 共生成 {len(chunks_with_meta)} 个文本块。")
    
    # 2. 生成统计信息
    print("\n📊 步骤2: 生成统计信息...")
    build_statistics(chunks_with_meta, file_metadata_list, VECTOR_DB_DIR)
    
    # 3. 构建向量数据库
    print("\n🔧 步骤3: 构建向量数据库...")
    build_vector_db(chunks_with_meta, VECTOR_DB_DIR)
    
    # 4. 测试查询
    print("\n🔍 步骤4: 测试向量数据库查询...")
    try:
        db = VectorDatabase(VECTOR_DB_DIR)
        
        # 测试查询
        test_queries = [
            "空气动力学标准",
            "材料规范",
            "测试方法"
        ]
        
        print("\n🧪 测试查询结果:")
        for query in test_queries:
            print(f"\n查询: '{query}'")
            results = db.search(query, top_k=3)
            
            for result in results:
                print(f"  得分: {result['score']:.4f} - 文件: {result['file_name']}")
                if result['title']:
                    print(f"      标题: {result['title']}")
    
    except Exception as e:
        print(f"⚠️ 测试查询失败: {e}")
    
    print("\n🎉 向量数据库构建完毕！")
    print(f"📁 输出目录: {VECTOR_DB_DIR}")
    print(f"📄 主要文件:")
    print(f"  - faiss.index: FAISS索引文件")
    print(f"  - embeddings.npy: 向量数据")
    print(f"  - metadata.jsonl: 元数据")
    print(f"  - chunks.json: 完整块数据")
    print(f"  - config.json: 配置信息")
    print(f"  - statistics.json: 统计信息")