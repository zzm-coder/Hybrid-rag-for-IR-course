"""向量检索器 - 使用块向量库进行检索"""
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from data_types import SystemConfig, RetrievedChunk
from entity_matcher import EntityMatcher

logger = logging.getLogger(__name__)

class VectorRetriever:
    """增强版向量检索器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.vector_db_path = Path(config.vector_db_path)
        
        # 加载语义模型
        logger.info(f"加载语义模型: {config.semantic_model_path}")
        self.embedding_model = SentenceTransformer(config.semantic_model_path)
        
        # 加载交叉编码器
        logger.info(f"加载交叉编码器: {config.cross_encoder_model}")
        self.cross_encoder = CrossEncoder(config.cross_encoder_model)
        
        # 初始化实体匹配器
        self.entity_matcher = EntityMatcher(config)
        
        # 加载FAISS索引
        self.index = self._load_faiss_index()
        
        # 加载文档块
        self.chunks = self._load_chunks()
        
        # 加载向量嵌入数组（用于精确相似度计算）
        self.embeddings = self._load_embeddings()
        self._compute_embeddings_norms()
        
        # 建立索引到块的映射
        self.chunk_mapping = self._build_chunk_mapping()
        
        logger.info(f"向量检索器初始化完成，块数量: {len(self.chunks)}，FAISS索引大小: {self.index.ntotal}")
    
    def _load_faiss_index(self) -> faiss.Index:
        """加载FAISS索引"""
        index_path = self.vector_db_path / "faiss.index"
        if not index_path.exists():
            # 尝试查找其他可能的索引文件
            index_files = list(self.vector_db_path.glob("*.index"))
            if index_files:
                index_path = index_files[0]
                logger.info(f"使用找到的索引文件: {index_path}")
            else:
                raise FileNotFoundError(f"FAISS索引不存在: {index_path}")
        
        try:
            index = faiss.read_index(str(index_path))
            logger.info(f"FAISS索引加载成功，维度: {index.d}, 文档数: {index.ntotal}")
            return index
        except Exception as e:
            logger.error(f"加载FAISS索引失败: {e}")
            raise
    
    def _load_chunks(self) -> Dict[str, Dict]:
        """加载文档块"""
        chunks_path = self.vector_db_path / "chunks.json"
        if not chunks_path.exists():
            # 尝试查找其他可能的chunk文件
            chunk_files = list(self.vector_db_path.glob("*chunk*.json"))
            if chunk_files:
                chunks_path = chunk_files[0]
                logger.info(f"使用找到的块文件: {chunks_path}")
            else:
                logger.error(f"块文件不存在: {chunks_path}")
                return {}
        
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # 检查数据格式，可能是列表或字典
            if isinstance(chunks_data, list):
                # 如果是列表，保持原始列表顺序并同时创建按序号的 dict
                logger.info(f"chunks.json 是列表格式，包含 {len(chunks_data)} 个元素，保持顺序并创建映射")
                # 保存原始列表，便于将 FAISS 索引（基于 embeddings 行序）映射到块
                self.chunks_list = chunks_data
                chunks_dict = {}
                for i, chunk in enumerate(chunks_data):
                    chunks_dict[str(i)] = chunk
                return chunks_dict
            elif isinstance(chunks_data, dict):
                logger.info(f"加载文档块成功，数量: {len(chunks_data)}")
                return chunks_data
            else:
                logger.error(f"chunks.json 格式未知: {type(chunks_data)}")
                return {}
                
        except Exception as e:
            logger.error(f"加载文档块失败: {e}")
            return {}
    
    def _build_chunk_mapping(self) -> Dict[int, Dict]:
        """建立索引到块的映射"""
        mapping = {}
        
        # 检查 chunks 是否是字典
        if not isinstance(self.chunks, dict):
            logger.error(f"chunks 不是字典格式: {type(self.chunks)}")
            return mapping
        
        # 如果存在 chunks_list（表示 chunks.json 为列表），则直接按列表索引建立映射
        if hasattr(self, 'chunks_list') and isinstance(self.chunks_list, list):
            for i, chunk in enumerate(self.chunks_list):
                mapping[i] = chunk
            logger.info(f"使用 chunks_list 建立按序映射，映射数量: {len(mapping)}")
            return mapping

        for chunk_id_str, chunk_data in self.chunks.items():
            try:
                # 尝试将 chunk_id 转换为整数
                idx = int(chunk_id_str)
                mapping[idx] = chunk_data
            except ValueError:
                # 如果转换失败，尝试其他格式
                try:
                    # 尝试提取数字部分
                    import re
                    match = re.search(r'\d+', chunk_id_str)
                    if match:
                        idx = int(match.group())
                        mapping[idx] = chunk_data
                except:
                    logger.warning(f"无法解析 chunk_id: {chunk_id_str}")
                    continue
        
        logger.info(f"块映射建立完成，映射数量: {len(mapping)}")
        return mapping
    
    def retrieve(self, question: str, entities: List[str], kg_entities: List[str] = None) -> List[RetrievedChunk]:
        """
        向量检索
        
        Args:
            question: 问题文本
            entities: 提取的实体列表
            kg_entities: 知识图谱实体列表
            
        Returns:
            检索到的文档块列表
        """
        start_time = time.time()
        
        try:
            # 检查索引是否有效
            if self.index.ntotal == 0:
                logger.warning("FAISS索引为空，无法进行检索")
                return []
            
            # 1. 提取关键词
            keywords = self.entity_matcher.extract_keywords_from_question(question)
            
            # 2. 构建增强查询
            enhanced_query = self._build_enhanced_query(question, keywords, entities, kg_entities)
            logger.info(f"原始查询: '{question[:50]}...'")
            logger.info(f"增强查询: '{enhanced_query[:100]}...'")
            
            # 3. 计算查询嵌入
            query_embedding = self.embedding_model.encode([enhanced_query])[0]
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # 4. FAISS搜索（先返回候选索引），随后使用原始 embeddings 计算余弦相似度
            top_k = min(self.config.top_k_vector * 3, self.index.ntotal)
            if top_k == 0:
                logger.warning("top_k为0，无法进行搜索")
                return []

            _, indices = self.index.search(query_embedding, top_k)

            # 5. 获取文档块并使用 embeddings 计算余弦相似度
            retrieved_chunks = []
            q_vec = np.asarray(query_embedding).reshape(-1)
            q_norm = np.linalg.norm(q_vec)

            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue

                if idx in self.chunk_mapping:
                    chunk_data = self.chunk_mapping[idx]

                    # 通过 embeddings.npy 计算余弦相似度（更可靠）
                    try:
                        if self.embeddings is None or idx >= self.embeddings.shape[0]:
                            similarity = 0.0
                        else:
                            emb_vec = self.embeddings[idx]
                            emb_norm = self.embeddings_norms[idx] if hasattr(self, 'embeddings_norms') else np.linalg.norm(emb_vec)
                            if q_norm == 0 or emb_norm == 0:
                                similarity = 0.0
                            else:
                                similarity = float(np.dot(q_vec, emb_vec) / (q_norm * emb_norm))
                    except Exception:
                        similarity = 0.0

                    # 实体匹配增强
                    entity_boost = self._calculate_entity_boost(chunk_data, keywords, entities, kg_entities)
                    enhanced_similarity = similarity * (1.0 + entity_boost)

                    # 创建检索块
                    retrieved_chunk = RetrievedChunk(
                        chunk_id=str(idx),
                        source=chunk_data.get("metadata", {}).get("file_name", f"chunk_{idx}"),
                        chunk_text=chunk_data.get("original_text", ""),
                        metadata=chunk_data.get("metadata", {}),
                        similarity_score=enhanced_similarity,
                        retrieval_source="vector"
                    )

                    retrieved_chunks.append(retrieved_chunk)
                else:
                    logger.warning(f"索引 {idx} 不在 chunk_mapping 中")
            
            # 6. 按相似度排序
            retrieved_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # 7. 过滤低质量结果
            filtered_chunks = [
                chunk for chunk in retrieved_chunks[:self.config.top_k_vector]
                if chunk.similarity_score >= self.config.similarity_threshold
            ]
            
            logger.info(f"向量检索完成: 找到 {len(filtered_chunks)} 个相关块，最佳相似度: {filtered_chunks[0].similarity_score if filtered_chunks else 0:.3f}")
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def _build_enhanced_query(self, question: str, keywords: List[str], 
                             entities: List[str], kg_entities: List[str] = None) -> str:
        """构建增强查询"""
        parts = [question]
        
        # 添加关键词
        if keywords:
            parts.extend(keywords[:5])
        
        # 添加实体
        if entities:
            parts.extend(entities[:5])
        
        # 添加KG实体
        if kg_entities:
            parts.extend(kg_entities[:5])
        
        return " ".join(parts)
    
    def _calculate_entity_boost(self, chunk_data: Dict, keywords: List[str], 
                               entities: List[str], kg_entities: List[str] = None) -> float:
        """计算实体匹配增强分数"""
        boost = 0.0
        
        if not chunk_data:
            return boost
            
        chunk_text = chunk_data.get("original_text", "").lower()
        metadata = chunk_data.get("metadata", {})
        metadata_text = json.dumps(metadata).lower()
        
        # 检查关键词匹配
        for keyword in keywords[:3]:
            if keyword and keyword.lower() in chunk_text:
                boost += 0.1
            if keyword and keyword.lower() in metadata_text:
                boost += 0.05
        
        # 检查实体匹配
        for entity in entities[:3]:
            if entity and entity.lower() in chunk_text:
                boost += 0.15
        
        # 检查KG实体匹配
        if kg_entities:
            for kg_entity in kg_entities[:3]:
                if kg_entity and kg_entity.lower() in chunk_text:
                    boost += 0.2
        
        return min(boost, 0.5)  # 最大增强50%
    
    def rerank_chunks(self, question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """使用交叉编码器重排序"""
        if not chunks:
            return []
        
        try:
            # 准备交叉编码器输入
            pairs = [[question, chunk.chunk_text[:500]] for chunk in chunks]
            
            # 计算相关性分数
            scores = self.cross_encoder.predict(pairs)
            
            # 更新块的分数
            for i, chunk in enumerate(chunks):
                chunk.rerank_score = float(scores[i])
            
            # 按重排序分数排序
            reranked = sorted(chunks, key=lambda x: x.rerank_score, reverse=True)
            
            logger.info(f"重排序完成: 最佳重排序分数: {reranked[0].rerank_score if reranked else 0:.3f}")
            
            return reranked[:self.config.top_k_rerank]
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return chunks[:self.config.top_k_rerank]

    def _load_embeddings(self):
        """加载 embeddings.npy（如果存在）"""
        emb_path = self.vector_db_path / "embeddings.npy"
        if not emb_path.exists():
            logger.warning(f"embeddings.npy 不存在: {emb_path}")
            return None

        try:
            emb = np.load(str(emb_path))
            if emb.dtype != np.float32:
                emb = emb.astype('float32')
            logger.info(f"加载 embeddings.npy 成功，数量: {emb.shape[0]}, 维度: {emb.shape[1] if emb.ndim>1 else 'N/A'}")
            return emb
        except Exception as e:
            logger.error(f"加载 embeddings.npy 失败: {e}")
            return None

    def _compute_embeddings_norms(self):
        """预计算 embeddings 的范数以加速余弦相似度计算"""
        if getattr(self, 'embeddings', None) is None:
            self.embeddings_norms = None
            return

        try:
            self.embeddings_norms = np.linalg.norm(self.embeddings, axis=1)
        except Exception:
            # 退回到逐向量计算
            self.embeddings_norms = None