"""系统配置和数据类型定义"""
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union

@dataclass
class SystemConfig:
    """系统配置"""
    # Neo4j配置
    neo4j_uri: str = "URL"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "PASSWORD"
    
    # 向量库配置
    vector_db_path: str = "/home/zzm/Project_1/kg-hk/2_kg_construction/kg_vector_db"
    
    # LLM配置
    llm_service_url: str = "URL"
    llm_api_key: str = "EMPTY"
    llm_model: str = "MODEL_NAME"
    
    # 语义模型配置
    # semantic_model_path: str = "/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese"
    # cross_encoder_model: str = "/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese"
    semantic_model_path: str = "/hdd1/checkpoints/sentence-transformers/zzm_sbert_finetuned"
    cross_encoder_model: str = "/hdd1/checkpoints/sentence-transformers/crossencoder_finetuned"
    
    # 缓存配置
    kg_cache_path: str = "/home/zzm/Project_1/kg-hk/2_kg_construction/kg_cache"
    
    # 输出配置
    output_path: str = "/home/zzm/Project_1/kg-hk/4_RAG_method/results"
    
    # 检索参数
    top_k_kg: int = 10
    top_k_vector: int = 15
    top_k_rerank: int = 8
    similarity_threshold: float = 0.6
    
    # 路由参数
    simple_max_length: int = 30
    simple_keywords: List[str] = None
    
    def __post_init__(self):
        if self.simple_keywords is None:
            self.simple_keywords = ["是什么", "是多少", "什么时间", "谁", "哪里", "何时", "定义", "标准"]

class QuestionType(Enum):
    """问题类型枚举"""
    SIMPLE_FACT = "simple_fact"
    COMPLEX_LOGIC = "complex_logic"
    OPEN_SEMANTIC = "open_semantic"
    
    @classmethod
    def from_int(cls, value: int) -> 'QuestionType':
        mapping = {
            1: cls.SIMPLE_FACT,
            2: cls.COMPLEX_LOGIC,
            3: cls.OPEN_SEMANTIC
        }
        return mapping.get(value, cls.SIMPLE_FACT)

@dataclass
class KGTriple:
    """知识图谱三元组"""
    head: str
    relation: str
    tail: str
    source: str
    paragraph: str = ""
    confidence: float = 1.0
    
    def to_dict(self):
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "source": self.source,
            "paragraph": self.paragraph,
            "confidence": self.confidence
        }

@dataclass
class KGResult:
    """知识图谱查询结果"""
    triples: List[KGTriple]
    entities: List[str]
    query_time: float = 0.0
    
    def to_natural_language(self) -> str:
        if not self.triples:
            return ""
        
        descriptions = []
        for triple in self.triples[:10]:
            if triple.paragraph:
                descriptions.append(f"{triple.head} {triple.relation} {triple.tail}（来源：{triple.source}，段落：{triple.paragraph[:100]}...）")
            else:
                descriptions.append(f"{triple.head} {triple.relation} {triple.tail}（来源：{triple.source}）")
        
        return "\n".join(descriptions)
    
    def to_dict(self):
        return {
            "triples": [t.to_dict() for t in self.triples],
            "entities": self.entities,
            "query_time": self.query_time
        }

@dataclass
class RetrievedChunk:
    """检索到的文档块"""
    chunk_id: str
    source: str
    chunk_text: str
    metadata: Dict[str, Any]
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_source: str = ""
    
    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "chunk_text": self.chunk_text,
            "similarity_score": round(self.similarity_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "retrieval_source": self.retrieval_source
        }

@dataclass
class RAGContext:
    """RAG上下文"""
    question: str
    question_type: QuestionType
    kg_results: Optional[KGResult] = None
    vector_chunks: List[RetrievedChunk] = field(default_factory=list)
    reranked_chunks: List[RetrievedChunk] = field(default_factory=list)
    retrieval_time: float = 0.0
    
    def get_enhanced_context(self) -> str:
        context_parts = []
        # 1) KG 信息：按 source -> paragraph 分组，列出每个三元组（完整段落）
        if self.kg_results and self.kg_results.triples:
            from collections import defaultdict
            context_parts.append("【知识图谱信息】")
            grouped = defaultdict(lambda: defaultdict(list))
            for t in self.kg_results.triples:
                src = t.source or "unknown"
                para = t.paragraph or ""
                grouped[src][para].append(t)

            # 为来源添加序号 [1], [2], ...
            for idx, (src, para_map) in enumerate(grouped.items(), start=1):
                context_parts.append(f"[{idx}] 来源: {src}")
                for para_text, triples in para_map.items():
                    if para_text:
                        context_parts.append(f"段落: {para_text}")
                    for tri in triples:
                        context_parts.append(f"- {tri.head} | {tri.relation} | {tri.tail} (confidence={tri.confidence})")
                    context_parts.append("")
        
        # 2) 文档段落：先使用重排序结果（若有），否则使用向量检索结果
        docs = self.reranked_chunks if self.reranked_chunks else self.vector_chunks
        if docs:
            context_parts.append("\n【相关文档段落】")
            # 相关文档段落的序号应接着 KG 来源的序号继续编号
            kg_count = 0
            if self.kg_results and self.kg_results.triples:
                try:
                    kg_count = len(set([t.source or "unknown" for t in self.kg_results.triples]))
                except Exception:
                    kg_count = 0

            for i, chunk in enumerate(docs, start=kg_count + 1):
                context_parts.append(f"[{i}] 来源: {chunk.source}")
                # 包含完整段落文本，便于大模型获取充分上下文
                context_parts.append(f"内容: {chunk.chunk_text}")
                # 添加元数据引用（如存在）
                try:
                    meta_info = chunk.metadata or {}
                    if meta_info:
                        meta_str = ", ".join(f"{k}:{v}" for k, v in list(meta_info.items())[:6])
                        context_parts.append(f"元信息: {meta_str}")
                except Exception:
                    pass
                context_parts.append("---")

        return "\n".join(context_parts) if context_parts else "无相关上下文信息。"

@dataclass 
class LLMResponse:
    """LLM响应"""
    answer: str
    evidence_citations: List[str]
    raw_response: str
    generation_time: float

@dataclass
class EvaluationResult:
    """评估结果"""
    # 检索指标
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    # 段落级检索指标
    para_hit_at_1: float = 0.0
    para_hit_at_5: float = 0.0
    para_hit_at_10: float = 0.0
    para_recall_at_1: float = 0.0
    para_recall_at_5: float = 0.0
    para_recall_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    
    # 上下文质量
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevance: float = 0.0
    
    # 生成质量
    em_score: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    
    # 性能
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    
    # 可信度
    citation_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    
    def to_dict(self):
        return asdict(self)