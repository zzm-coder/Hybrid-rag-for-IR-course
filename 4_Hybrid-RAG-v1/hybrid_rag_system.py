"""混合RAG系统 - 结合实体匹配、知识图谱和向量检索"""
import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from data_types import SystemConfig, RAGContext, QuestionType, KGResult, LLMResponse
from query_router import QueryRouter, RouterResponse
from kg_retriever import KnowledgeGraphRetriever
from vector_retriever import VectorRetriever
from llm_generator import LLMGenerator

logger = logging.getLogger(__name__)

class HybridRAGSystem:
    """混合RAG系统"""
    
    def __init__(self, config: SystemConfig, use_kg: bool = True, use_vector: bool = True):
        self.config = config
        self.use_kg = use_kg
        self.use_vector = use_vector
        
        # 初始化组件
        self.query_router = QueryRouter(config)
        self.kg_retriever = KnowledgeGraphRetriever(config) if self.use_kg else None
        self.vector_retriever = VectorRetriever(config) if self.use_vector else None
        self.llm_generator = LLMGenerator(config)
        
        # 创建输出目录
        output_path = Path(config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建上下文保存目录
        self.context_save_path = output_path / "contexts"
        self.context_save_path.mkdir(exist_ok=True)
        
        logger.info("混合RAG系统初始化完成")
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        处理查询的完整流程
        """
        start_time = time.time()
        
        try:
            # 1. 查询理解与路由
            router_response = self.query_router.analyze_query(question)
            logger.info(f"问题分析: 类型={router_response.question_type.value}(ID={router_response.type_id}), 实体={router_response.entities}, 意图={router_response.intent}")
            
            # 2. 混合检索
            rag_context = self._hybrid_retrieval(question, router_response)
            
            # 3. 重排序
            if rag_context.vector_chunks:
                reranked = self.vector_retriever.rerank_chunks(question, rag_context.vector_chunks)
                rag_context.reranked_chunks = reranked
            
            # 4. 生成答案
            enhanced_context = rag_context.get_enhanced_context()
            llm_response = self.llm_generator.generate_answer(question, enhanced_context)
            
            # 5. 保存RAG上下文（用于评估）
            self._save_rag_context(rag_context, router_response)
            
            # 6. 计算总时间
            total_time = time.time() - start_time
            
            # 7. 保存结果
            result = self._package_results(
                question=question,
                router_response=router_response,
                rag_context=rag_context,
                llm_response=llm_response,
                total_time=total_time
            )
            
            # 8. 保存到文件
            self._save_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            return {
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def _hybrid_retrieval(self, question: str, router_response: RouterResponse) -> RAGContext:
        """
        执行混合检索
        """
        start_time = time.time()
        
        # 初始化上下文
        rag_context = RAGContext(
            question=question,
            question_type=router_response.question_type,
            retrieval_time=0.0
        )
        
        # 根据问题类型执行不同检索策略
        if router_response.question_type == QuestionType.SIMPLE_FACT:
            # 简单事实型：KG为主，向量为辅
            logger.info("使用简单事实型检索策略")
            rag_context = self._retrieve_for_simple_fact(question, router_response, rag_context)
            
        elif router_response.question_type == QuestionType.COMPLEX_LOGIC:
            # 复杂逻辑型：KG多跳推理 + 向量语义检索
            logger.info("使用复杂逻辑型检索策略")
            rag_context = self._retrieve_for_complex_logic(question, router_response, rag_context)
            
        else:  # QuestionType.OPEN_SEMANTIC
            # 开放语义型：向量为主，KG提供锚点
            logger.info("使用开放语义型检索策略")
            rag_context = self._retrieve_for_open_semantic(question, router_response, rag_context)
        
        rag_context.retrieval_time = time.time() - start_time
        
        logger.info(f"检索完成: KG三元组={len(rag_context.kg_results.triples) if rag_context.kg_results else 0}, 向量块={len(rag_context.vector_chunks)}")
        
        return rag_context
    
    def _retrieve_for_simple_fact(self, question: str, router_response: RouterResponse, 
                                 rag_context: RAGContext) -> RAGContext:
        """简单事实型检索策略"""
        # 1. KG检索（主要）
        if self.use_kg and self.kg_retriever:
            kg_results = self.kg_retriever.query_kg(
                question=question,
                entities=router_response.entities,
                question_type=router_response.question_type
            )
            rag_context.kg_results = kg_results
        else:
            rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)
        
        # 2. 向量检索（补充）
        if self.use_vector and self.vector_retriever:
            kg_entities = rag_context.kg_results.entities if rag_context.kg_results else []
            vector_chunks = self.vector_retriever.retrieve(
                question=question,
                entities=router_response.entities,
                kg_entities=kg_entities
            )
            rag_context.vector_chunks = vector_chunks
        else:
            rag_context.vector_chunks = []
        
        return rag_context
    
    def _retrieve_for_complex_logic(self, question: str, router_response: RouterResponse,
                                   rag_context: RAGContext) -> RAGContext:
        """复杂逻辑型检索策略"""
        # 1. KG检索（多跳推理）
        if self.use_kg and self.kg_retriever:
            kg_results = self.kg_retriever.query_kg(
                question=question,
                entities=router_response.entities,
                question_type=router_response.question_type
            )
            rag_context.kg_results = kg_results
        else:
            rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)
        
        # 2. 向量检索（语义补充）
        all_entities = list(set(router_response.entities))
        if rag_context.kg_results and rag_context.kg_results.entities:
            all_entities.extend(rag_context.kg_results.entities)

        if self.use_vector and self.vector_retriever:
            vector_chunks = self.vector_retriever.retrieve(
                question=question,
                entities=router_response.entities,
                kg_entities=all_entities
            )
            rag_context.vector_chunks = vector_chunks
        else:
            rag_context.vector_chunks = []
        
        return rag_context
    
    def _retrieve_for_open_semantic(self, question: str, router_response: RouterResponse,
                                   rag_context: RAGContext) -> RAGContext:
        """开放语义型检索策略"""
        # 1. 向量检索（主要）
        # KG检索：可选
        if self.use_kg and self.kg_retriever:
            kg_results = self.kg_retriever.query_kg(
                question=question,
                entities=router_response.entities,
                question_type=router_response.question_type
            )
            rag_context.kg_results = kg_results
        else:
            rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)

        # 向量检索：文本段落检索（可选）
        if self.use_vector and self.vector_retriever:
            vector_chunks = self.vector_retriever.retrieve(
                question=question,
                entities=router_response.entities,
                kg_entities=rag_context.kg_results.entities if rag_context.kg_results else []
            )
            rag_context.vector_chunks = vector_chunks
        else:
            rag_context.vector_chunks = []
        
        # 2. KG检索（提供锚点）
        if router_response.entities and self.use_kg and self.kg_retriever:
            kg_results = self.kg_retriever.query_kg(
                question=question,
                entities=router_response.entities,
                question_type=router_response.question_type
            )
            rag_context.kg_results = kg_results
        
        return rag_context
    
    def _save_rag_context(self, rag_context: RAGContext, router_response: RouterResponse):
        """保存RAG上下文（用于评估）"""
        try:
            # 提取详细信息
            context_details = {
                "question": rag_context.question,
                "question_type": rag_context.question_type.value,
                "router_entities": router_response.entities,
                "router_intent": router_response.intent,
                "retrieval_time": rag_context.retrieval_time,
                "kg_results": None,
                "vector_results": [],
                "reranked_results": [],
                "combined_context": rag_context.get_enhanced_context()
            }
            
            # KG结果详情：按 source 分组，每个 source 包含段落列表，每段落包含相关三元组
            if rag_context.kg_results:
                from collections import defaultdict

                # 按 source -> paragraph -> 列表（三元组）分组
                grouped = defaultdict(lambda: defaultdict(list))
                for triple in rag_context.kg_results.triples:
                    src = triple.source or "unknown"
                    para = triple.paragraph or ""
                    grouped[src][para].append({
                        "head": triple.head,
                        "relation": triple.relation,
                        "tail": triple.tail,
                        "confidence": triple.confidence
                    })

                ke_results = []
                for src, para_map in grouped.items():
                    para_list = []
                    for para_text, triples in para_map.items():
                        para_list.append({
                            "text": para_text,
                            "triples": triples
                        })

                    ke_results.append({
                        "source": src,
                        "paragrepa": para_list
                    })

                context_details["kg_results"] = {
                    "ke_results": ke_results,
                    "entities": rag_context.kg_results.entities,
                    "query_time": rag_context.kg_results.query_time
                }
            
            # 向量结果详情
            for chunk in rag_context.vector_chunks:
                context_details["vector_results"].append({
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "text_preview": chunk.chunk_text,
                    "similarity_score": chunk.similarity_score
                })
            
            # 重排序结果详情
            for chunk in rag_context.reranked_chunks:
                context_details["reranked_results"].append({
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "text_preview": chunk.chunk_text,
                    "rerank_score": chunk.rerank_score
                })
            
            # # 保存到文件
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # question_hash = hash(rag_context.question) % 10000
            # filename = f"context_{timestamp}_{question_hash}.json"
            # filepath = self.context_save_path / filename
            
            # with open(filepath, 'w', encoding='utf-8') as f:
            #     json.dump(context_details, f, ensure_ascii=False, indent=2)
            
            # logger.info(f"RAG上下文已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存RAG上下文失败: {e}")
    
    def _package_results(self, question: str, router_response: RouterResponse,
                        rag_context: RAGContext, llm_response: LLMResponse,
                        total_time: float) -> Dict[str, Any]:
        """打包所有结果"""
        # 将 KG 结果转换为按 source 分组的结构（与保存上下文的格式一致）
        kg_dict = {}
        if rag_context.kg_results:
            from collections import defaultdict
            grouped = defaultdict(lambda: defaultdict(list))
            for triple in rag_context.kg_results.triples:
                src = triple.source or "unknown"
                para = triple.paragraph or ""
                grouped[src][para].append({
                    "head": triple.head,
                    "relation": triple.relation,
                    "tail": triple.tail,
                    "confidence": triple.confidence
                })

            ke_results = []
            for src, para_map in grouped.items():
                para_list = []
                for para_text, triples in para_map.items():
                    para_list.append({
                        "text": para_text,
                        "triples": triples
                    })

                ke_results.append({
                    "source": src,
                    "paragrepa": para_list
                })

            kg_dict = {
                "ke_results": ke_results,
                "entities": rag_context.kg_results.entities,
                "query_time": rag_context.kg_results.query_time
            }
        else:
            kg_dict = {
                "ke_results": [],
                "entities": [],
                "query_time": 0.0
            }
        
        # 构建候选 md 列表：来自 KG ke_results 的 source 与 重排序结果的 source
        candidate_md = []
        # 从 ke_results 提取
        try:
            for item in kg_dict.get("ke_results", []):
                src = item.get("source") if isinstance(item, dict) else None
                if src:
                    # 提取以 .md 结尾的文件名或路径最后一部分
                    m = re.search(r'([^\\/\s]+\.md)', str(src), re.IGNORECASE)
                    if m:
                        fname = m.group(1).split('/')[-1]
                        candidate_md.append(fname)
                    else:
                        # 有时 source 本身就是文件名但不包含 .md，尝试直接取字符串
                        candidate_md.append(str(src))
        except Exception:
            pass

        # 从重排序结果中提取 source
        try:
            for c in rag_context.reranked_chunks:
                src = getattr(c, 'source', None) or (c.get('source') if isinstance(c, dict) else None)
                if src:
                    m = re.search(r'([^\\/\s]+\.md)', str(src), re.IGNORECASE)
                    if m:
                        fname = m.group(1).split('/')[-1]
                        candidate_md.append(fname)
                    else:
                        candidate_md.append(str(src))
        except Exception:
            pass

        # 去重并保持顺序
        seen = set(); candidate_md_list = []
        for x in candidate_md:
            if x and x not in seen:
                seen.add(x); candidate_md_list.append(x)

        # 从 LLM 原始响应中提取【证据】区文本，作为 evidence_block 保存
        evidence_block = ""
        try:
            em = re.search(r'【证据】\s*(.*)', llm_response.raw_response or "", re.DOTALL)
            if em:
                evidence_block = em.group(1).strip()
        except Exception:
            evidence_block = ""

        # 在 evidence_block 中匹配候选 md 文件名（按候选顺序）
        matched_citations = []
        try:
            low_block = (evidence_block or "").lower()
            for md in candidate_md_list:
                if md.lower() in low_block:
                    matched_citations.append(md)
            # 退回策略：若未匹配到，但 evidence_block 中包含任意 .md 名称，则直接提取这些 md
            if not matched_citations and evidence_block:
                found = re.findall(r'([^\\/\s]+\.md)', evidence_block, re.IGNORECASE)
                # 去重并保留出现顺序
                seen2 = set()
                for f in found:
                    fn = f.split('/')[-1]
                    if fn not in seen2:
                        seen2.add(fn); matched_citations.append(fn)
        except Exception:
            matched_citations = []

        return {
            "question": question,
            "router_analysis": {
                "type_id": router_response.type_id,
                "question_type": router_response.question_type.value,
                "entities": router_response.entities,
                "intent": router_response.intent,
                "metadata": router_response.metadata
            },
            "retrieval": {
                "kg_results": kg_dict,
                "vector_results": [c.to_dict() for c in rag_context.vector_chunks],
                "reranked_results": [c.to_dict() for c in rag_context.reranked_chunks],
                "retrieval_time": rag_context.retrieval_time
            },
            "generation": {
                "answer": llm_response.answer,
                # matched_citations 基于上下文 candidate_md_list 与 LLM 依据块的匹配结果
                "citations": matched_citations,
                "citation_extracted_files": matched_citations,
                "evidence_block": evidence_block,
                "generation_time": llm_response.generation_time,
                "raw_response": llm_response.raw_response
            },
            "performance": {
                "total_time": total_time,
                "retrieval_time": rag_context.retrieval_time,
                "generation_time": llm_response.generation_time
            },
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "kg_uri": self.config.neo4j_uri,
                "vector_db": str(self.config.vector_db_path),
                "llm_model": self.config.llm_model
            }
        }
    
    def _save_result(self, result: Dict[str, Any]):
        """保存结果到文件"""
        output_path = Path(self.config.output_path)
        
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # question_hash = hash(result["question"]) % 10000
        # filename = f"rag_result_{timestamp}_{question_hash}.json"
        
        # # 保存完整结果
        # results_dir = output_path / "results"
        # results_dir.mkdir(exist_ok=True)
        # filepath = results_dir / filename
        
        # with open(filepath, 'w', encoding='utf-8') as f:
        #     json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 追加到日志文件
        log_file = output_path / "rag_log.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"结果已保存: {log_file}")
    
    def close(self):
        """关闭连接"""
        self.kg_retriever.close()