"""RAG评估器"""
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any
import re

from config import SystemConfig
from hybrid_rag_system import HybridRAGSystem
from data_types import EvaluationResult
from collections import defaultdict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """RAG评估器"""
    
    def __init__(self, qa_dataset_path: str):
        self.qa_dataset = self._load_qa_dataset(qa_dataset_path)
        self.evaluation_results = []
    
    def _load_qa_dataset(self, path: str) -> Dict[str, Any]:
        """加载QA数据集"""
        dataset_path = Path(path)
        
        if dataset_path.is_dir():
            dataset = {}
            for file in dataset_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            dataset[item["id"]] = item
                    else:
                        dataset[file.stem] = data
            return dataset
        elif dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {item["id"]: item for item in data}
                else:
                    return data
        else:
            raise FileNotFoundError(f"QA数据集不存在: {path}")
    
    def evaluate_system(self, rag_system: HybridRAGSystem, 
                       sample_size: int = 50) -> EvaluationResult:
        """评估RAG系统"""
        logger.info(f"开始评估RAG系统，样本大小: {sample_size}")
        # 保留对 rag_system 的引用以便调用 LLM 评估接口
        self.rag_system = rag_system
        
        all_items = list(self.qa_dataset.values())
        if len(all_items) > sample_size:
            import random
            sample_items = random.sample(all_items, sample_size)
        else:
            sample_items = all_items
        
        all_metrics = {
            "retrieval_metrics": [],
            "generation_metrics": [],
            "performance_metrics": []
        }
        per_item_details = []
        
        # 用于路由类型混淆矩阵统计
        type_confusion = defaultdict(lambda: defaultdict(int))

        for i, qa_item in enumerate(sample_items):
            logger.info(f"评估进度: {i+1}/{len(sample_items)}")
            
            try:
                question = qa_item["question"]
                ground_truth = qa_item["answer"]
                supporting_facts = qa_item.get("supporting_facts", [])
                
                result = rag_system.process_query(question)

                # 记录路由类型混淆统计（若数据集提供 ground truth type）
                gt_type = qa_item.get("type") or qa_item.get("question_type")
                pred_type = result.get("router_analysis", {}).get("question_type")
                if gt_type and pred_type:
                    type_confusion[gt_type][pred_type] += 1
                
                metrics = self._calculate_metrics(
                    question=question,
                    ground_truth=ground_truth,
                    supporting_facts=supporting_facts,
                    system_result=result
                )
                
                all_metrics["retrieval_metrics"].append(metrics["retrieval"])
                all_metrics["generation_metrics"].append(metrics["generation"])
                all_metrics["performance_metrics"].append(metrics["performance"])

                # 收集每项的引用提取与匹配详情，便于在结果中展示
                extracted = result.get("generation", {}).get("citation_extracted_files") or result.get("generation", {}).get("citations", [])
                gt_docs = [sf["source"] for sf in supporting_facts]
                matched = [d for d in gt_docs if d in extracted]
                citation_acc = (len(matched) / len(gt_docs)) if gt_docs else 0.0

                per_item_details.append({
                    "id": qa_item.get("id"),
                    "gt_type": gt_type,
                    "pred_type": pred_type,
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": result.get("generation", {}).get("answer", ""),
                    "extracted_citations": extracted,
                    "supporting_facts_docs": gt_docs,
                    "matched_docs": matched,
                    "citation_accuracy": citation_acc,
                    "metrics": metrics
                })
                
            except Exception as e:
                logger.error(f"评估项目失败: {e}")
                continue
        
        final_result = self._aggregate_metrics(all_metrics)
        # 将混淆矩阵附加到保存的结果中，但仍然返回 EvaluationResult 对象
        confusion_dict = {g: dict(d) for g, d in type_confusion.items()}
        self._save_evaluation_results(final_result, sample_size, confusion_matrix=confusion_dict, per_item_details=per_item_details)
        return final_result
    
    def _calculate_metrics(self, question: str, ground_truth: str, 
                          supporting_facts: List[Dict], 
                          system_result: Dict) -> Dict[str, Dict]:
        """计算单个查询的指标"""
        
        metrics = {
            "retrieval": {},
            "generation": {},
            "performance": {}
        }
        
        # 检索指标
        retrieval_data = system_result.get("retrieval", {})
        # 优先使用重排序结果进行评估（更接近最终传入LLM的上下文）
        reranked = retrieval_data.get("reranked_results", [])
        vector_results = reranked if reranked else retrieval_data.get("vector_results", [])
        
        # 支持事实来源与段落（尝试多种可能的字段名）
        def _get_gt_para(sf: Dict) -> str:
            for k in ("chunk", "paragraph", "para", "text", "paragraph_text"):
                v = sf.get(k)
                if v:
                    return v
            # 有时 supporting_fact 可能包含嵌套结构
            if isinstance(sf.get("context"), str):
                return sf.get("context")
            return ""

        ground_truth_docs = [sf.get("source") or sf.get("doc") or sf.get("document") for sf in supporting_facts]
        ground_truth_paras = [_get_gt_para(sf) for sf in supporting_facts]

        retrieved_docs = [
            (r.get("source") if isinstance(r, dict) else getattr(r, "source", None))
            for r in vector_results
        ]
        # 取出检索段落文本用于段落级命中判断，尝试多种字段
        def _get_chunk_text(r: Any) -> str:
            if not r:
                return ""
            if isinstance(r, dict):
                for k in ("text_preview", "chunk_text", "chunk", "text", "content", "snippet"):
                    v = r.get(k)
                    if v:
                        return v
                return ""
            else:
                return getattr(r, "chunk_text", getattr(r, "chunk", getattr(r, "text_preview", ""))) or ""

        retrieved_paras = [_get_chunk_text(r) for r in vector_results]
        
        # 文档级 hit@k
        metrics["retrieval"]["hit_at_1"] = self._calculate_hit_at_k(retrieved_docs, ground_truth_docs, 1)
        metrics["retrieval"]["hit_at_3"] = self._calculate_hit_at_k(retrieved_docs, ground_truth_docs, 3)
        metrics["retrieval"]["hit_at_5"] = self._calculate_hit_at_k(retrieved_docs, ground_truth_docs, 5)

        # 段落级 hit@k & recall@k（将 retrieved_paras 与 ground_truth_paras 进行匹配）
        metrics["retrieval"]["para_hit_at_1"] = self._calculate_para_hit_at_k(retrieved_paras, ground_truth_paras, 1)
        metrics["retrieval"]["para_hit_at_5"] = self._calculate_para_hit_at_k(retrieved_paras, ground_truth_paras, 5)
        metrics["retrieval"]["para_hit_at_10"] = self._calculate_para_hit_at_k(retrieved_paras, ground_truth_paras, 10)
        
        metrics["retrieval"]["recall_at_1"] = self._calculate_recall_at_k(retrieved_docs, ground_truth_docs, 1)
        metrics["retrieval"]["recall_at_3"] = self._calculate_recall_at_k(retrieved_docs, ground_truth_docs, 3)
        metrics["retrieval"]["recall_at_5"] = self._calculate_recall_at_k(retrieved_docs, ground_truth_docs, 5)

        # 段落级 recall@k
        metrics["retrieval"]["para_recall_at_1"] = self._calculate_para_recall_at_k(retrieved_paras, ground_truth_paras, 1)
        metrics["retrieval"]["para_recall_at_5"] = self._calculate_para_recall_at_k(retrieved_paras, ground_truth_paras, 5)
        metrics["retrieval"]["para_recall_at_10"] = self._calculate_para_recall_at_k(retrieved_paras, ground_truth_paras, 10)
        
        metrics["retrieval"]["mrr"] = self._calculate_mrr(retrieved_docs, ground_truth_docs)
        # 上下文质量
        metrics["retrieval"]["context_precision"] = self._calculate_context_precision(retrieved_docs, ground_truth_docs)
        # 使用语义匹配计算 context_recall：比较检索到的段落与 ground-truth 段落的语义相似度
        metrics["retrieval"]["context_recall"] = self._calculate_context_recall(retrieved_paras, ground_truth_paras)
        metrics["retrieval"]["context_relevance"] = self._calculate_context_relevance(retrieved_paras, question)
        
        # 生成指标
        generated_answer = system_result.get("generation", {}).get("answer", "")
        
        metrics["generation"]["em_score"] = 1.0 if ground_truth.strip() == generated_answer.strip() else 0.0
        # 准确率：优先使用大模型来评估 answer vs ground_truth 的准确率（0-1），若失败使用简易语义相似度回退
        accuracy_val = None
        try:
            if hasattr(self, 'rag_system') and getattr(self.rag_system, 'llm_generator', None):
                acc = self.rag_system.llm_generator.evaluate_accuracy(question, generated_answer, ground_truth)
                if isinstance(acc, float) and acc >= 0.0:
                    accuracy_val = acc
        except Exception as e:
            logger.warning(f"LLM 评估 accuracy 失败，回退到语义相似度: {e}")

        if accuracy_val is None or accuracy_val < 0.0:
            accuracy_val = self._semantic_similarity(generated_answer, ground_truth)

        metrics["generation"]["accuracy"] = float(max(0.0, min(1.0, accuracy_val)))
        # 从检索端获取 context_relevance（段落级相关性），用于作为 recall
        context_rel_val = metrics["retrieval"].get("context_relevance", 0.0)
        # 使用 accuracy 和 context_relevance 计算 F1
        metrics["generation"]["f1_score"] = self._calculate_f1_score(generated_answer, ground_truth, accuracy=accuracy_val, context_relevance=context_rel_val)
        # 忠实度：语义相似度 between generated answer and retrieved context (top3)
        metrics["generation"]["faithfulness"] = self._calculate_semantic_faithfulness(generated_answer, vector_results)
        metrics["generation"]["answer_relevance"] = self._semantic_similarity(generated_answer, question)
        
        citations = system_result.get("generation", {}).get("citations", [])
        metrics["generation"]["citation_accuracy"] = self._calculate_citation_accuracy(citations, ground_truth_docs)
        metrics["generation"]["hallucination_rate"] = self._calculate_hallucination_rate(generated_answer, system_result)
        
        # 性能指标
        perf_data = system_result.get("performance", {})
        metrics["performance"]["retrieval_time"] = perf_data.get("retrieval_time", 0.0)
        metrics["performance"]["generation_time"] = perf_data.get("generation_time", 0.0)
        metrics["performance"]["total_time"] = perf_data.get("total_time", 0.0)
        
        return metrics
    
    def _calculate_hit_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        if not retrieved:
            return 0.0
        
        top_k = retrieved[:k]
        for doc in top_k:
            if doc in relevant:
                return 1.0
        return 0.0

    def _calculate_para_hit_at_k(self, retrieved_paras: List[str], relevant_paras: List[str], k: int) -> float:
        if not retrieved_paras:
            return 0.0

        top_k = retrieved_paras[:k]
        for para in top_k:
            for gt in relevant_paras:
                if self._is_paragraph_match(para, gt):
                    return 1.0
        return 0.0

    def _calculate_para_recall_at_k(self, retrieved_paras: List[str], relevant_paras: List[str], k: int) -> float:
        if not relevant_paras:
            return 0.0

        top_k = retrieved_paras[:k]
        found = 0
        for gt in relevant_paras:
            for para in top_k:
                if self._is_paragraph_match(para, gt):
                    found += 1
                    break
        return found / len(relevant_paras)

    def _is_paragraph_match(self, a: str, b: str, threshold: float = 0.5) -> bool:
        """判断两个段落是否匹配：使用词集合重叠比例或序列相似度"""
        if not a or not b:
            return False
        # 抽取中文与英文/数字 token
        a_tokens = set(re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', a))
        b_tokens = set(re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', b))
        if not a_tokens or not b_tokens:
            return False

        overlap = a_tokens.intersection(b_tokens)
        # 使用重叠占较短文本比例作为匹配依据，降低阈值以提升召回
        min_len = min(len(a_tokens), len(b_tokens)) if min(len(a_tokens), len(b_tokens)) > 0 else 1
        overlap_ratio = len(overlap) / min_len
        if overlap_ratio >= 0.25:
            return True

        # 备选：Jaccard 比例
        jaccard = len(overlap) / max(len(a_tokens.union(b_tokens)), 1)
        if jaccard >= max(threshold * 0.6, 0.15):
            return True

        # 最后退回到序列相似度（对较短文本要求更低阈值）
        ratio = SequenceMatcher(None, a, b).ratio()
        return ratio >= 0.55
    
    def _calculate_recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        if not relevant:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_found = sum(1 for doc in top_k if doc in relevant)
        return relevant_found / len(relevant)
    
    def _calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_context_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        if not retrieved:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(retrieved)
    
    def _calculate_context_recall(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        计算上下文召回（0-1）：
        - 优先使用语义匹配：当输入为段落文本时，针对每个 ground-truth 段落，计算其与所有检索段落的最大语义相似度，最终取这些最大相似度的平均值。
        - 回退为集合匹配：当输入看起来像文档 id 列表时，计算覆盖的不同相关文档比例（使用集合交集，确保结果在 0-1 之间）。
        """
        if not relevant:
            return 0.0

        # 如果检索项为空，返回 0
        if not retrieved:
            return 0.0

        # 判断输入是否更像是段落文本（基于平均长度或包含多字的文本）
        try:
            avg_len = sum(len(str(x)) for x in retrieved) / max(len(retrieved), 1)
        except Exception:
            avg_len = 0

        is_texts = avg_len > 30 or any(len(str(x)) > 30 for x in retrieved)

        if is_texts:
            # 使用语义相似度计算每个 ground-truth 段落的最佳匹配得分
            scores = []
            for gt in relevant:
                best = 0.0
                for r in retrieved:
                    try:
                        sim = self._semantic_similarity(str(gt), str(r))
                        if sim > best:
                            best = sim
                    except Exception:
                        continue
                scores.append(best)
            return float(np.mean(scores)) if scores else 0.0
        else:
            # 回退到集合匹配（去重）以避免重复计数导致 >1 的情况
            relevant_set = set(relevant)
            retrieved_set = set(retrieved)
            matched = len(relevant_set.intersection(retrieved_set))
            return matched / len(relevant_set)

    def _calculate_context_relevance(self, retrieved_paras: List[str], question: str) -> float:
        """
        计算问题与检索上下文的相关性（0-1）：
        - 对检索到的段落（或上下文片段）与问题分别计算语义相似度，取前 N 条的平均值作为相关性得分。
        - 保证返回值在 0-1 之间。
        """
        if not question:
            return 0.0

        if not retrieved_paras:
            return 0.0

        sims = []
        for para in (retrieved_paras if isinstance(retrieved_paras, list) else []):
            try:
                sim = self._semantic_similarity(question, str(para))
                sims.append(max(0.0, min(1.0, float(sim))))
            except Exception:
                continue

        if not sims:
            return 0.0

        # 平均相似度作为相关性得分
        return float(np.mean(sims))
    
    def _calculate_f1_score(self, answer: str, ground_truth: str, accuracy: float = None, context_relevance: float = None) -> float:
        """
        将 precision 定义为 `accuracy`（answer vs ground_truth 的语义相似度），
        将 recall 定义为 `context_relevance`（问题与检索上下文相关性），
        并计算 F1 = 2 * precision * recall / (precision + recall)，保证返回在 0-1 之间。
        如果任一值缺失，则尝试计算或使用 0.0 回退。
        """
        try:
            precision = float(accuracy) if accuracy is not None else self._semantic_similarity(answer, ground_truth)
        except Exception:
            precision = 0.0

        try:
            recall = float(context_relevance) if context_relevance is not None else 0.0
        except Exception:
            recall = 0.0

        # 保证在 [0,1]
        precision = max(0.0, min(1.0, precision))
        recall = max(0.0, min(1.0, recall))

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return float(max(0.0, min(1.0, f1)))

    def _semantic_similarity(self, a: str, b: str) -> float:
        """语义相似度近似：使用 SequenceMatcher 比例作为简易语义相似度（0-1）"""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    
    # def _calculate_answer_accuracy(self, answer: str, ground_truth: str) -> float:
    #     key_phrases = self._extract_key_phrases(ground_truth)
    #     if not key_phrases:
    #         return 0.0
        
    #     matches = 0
    #     for phrase in key_phrases:
    #         if phrase in answer:
    #             matches += 1
        
    #     return matches / len(key_phrases)
    
    # def _calculate_faithfulness(self, answer: str, system_result: Dict) -> float:
    #     # 旧的方法保留为兼容，但主要使用语义相似度
    #     retrieved = system_result.get("retrieval", {}).get("reranked_results", [])
    #     texts = [r.get("text_preview") or r.get("chunk_text") or "" for r in retrieved[:3]]
    #     combined = " \n ".join(texts)
    #     return self._semantic_similarity(answer, combined)

    def _calculate_semantic_faithfulness(self, answer: str, retrieved_chunks: List[Dict]) -> float:
        texts = [r.get("text_preview") or r.get("chunk_text") or "" for r in (retrieved_chunks if isinstance(retrieved_chunks, list) else [])]
        combined = " \n ".join(texts)
        return self._semantic_similarity(answer, combined)
    
    # def _calculate_answer_relevance(self, answer: str, question: str) -> float:
    #     # 使用语义相似度替代关键词匹配
    #     return self._semantic_similarity(answer, question)
    
    def _calculate_citation_accuracy(self, citations: List[str], relevant_docs: List[str]) -> float:
        # 目标：按验证集 md 文档的匹配数量计算准确率
        # 如果没有 supporting_facts，则返回 0
        if not relevant_docs:
            return 0.0

        if not citations:
            return 0.0

        # 提取 citations 中的文件名
        cited_files = set()
        for c in citations:
            # 支持 lines like: "1. 31450-2015-gbt-cd-300.md (段落34)：..."
            m = re.search(r'([\w\-\+]+\.md)', c)
            if m:
                cited_files.add(m.group(1))
            else:
                parts = re.split(r'[,;\s]', c)
                for p in parts:
                    if p.endswith('.md'):
                        cited_files.add(p.split('/')[-1])

        if not cited_files:
            return 0.0

        relevant_set = set(relevant_docs)
        matched = sum(1 for f in relevant_set if f in cited_files)
        return matched / len(relevant_set)
    
    def _calculate_hallucination_rate(self, answer: str, system_result: Dict) -> float:
        retrieved_entities = self._extract_entities_from_retrieval(system_result)
        answer_entities = self._extract_entities(answer)
        
        if not answer_entities:
            return 0.0
        
        hallucinated = sum(1 for entity in answer_entities if entity not in retrieved_entities)
        return hallucinated / len(answer_entities)
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        phrases = []
        
        number_pattern = r'\d+\.?\d*\s*(mm|MPa|℃|°C|g/cm³|kN|小时|分钟)'
        phrases.extend(re.findall(number_pattern, text))
        
        standard_pattern = r'(HB|GB/T|GJB)\s*\d+[-\s]\d{4}'
        phrases.extend(re.findall(standard_pattern, text))
        
        material_pattern = r'[A-Z]{2,}\d+[A-Z]*'
        phrases.extend(re.findall(material_pattern, text))
        
        return list(set(phrases))
    
    def _extract_keywords(self, text: str) -> List[str]:
        stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        
        words = re.findall(r'[\u4e00-\u9fff]+|[A-Za-z]+|\d+', text)
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        
        return list(set(keywords))
    
    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        
        entities.extend(re.findall(r'(HB|GB/T|GJB)\s*\d+[-\s]\d{4}', text))
        entities.extend(re.findall(r'[A-Z]{2,}\d+[A-Z]*', text))
        entities.extend(re.findall(r'\d+\.?\d*', text))
        
        return list(set(entities))
    
    def _extract_entities_from_retrieval(self, system_result: Dict) -> Set[str]:
        entities = set()
        
        for chunk in system_result.get("retrieval", {}).get("vector_results", []):
            chunk_text = chunk.get("chunk_text", "")
            entities.update(self._extract_entities(chunk_text))
        
        kg_entities = system_result.get("retrieval", {}).get("kg_results", {}).get("entities", [])
        entities.update(kg_entities)
        
        return entities
    
    def _aggregate_metrics(self, all_metrics: Dict[str, List]) -> EvaluationResult:
        eval_result = EvaluationResult()
        
        retrieval_metrics = all_metrics["retrieval_metrics"]
        if retrieval_metrics:
            eval_result.hit_at_1 = np.mean([m.get("hit_at_1", 0) for m in retrieval_metrics])
            eval_result.hit_at_3 = np.mean([m.get("hit_at_3", 0) for m in retrieval_metrics])
            eval_result.hit_at_5 = np.mean([m.get("hit_at_5", 0) for m in retrieval_metrics])
            # 段落级指标
            eval_result.para_hit_at_1 = np.mean([m.get("para_hit_at_1", 0) for m in retrieval_metrics])
            eval_result.para_hit_at_5 = np.mean([m.get("para_hit_at_5", 0) for m in retrieval_metrics])
            eval_result.para_hit_at_10 = np.mean([m.get("para_hit_at_10", 0) for m in retrieval_metrics])
            eval_result.para_recall_at_1 = np.mean([m.get("para_recall_at_1", 0) for m in retrieval_metrics])
            eval_result.para_recall_at_5 = np.mean([m.get("para_recall_at_5", 0) for m in retrieval_metrics])
            eval_result.para_recall_at_10 = np.mean([m.get("para_recall_at_10", 0) for m in retrieval_metrics])
            eval_result.recall_at_1 = np.mean([m.get("recall_at_1", 0) for m in retrieval_metrics])
            eval_result.recall_at_3 = np.mean([m.get("recall_at_3", 0) for m in retrieval_metrics])
            eval_result.recall_at_5 = np.mean([m.get("recall_at_5", 0) for m in retrieval_metrics])
            eval_result.mrr = np.mean([m.get("mrr", 0) for m in retrieval_metrics])
            eval_result.context_precision = np.mean([m.get("context_precision", 0) for m in retrieval_metrics])
            eval_result.context_recall = np.mean([m.get("context_recall", 0) for m in retrieval_metrics])
            eval_result.context_relevance = np.mean([m.get("context_relevance", 0) for m in retrieval_metrics])
        
        generation_metrics = all_metrics["generation_metrics"]
        if generation_metrics:
            eval_result.em_score = np.mean([m.get("em_score", 0) for m in generation_metrics])
            eval_result.f1_score = np.mean([m.get("f1_score", 0) for m in generation_metrics])
            eval_result.accuracy = np.mean([m.get("accuracy", 0) for m in generation_metrics])
            eval_result.faithfulness = np.mean([m.get("faithfulness", 0) for m in generation_metrics])
            eval_result.answer_relevance = np.mean([m.get("answer_relevance", 0) for m in generation_metrics])
            eval_result.citation_accuracy = np.mean([m.get("citation_accuracy", 0) for m in generation_metrics])
            eval_result.hallucination_rate = np.mean([m.get("hallucination_rate", 0) for m in generation_metrics])
        
        performance_metrics = all_metrics["performance_metrics"]
        if performance_metrics:
            eval_result.retrieval_time = np.mean([m.get("retrieval_time", 0) for m in performance_metrics])
            eval_result.generation_time = np.mean([m.get("generation_time", 0) for m in performance_metrics])
            eval_result.total_time = np.mean([m.get("total_time", 0) for m in performance_metrics])
        
        return eval_result
    
    def _save_evaluation_results(self, result: EvaluationResult, sample_size: int, confusion_matrix: dict = None, per_item_details: List[Dict] = None):
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 保存到项目 results 目录下
        filename = f"evaluation_{timestamp}_n{sample_size}.json"
        filepath = Path("/home/zzm/Project_1/kg-hk/4_RAG_method/results") / filename

        result_dict = result.to_dict()
        result_dict["sample_size"] = sample_size
        result_dict["timestamp"] = timestamp
        if confusion_matrix:
            result_dict["confusion_matrix"] = confusion_matrix
        if per_item_details:
            result_dict["per_item_details"] = per_item_details

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        self._generate_report(result_dict, filepath)

        # 生成按 QA 类型的单独报告：优先按 gt_type 分组；若没有 gt_type，则按 pred_type 分组
        try:
            if per_item_details:
                self._generate_type_reports(result_dict, filepath, per_item_details)
        except Exception as e:
            logger.warning(f"生成按类型报告失败: {e}")
    
    def _generate_report(self, result: Dict, filepath: Path):
        # 构建更详细的 Markdown 报告，包含段落级指标与混淆矩阵
        lines = []
        lines.append("# RAG系统评估报告")
        lines.append("")
        lines.append("## 基本信息")
        lines.append(f"- 评估时间: {result.get('timestamp', '')}")
        lines.append(f"- 样本大小: {result.get('sample_size', 0)}")
        lines.append("")
        lines.append("## 检索性能（文档级）")
        lines.append(f"- Hit@1: {result.get('hit_at_1', 0):.3f}")
        lines.append(f"- Hit@3: {result.get('hit_at_3', 0):.3f}")
        lines.append(f"- Hit@5: {result.get('hit_at_5', 0):.3f}")
        lines.append(f"- Recall@1: {result.get('recall_at_1', 0):.3f}")
        lines.append(f"- Recall@3: {result.get('recall_at_3', 0):.3f}")
        lines.append(f"- Recall@5: {result.get('recall_at_5', 0):.3f}")
        lines.append(f"- MRR: {result.get('mrr', 0):.3f}")
        lines.append("")
        lines.append("## 检索性能（段落级）")
        lines.append(f"- Para Hit@1: {result.get('para_hit_at_1', result.get('retrieval', {}).get('para_hit_at_1', 0)):.3f}")
        lines.append(f"- Para Hit@5: {result.get('para_hit_at_5', result.get('retrieval', {}).get('para_hit_at_5', 0)):.3f}")
        lines.append(f"- Para Hit@10: {result.get('para_hit_at_10', result.get('retrieval', {}).get('para_hit_at_10', 0)):.3f}")
        lines.append(f"- Para Recall@1: {result.get('para_recall_at_1', result.get('retrieval', {}).get('para_recall_at_1', 0)):.3f}")
        lines.append(f"- Para Recall@5: {result.get('para_recall_at_5', result.get('retrieval', {}).get('para_recall_at_5', 0)):.3f}")
        lines.append(f"- Para Recall@10: {result.get('para_recall_at_10', result.get('retrieval', {}).get('para_recall_at_10', 0)):.3f}")
        lines.append("")
        lines.append("## 检索上下文质量")
        lines.append(f"- 上下文精度: {result.get('context_precision', result.get('retrieval', {}).get('context_precision', 0)):.3f}")
        lines.append(f"- 上下文召回率: {result.get('context_recall', result.get('retrieval', {}).get('context_recall', 0)):.3f}")
        lines.append(f"- 上下文相关性: {result.get('context_relevance', result.get('retrieval', {}).get('context_relevance', 0)):.3f}")
        lines.append("")
        lines.append("## 生成质量")
        lines.append(f"- 精确匹配 (EM): {result.get('em_score', result.get('generation', {}).get('em_score', 0)):.3f}")
        lines.append(f"- F1分数: {result.get('f1_score', result.get('generation', {}).get('f1_score', 0)):.3f}")
        lines.append(f"- 准确率: {result.get('accuracy', result.get('generation', {}).get('accuracy', 0)):.3f}")
        lines.append(f"- 忠实度: {result.get('faithfulness', result.get('generation', {}).get('faithfulness', 0)):.3f}")
        lines.append(f"- 答案相关性: {result.get('answer_relevance', result.get('generation', {}).get('answer_relevance', 0)):.3f}")
        lines.append("")
        lines.append("## 可信度")
        lines.append(f"- 引用准确率: {result.get('citation_accuracy', result.get('generation', {}).get('citation_accuracy', 0)):.3f}")
        lines.append(f"- 幻觉率: {result.get('hallucination_rate', result.get('generation', {}).get('hallucination_rate', 0)):.3f}")
        lines.append("")
        lines.append("## 路由混淆矩阵")
        cm = result.get('confusion_matrix', {})
        if cm:
            lines.append("")
            lines.append("| GT \\ Pred | " + " | ".join(sorted({p for preds in cm.values() for p in preds.keys()})) + " |")
            lines.append("|---" + "|---" * len({p for preds in cm.values() for p in preds.keys()}) + "|")
            preds_sorted = sorted({p for preds in cm.values() for p in preds.keys()})
            for gt in sorted(cm.keys()):
                row = [str(cm[gt].get(p, 0)) for p in preds_sorted]
                lines.append("| {} | {} |".format(gt, " | ".join(row)))
        else:
            lines.append("- 无混淆矩阵数据")
        lines.append("")
        # lines.append("## 引用详情（示例）")
        # lines.append("- 输出文件包含每项指标的汇总数值。下表显示每条样本的引用提取与匹配详情（若有）。")
        # lines.append("")
        # # 如果存在逐项详情，列出前 20 条的引用匹配情况
        # pid = result.get('per_item_details', [])
        # if pid:
        #     lines.append("| id | extracted_citations | supporting_facts_docs | matched_docs | citation_accuracy |")
        #     lines.append("|---|---|---|---|---|")
        #     for item in pid[:20]:
        #         ext = ",".join(item.get('extracted_citations', []))
        #         sfs = ",".join(item.get('supporting_facts_docs', []))
        #         matched = ",".join(item.get('matched_docs', []))
        #         acc = item.get('citation_accuracy', 0.0)
        #         lines.append(f"| {item.get('id')} | {ext} | {sfs} | {matched} | {acc:.3f} |")
        # else:
        #     lines.append("- 无逐条引用匹配详情")
        # lines.append("")
        lines.append("## 性能指标")
        lines.append(f"- 平均检索时间: {result.get('retrieval_time', result.get('performance', {}).get('retrieval_time', 0)):.3f}秒")
        lines.append(f"- 平均生成时间: {result.get('generation_time', result.get('performance', {}).get('generation_time', 0)):.3f}秒")
        lines.append(f"- 平均总时间: {result.get('total_time', result.get('performance', {}).get('total_time', 0)):.3f}秒")

        report = "\n".join(lines)
        report_path = filepath.with_suffix('.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"评估报告已生成: {report_path}")


    def _generate_type_reports(self, result: Dict, filepath: Path, per_item_details: List[Dict]):
        """为每个 QA 类型生成单独的评估报告（优先使用 gt_type）。
        同时生成按 pred_type 的诊断报告以便分析路由/分类误差。
        """
        base_dir = filepath.parent
        timestamp = result.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))

        # 准备按 gt_type 分组，如果 gt_type 全部为空则按 pred_type 分组
        groups_by_gt = {}
        groups_by_pred = {}
        for item in per_item_details:
            gt = item.get('gt_type') or ''
            pred = item.get('pred_type') or ''
            groups_by_gt.setdefault(gt, []).append(item)
            groups_by_pred.setdefault(pred, []).append(item)

        # 选择优先分组：若存在非空 gt keys 则以 gt 为主，否则以 pred 为主
        use_gt = any(k for k in groups_by_gt.keys())
        primary_groups = groups_by_gt if use_gt else groups_by_pred
        primary_key_name = 'gt_type' if use_gt else 'pred_type'

        # 辅助函数：计算一组 items 的聚合指标
        def aggregate_items(items: List[Dict]) -> Dict:
            agg = {
                'count': len(items),
                'hit_at_1': 0.0, 'hit_at_3': 0.0, 'hit_at_5': 0.0,
                'recall_at_1': 0.0, 'recall_at_3': 0.0, 'recall_at_5': 0.0, 'mrr': 0.0,
                'para_hit_at_1': 0.0, 'para_hit_at_5': 0.0, 'para_hit_at_10': 0.0,
                'para_recall_at_1': 0.0, 'para_recall_at_5': 0.0, 'para_recall_at_10': 0.0,
                'context_precision': 0.0, 'context_recall': 0.0, 'context_relevance': 0.0,
                'em_score': 0.0, 'f1_score': 0.0, 'accuracy': 0.0, 'faithfulness': 0.0,
                'citation_accuracy': 0.0, 'hallucination_rate': 0.0,
                'retrieval_time': 0.0, 'generation_time': 0.0, 'total_time': 0.0
            }
            if not items:
                return agg

            n = len(items)
            for it in items:
                m_retr = it.get('metrics', {}).get('retrieval', {})
                m_gen = it.get('metrics', {}).get('generation', {})
                m_perf = it.get('metrics', {}).get('performance', {})

                agg['hit_at_1'] += m_retr.get('hit_at_1', 0)
                agg['hit_at_3'] += m_retr.get('hit_at_3', 0)
                agg['hit_at_5'] += m_retr.get('hit_at_5', 0)
                agg['recall_at_1'] += m_retr.get('recall_at_1', 0)
                agg['recall_at_3'] += m_retr.get('recall_at_3', 0)
                agg['recall_at_5'] += m_retr.get('recall_at_5', 0)
                agg['mrr'] += m_retr.get('mrr', 0)
                agg['para_hit_at_1'] += m_retr.get('para_hit_at_1', 0)
                agg['para_hit_at_5'] += m_retr.get('para_hit_at_5', 0)
                agg['para_hit_at_10'] += m_retr.get('para_hit_at_10', 0)
                agg['para_recall_at_1'] += m_retr.get('para_recall_at_1', 0)
                agg['para_recall_at_5'] += m_retr.get('para_recall_at_5', 0)
                agg['para_recall_at_10'] += m_retr.get('para_recall_at_10', 0)
                agg['context_precision'] += m_retr.get('context_precision', 0)
                agg['context_recall'] += m_retr.get('context_recall', 0)
                agg['context_relevance'] += m_retr.get('context_relevance', 0)

                agg['em_score'] += m_gen.get('em_score', 0)
                agg['f1_score'] += m_gen.get('f1_score', 0)
                agg['accuracy'] += m_gen.get('accuracy', 0)
                agg['faithfulness'] += m_gen.get('faithfulness', 0)
                agg['citation_accuracy'] += m_gen.get('citation_accuracy', 0)
                agg['hallucination_rate'] += m_gen.get('hallucination_rate', 0)

                agg['retrieval_time'] += m_perf.get('retrieval_time', 0)
                agg['generation_time'] += m_perf.get('generation_time', 0)
                agg['total_time'] += m_perf.get('total_time', 0)

            # 平均化
            for k in list(agg.keys()):
                if k != 'count':
                    agg[k] = agg[k] / n

            return agg

        # 为每个类型生成 md 文件
        for t, items in primary_groups.items():
            label = t if t else 'UNKNOWN'
            stats = aggregate_items(items)
            md_lines = []
            md_lines.append(f"# 类型评估报告 — {primary_key_name}: {label}")
            md_lines.append("")
            md_lines.append(f"- 总样本数: {stats['count']}")
            md_lines.append("")
            md_lines.append("## 检索 (文档级)")
            md_lines.append(f"- Hit@1: {stats['hit_at_1']:.3f}")
            md_lines.append(f"- Hit@3: {stats['hit_at_3']:.3f}")
            md_lines.append(f"- Hit@5: {stats['hit_at_5']:.3f}")
            md_lines.append(f"- Recall@1: {stats['recall_at_1']:.3f}")
            md_lines.append(f"- Recall@3: {stats['recall_at_3']:.3f}")
            md_lines.append(f"- Recall@5: {stats['recall_at_5']:.3f}")
            md_lines.append(f"- MRR: {stats['mrr']:.3f}")
            md_lines.append("")
            md_lines.append("## 检索 (段落级)")
            md_lines.append(f"- Para Hit@1: {stats['para_hit_at_1']:.3f}")
            md_lines.append(f"- Para Hit@5: {stats['para_hit_at_5']:.3f}")
            md_lines.append(f"- Para Hit@10: {stats['para_hit_at_10']:.3f}")
            md_lines.append(f"- Para Recall@1: {stats['para_recall_at_1']:.3f}")
            md_lines.append(f"- Para Recall@5: {stats['para_recall_at_5']:.3f}")
            md_lines.append(f"- Para Recall@10: {stats['para_recall_at_10']:.3f}")
            md_lines.append("")
            md_lines.append("## 上下文质量")
            md_lines.append(f"- 上下文精度: {stats['context_precision']:.3f}")
            md_lines.append(f"- 上下文召回: {stats['context_recall']:.3f}")
            md_lines.append(f"- 上下文相关性: {stats['context_relevance']:.3f}")
            md_lines.append("")
            md_lines.append("## 生成质量")
            md_lines.append(f"- EM: {stats['em_score']:.3f}")
            md_lines.append(f"- F1: {stats['f1_score']:.3f}")
            md_lines.append(f"- 准确率: {stats['accuracy']:.3f}")
            md_lines.append(f"- 忠实度: {stats['faithfulness']:.3f}")
            md_lines.append(f"- 引用准确率: {stats['citation_accuracy']:.3f}")
            md_lines.append(f"- 幻觉率: {stats['hallucination_rate']:.3f}")
            md_lines.append("")
            md_lines.append("## 性能")
            md_lines.append(f"- 平均检索时间: {stats['retrieval_time']:.3f}s")
            md_lines.append(f"- 平均生成时间: {stats['generation_time']:.3f}s")
            md_lines.append(f"- 平均总时间: {stats['total_time']:.3f}s")

            type_report_path = base_dir / f"evaluation_{timestamp}_by_{primary_key_name}_{label}.md"
            with open(type_report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))

        # 另外产出 pred_type 的诊断报告（帮助分析路由错误）
        
        for t, items in groups_by_pred.items():
            label = t if t else 'UNKNOWN'
            stats = aggregate_items(items)
            md_lines = []
            md_lines.append(f"# 类型评估报告 — {primary_key_name}: {label}")
            md_lines.append("")
            md_lines.append(f"- 总样本数: {stats['count']}")
            md_lines.append("")
            md_lines.append("## 检索 (文档级)")
            md_lines.append(f"- Hit@1: {stats['hit_at_1']:.3f}")
            md_lines.append(f"- Hit@3: {stats['hit_at_3']:.3f}")
            md_lines.append(f"- Hit@5: {stats['hit_at_5']:.3f}")
            md_lines.append(f"- Recall@1: {stats['recall_at_1']:.3f}")
            md_lines.append(f"- Recall@3: {stats['recall_at_3']:.3f}")
            md_lines.append(f"- Recall@5: {stats['recall_at_5']:.3f}")
            md_lines.append(f"- MRR: {stats['mrr']:.3f}")
            md_lines.append("")
            md_lines.append("## 检索 (段落级)")
            md_lines.append(f"- Para Hit@1: {stats['para_hit_at_1']:.3f}")
            md_lines.append(f"- Para Hit@5: {stats['para_hit_at_5']:.3f}")
            md_lines.append(f"- Para Hit@10: {stats['para_hit_at_10']:.3f}")
            md_lines.append(f"- Para Recall@1: {stats['para_recall_at_1']:.3f}")
            md_lines.append(f"- Para Recall@5: {stats['para_recall_at_5']:.3f}")
            md_lines.append(f"- Para Recall@10: {stats['para_recall_at_10']:.3f}")
            md_lines.append("")
            md_lines.append("## 上下文质量")
            md_lines.append(f"- 上下文精度: {stats['context_precision']:.3f}")
            md_lines.append(f"- 上下文召回: {stats['context_recall']:.3f}")
            md_lines.append(f"- 上下文相关性: {stats['context_relevance']:.3f}")
            md_lines.append("")
            md_lines.append("## 生成质量")
            md_lines.append(f"- EM: {stats['em_score']:.3f}")
            md_lines.append(f"- F1: {stats['f1_score']:.3f}")
            md_lines.append(f"- 准确率: {stats['accuracy']:.3f}")
            md_lines.append(f"- 忠实度: {stats['faithfulness']:.3f}")
            md_lines.append(f"- 引用准确率: {stats['citation_accuracy']:.3f}")
            md_lines.append(f"- 幻觉率: {stats['hallucination_rate']:.3f}")
            md_lines.append("")
            md_lines.append("## 性能")
            md_lines.append(f"- 平均检索时间: {stats['retrieval_time']:.3f}s")
            md_lines.append(f"- 平均生成时间: {stats['generation_time']:.3f}s")
            md_lines.append(f"- 平均总时间: {stats['total_time']:.3f}s")
            
            pred_report_path = base_dir / f"evaluation_{timestamp}_by_pred_type_{label}.md"
            with open(pred_report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))
