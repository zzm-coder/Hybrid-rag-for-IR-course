import os
import json
import uuid
import numpy as np
import faiss
import random
import re
from tqdm import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any, Optional, Set, Tuple

# ==================== 配置 ====================
MD_ROOT = "/home/zzm/Project_1/kg-hk/0_mineru_pdf/data_md_final"
NEO4J_URI = "URL"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "PASSWORD"

LLM_URL = "URL"
LLM_API_KEY = "EMPTY"
LLM_MODEL = "MODEL_NAME"

SEMANTIC_MODEL_PATH = "/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese"

VECTOR_DB_PATH = "/home/zzm/Project_1/kg-hk/2_kg_construction/kg_vector_db"
CHUNK_JSON = os.path.join(VECTOR_DB_PATH, "chunks.json")
EMBEDDINGS_NPY = os.path.join(VECTOR_DB_PATH, "embeddings.npy")
FAISS_INDEX = os.path.join(VECTOR_DB_PATH, "faiss.index")

KG_CACHE_PATH = "/home/zzm/Project_1/kg-hk/2_kg_construction/kg_cache"
ENTITY_CACHE_JSON = os.path.join(KG_CACHE_PATH, "entity_cache.json")

OUTPUT_DIR = "/home/zzm/Project_1/kg-hk/3_QA_creation/3_QA_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# OUTPUT_QA_ALL = os.path.join(OUTPUT_DIR, "qa_all_000_200.jsonl")
# OUTPUT_QA_ALL = os.path.join(OUTPUT_DIR, "qa_all_200_400.jsonl")
# OUTPUT_QA_ALL = os.path.join(OUTPUT_DIR, "qa_all_400_600.jsonl")
# OUTPUT_QA_ALL = os.path.join(OUTPUT_DIR, "qa_all_600_800.jsonl")
OUTPUT_QA_ALL = os.path.join(OUTPUT_DIR, "qa_all_800_932.jsonl")

# ==================== 工具函数 ====================

def serialize_neo4j_obj(obj):
    result = {}
    for key, value in obj.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif str(type(value)) in [
            "<class 'neo4j.time.DateTime'>",
            "<class 'neo4j.time.Date'>",
            "<class 'neo4j.time.Time'>",
            "<class 'neo4j.time.LocalDateTime'>"
        ]:
            continue
        else:
            result[key] = str(value)
    return result

def triple_to_sentence(triple: Dict) -> str:
    return f"{triple['head']} {triple['relation']} {triple['tail']}"

def deduplicate_triples(triples: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for t in triples:
        key = (t.get("head", "").strip(), t.get("relation", "").strip(), t.get("tail", "").strip())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique

def trim_subgraph_triples(triples: List[Dict], max_relations: int = 10, current_source: str = "") -> List[Dict]:
    triples_sorted = sorted(triples, key=lambda x: x.get("source", "") != current_source)
    selected = triples_sorted[:max_relations]
    trimmed = []
    for t in selected:
        para = t.get("paragraph", "")
        if len(para) > 200:
            para = para[:200] + "..."
        trimmed.append({
            "head": t.get("head", ""),
            "relation": t.get("relation", ""),
            "tail": t.get("tail", ""),
            "source": t.get("source", ""),
            "paragraph": para
        })
    return trimmed

def extract_unique_paragraphs(triples: List[Dict], max_paragraphs: int = 5) -> List[str]:
    seen = set()
    paragraphs = []
    for t in triples:
        para = t.get("paragraph", "").strip()
        if para and para not in seen:
            seen.add(para)
            if len(para) > 300:
                para = para[:300] + "..."
            paragraphs.append(para)
            if len(paragraphs) >= max_paragraphs:
                break
    return paragraphs

# ==================== 初始化 ====================
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
semantic_model = SentenceTransformer(SEMANTIC_MODEL_PATH, device="cuda")

with open(CHUNK_JSON, "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)

chunks = []
for item in raw_chunks:
    source = item.get("metadata", {}).get("file_name", "unknown.md")
    text = item.get("original_text", "")
    chunks.append({
        "source": source,
        "chunk": text
    })

embeddings = np.load(EMBEDDINGS_NPY)
index = faiss.read_index(FAISS_INDEX)

with open(ENTITY_CACHE_JSON, "r", encoding="utf-8") as f:
    entity_cache = json.load(f)

# ==================== 去重系统 ====================
dedup_index = None
dedup_questions = []
dedup_threshold = 0.92

def init_dedup_index():
    global dedup_index
    dim = semantic_model.get_sentence_embedding_dimension()
    dedup_index = faiss.IndexFlatIP(dim)

def is_duplicate_question(question: str) -> bool:
    global dedup_index, dedup_questions
    if len(dedup_questions) == 0:
        emb = semantic_model.encode([question], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(emb)
        dedup_index.add(emb)
        dedup_questions.append(question)
        return False
    emb = semantic_model.encode([question], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(emb)
    D, I = dedup_index.search(emb, 1)
    if D[0][0] > dedup_threshold:
        return True
    else:
        dedup_index.add(emb)
        dedup_questions.append(question)
        return False

# ==================== Neo4j 子图查询 ====================

def get_one_hop_subgraph(tx, node_element_id: str):
    query = "MATCH (n)-[r]-(m) WHERE elementId(n) = $node_id RETURN n, r, m"
    result = tx.run(query, node_id=node_element_id)
    triples = []
    entities = {}
    for record in result:
        src = serialize_neo4j_obj(record["n"])
        rel = serialize_neo4j_obj(record["r"])
        tgt = serialize_neo4j_obj(record["m"])
        src["id"] = record["n"].element_id
        tgt["id"] = record["m"].element_id
        entities[src["id"]] = src
        entities[tgt["id"]] = tgt
        triples.append({
            "head": src.get("name", ""), "relation": rel.get("relation_type", ""),
            "tail": tgt.get("name", ""), "source": rel.get("source", ""),
            "paragraph": rel.get("paragraph", "")
        })
    valid_triples = [
        t for t in triples
        if t.get("head") and t.get("relation") and t.get("tail") and t.get("source") and t.get("paragraph")
    ]
    valid_triples = deduplicate_triples(valid_triples)
    return {"entities": list(entities.values()), "triples": valid_triples}

def get_k_hop_paths(tx, start_node_id: str, max_hops: int = 4):
    paths = []
    for hops in range(2, min(max_hops + 1, 5)):
        node_vars = [f"n{i}" for i in range(hops + 1)]
        rel_vars = [f"r{i}" for i in range(hops)]
        match_clause = f"MATCH ({node_vars[0]})" + "".join(
            f"-[{rel_vars[i]}]-({node_vars[i+1]})" for i in range(hops)
        )
        where_clause = f"WHERE elementId({node_vars[0]}) = $start_id"
        distinct_conditions = []
        for i in range(hops + 1):
            for j in range(i + 1, hops + 1):
                distinct_conditions.append(f"elementId({node_vars[i]}) <> elementId({node_vars[j]})")
        if distinct_conditions:
            where_clause += " AND " + " AND ".join(distinct_conditions)
        return_clause = "RETURN " + ", ".join(node_vars + rel_vars)
        query = f"{match_clause} {where_clause} {return_clause} LIMIT 10"
        try:
            result = tx.run(query, start_id=start_node_id)
            for record in result:
                path = []
                for i in range(hops + 1):
                    node = serialize_neo4j_obj(record[node_vars[i]])
                    node["id"] = record[node_vars[i]].element_id
                    path.append(("node", node))
                    if i < hops:
                        rel = serialize_neo4j_obj(record[rel_vars[i]])
                        path.append(("rel", rel))
                paths.append(path)
                if len(paths) >= 15:
                    break
        except Exception as e:
            continue
        if len(paths) >= 15:
            break
    return paths

def get_triples_by_source_and_paragraph(tx, source: str, paragraph: str) -> List[Dict]:
    query = """
    MATCH (a)-[r]->(b)
    WHERE r.source = $source AND r.paragraph = $paragraph
    RETURN a, r, b
    """
    result = tx.run(query, source=source, paragraph=paragraph)
    triples = []
    for record in result:
        a = serialize_neo4j_obj(record["a"])
        r = serialize_neo4j_obj(record["r"])
        b = serialize_neo4j_obj(record["b"])
        triple = {
            "head": a.get("name", ""),
            "relation": r.get("relation_type", ""),
            "tail": b.get("name", ""),
            "source": r.get("source", ""),
            "paragraph": r.get("paragraph", "")
        }
        if all(triple.get(k) for k in ["head", "relation", "tail", "source", "paragraph"]):
            triples.append(triple)
    return triples

# ==================== LLM 调用 ====================

def call_llm(prompt: str, json_mode: bool = True) -> Optional[str]:
    if len(prompt) > 120000:
        return None
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 10000,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        resp = requests.post(f"{LLM_URL}/chat/completions", headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            if json_mode:
                return json.loads(content)
            else:
                return content.strip()
        else:
            return None
    except Exception as e:
        return None

# ==================== 语义检索 ====================

def find_semantic_similar_chunks(paragraph: str, top_k=3):
    if not paragraph.strip():
        return []
    emb = semantic_model.encode([paragraph], convert_to_numpy=True).astype('float32')
    D, I = index.search(emb, top_k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results

# ==================== 核心验证函数 ====================

def get_relevant_facts_by_question_and_answer(
    question: str,
    answer: str,
    candidate_triples: List[Dict],
    top_k: int = 10,
    qa_type: str = "simple_fact"  # 新增参数
) -> List[Dict]:
    if not candidate_triples:
        return []
    
    candidate_texts = []
    text_to_triple = {}
    for t in candidate_triples:
        para = t["paragraph"].strip()
        if para:
            key = ("para", para, t["source"])
            candidate_texts.append(para)
            text_to_triple[key] = t
        sent = triple_to_sentence(t)
        if sent.strip():
            key = ("triple", sent, t["source"])
            candidate_texts.append(sent)
            text_to_triple[key] = t

    if not candidate_texts:
        return []

    q_emb = semantic_model.encode([question], convert_to_numpy=True).astype('float32')
    a_emb = semantic_model.encode([answer], convert_to_numpy=True).astype('float32')
    t_embs = semantic_model.encode(candidate_texts, convert_to_numpy=True).astype('float32')

    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    a_emb = a_emb / np.linalg.norm(a_emb, axis=1, keepdims=True)
    t_embs = t_embs / np.linalg.norm(t_embs, axis=1, keepdims=True)

    # 动态阈值
    q_threshold = 0.6 if qa_type == "complex_logic" else 0.7
    a_threshold = 0.5 if qa_type == "complex_logic" else 0.6

    q_sims = (q_emb @ t_embs.T)[0]
    stage1_indices = np.where(q_sims > q_threshold)[0]
    if len(stage1_indices) == 0:
        return []

    a_sims = (a_emb @ t_embs[stage1_indices].T)[0]
    valid_mask = a_sims > a_threshold
    final_indices = stage1_indices[valid_mask]
    if len(final_indices) == 0:
        return []

    sorted_indices = final_indices[np.argsort(-q_sims[final_indices])[:top_k]]

    fact_map = {}
    for idx in sorted_indices:
        for key, t in text_to_triple.items():
            if candidate_texts[idx] in key:
                if not all([t.get("head"), t.get("relation"), t.get("tail"), t.get("source"), t.get("paragraph")]):
                    continue
                fact_key = (t["source"], t["paragraph"])
                if fact_key not in fact_map:
                    fact_map[fact_key] = {"source": t["source"], "chunk": t["paragraph"], "triples": []}
                fact_map[fact_key]["triples"].append({
                    "head": t["head"], "relation": t["relation"], "tail": t["tail"]
                })
                break

    for fact in fact_map.values():
        fact["triples"] = deduplicate_triples(fact["triples"])

    result = [{"id": i+1, **v} for i, v in enumerate(fact_map.values())]
    return result

def get_sources_from_subgraph(triples: List[Dict]) -> Set[str]:
    return {t["source"] for t in triples}

def validate_and_fix_answer(
    question: str,
    answer: str,
    supporting_facts: List[Dict],
    qa_type: str,
    available_sources: Set[str]
) -> Tuple[bool, str, List[Dict]]:
    relevant_chunks = [c for c in chunks if c.get("source") in available_sources]
    if not relevant_chunks:
        return True, answer, supporting_facts

    q_emb = semantic_model.encode([question], convert_to_numpy=True).astype('float32')
    q_emb = q_emb / np.linalg.norm(q_emb)
    a_emb = semantic_model.encode([answer], convert_to_numpy=True).astype('float32')
    a_emb = a_emb / np.linalg.norm(a_emb)

    chunk_texts = [c["chunk"] for c in relevant_chunks]
    c_embs = semantic_model.encode(chunk_texts, convert_to_numpy=True).astype('float32')
    c_embs = c_embs / np.linalg.norm(c_embs, axis=1, keepdims=True)

    q_sims = (q_emb @ c_embs.T)[0]
    a_sims = (a_emb @ c_embs.T)[0]

    uncovered = []
    for i, (q_sim, a_sim) in enumerate(zip(q_sims, a_sims)):
        if q_sim > 0.8 and a_sim < 0.7:
            uncovered.append(relevant_chunks[i])

    if not uncovered:
        return True, answer, supporting_facts

    original_context = "\n".join([f["chunk"] for f in supporting_facts])
    uncovered_context = "\n".join([c["chunk"] for c in uncovered[:3]])
    all_context = (original_context + "\n" + uncovered_context).strip()

    sources_str = ", ".join(list({c["source"] for c in uncovered[:3]}))
    prompt = f"""
你之前生成的答案可能不完整。请基于以下完整上下文，重新生成一个**全面且准确**的答案。
问题：{question}
原始答案：{answer}
完整上下文（来自 {sources_str}）：
{all_context}
要求： 
- 答案必须整合所有相关信息，且满足问题要求；
- 主语明确，无代词；
- 仅输出答案文本，不要任何前缀或解释。
"""
    new_answer = call_llm(prompt, json_mode=False)
    if new_answer:
        new_answer = re.sub(r'(?:<think>.*?</think>)', '', new_answer, flags=re.DOTALL | re.IGNORECASE)
    if new_answer and len(new_answer) > 10:
        all_triples = []
        for f in supporting_facts:
            for t in f["triples"]:
                all_triples.append({
                    "head": t["head"], "relation": t["relation"], "tail": t["tail"],
                    "source": f["source"], "paragraph": f["chunk"]
                })
        with neo4j_driver.session() as session:
            for c in uncovered[:3]:
                source = c["source"]
                chunk = c["chunk"]
                triples_from_db = session.execute_read(
                    get_triples_by_source_and_paragraph, source, chunk
                )
                if triples_from_db:
                    all_triples.extend(triples_from_db)
                else:
                    if source and chunk:
                        all_triples.append({
                            "head": source,
                            "relation": "contains_text",
                            "tail": chunk[:50],
                            "source": source,
                            "paragraph": chunk
                        })
        all_triples = deduplicate_triples(all_triples)
        new_supporting = get_relevant_facts_by_question_and_answer(question, new_answer, all_triples, top_k=10, qa_type=qa_type)
        return False, new_answer, new_supporting
    else:
        return False, answer, supporting_facts

def validate_supporting_facts_by_type(qa_type: str, supporting_facts: List[Dict], available_sources: Set[str]) -> bool:
    sources_in_facts = {f["source"] for f in supporting_facts}
    if qa_type == "simple_fact":
        return len(sources_in_facts) == 1
    else:
        # Complex/Open: 必须至少2个 supporting_facts
        if len(supporting_facts) < 2:
            return False
        if len(available_sources) >= 2:
            return len(sources_in_facts) >= 2
        else:
            return True

# ==================== 新增：LLM 一致性验证 ====================

def generate_reference_answer(question: str, supporting_facts: List[Dict]) -> Optional[str]:
    context_lines = []
    for f in supporting_facts:
        context_lines.append(f"来源: {f['source']}")
        context_lines.append(f"原文: {f['chunk']}")
        for t in f["triples"]:
            context_lines.append(f"  - {t['head']} {t['relation']} {t['tail']}")
        context_lines.append("")
    
    context = "\n".join(context_lines).strip()
    if not context:
        return None

    prompt = f"""
你是一个技术标准专家。请基于以下事实，**直接回答问题**。
要求：
- 仅输出答案，不要解释、不要前缀；
- 答案必须基于以下事实，不得编造；
- 主语明确，无代词。

问题：{question}

事实依据：
{context}
"""
    return call_llm(prompt, json_mode=False)

def is_answer_consistent(original_answer: str, reference_answer: str) -> bool:
    if not reference_answer or not original_answer:
        return False
    try:
        emb1 = semantic_model.encode([original_answer], convert_to_numpy=True).astype('float32')
        emb2 = semantic_model.encode([reference_answer], convert_to_numpy=True).astype('float32')
        emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        sim = (emb1 @ emb2.T)[0][0]
        return sim > 0.8
    except:
        return False

# ==================== QA 生成主逻辑 ====================

def generate_qa_for_md_file(md_path: str, filename: str):
    print(f"📄 Processing: {filename}")
    qa_list = []
    simple_count = 0
    complex_count = 0
    open_count = 0
    MAX_PER_TYPE = 2

    with neo4j_driver.session() as session:
        query_relations = "MATCH (a)-[r]->(b) WHERE r.source = $source RETURN a, r, b"
        result = session.run(query_relations, source=filename)
        triples_from_file = []
        entities_from_file = {}

        for record in result:
            a = serialize_neo4j_obj(record["a"])
            r = serialize_neo4j_obj(record["r"])
            b = serialize_neo4j_obj(record["b"])
            a["id"] = record["a"].element_id
            b["id"] = record["b"].element_id
            entities_from_file[a["id"]] = a
            entities_from_file[b["id"]] = b
            triple = {
                "head": a.get("name", ""), "relation": r.get("relation_type", ""),
                "tail": b.get("name", ""), "source": r.get("source", ""),
                "paragraph": r.get("paragraph", "")
            }
            if all(triple.get(k) for k in ["head", "relation", "tail", "source", "paragraph"]):
                triples_from_file.append(triple)

        triples_from_file = deduplicate_triples(triples_from_file)
        if not triples_from_file:
            print(f"  ⛔ No relations found with r.source = '{filename}'")
            return []

        print(f"  🔗 Found {len(triples_from_file)} relations from this file")
        entity_list = list(entities_from_file.values())
        processed_ids = set()
        doc_paragraphs = extract_unique_paragraphs(triples_from_file, max_paragraphs=5)
        context_paragraphs = "\n".join([f"- {p}" for p in doc_paragraphs])
        file_sources = {filename}

        # === Simple QA ===
        for ent in entity_list:
            if simple_count >= MAX_PER_TYPE:
                break
            ent_id = ent["id"]
            if ent_id in processed_ids:
                continue
            processed_ids.add(ent_id)

            subgraph1 = session.execute_read(get_one_hop_subgraph, ent_id)
            if subgraph1["triples"]:
                trimmed_triples1 = trim_subgraph_triples(subgraph1["triples"], max_relations=8, current_source=filename)
                if trimmed_triples1:
                    prompt = f"""
你是一个技术标准专家。请基于以下信息生成一个**简单事实型问题**。
要求：
- 问题必须包含完整主语，禁止使用代词。
- 答案必须明确、完整。
- 仅使用以下数据。
- 输出 JSON: {{"question": "...", "answer": "..."}}

中心实体：
{json.dumps(ent, ensure_ascii=False, indent=2)}

相关文档内容（来自 {filename}）：
{context_paragraphs}

相关关系（结构化）：
{json.dumps(trimmed_triples1, ensure_ascii=False, indent=2)}
"""
                    resp = call_llm(prompt, json_mode=True)
                    if resp and "question" in resp and "answer" in resp:
                        q = resp["question"].strip()
                        a = resp["answer"].strip()
                        if len(q) >= 10 and len(a) >= 10 and not is_duplicate_question(q):
                            supporting_facts = get_relevant_facts_by_question_and_answer(q, a, subgraph1["triples"], top_k=10, qa_type="simple_fact")
                            if not supporting_facts:
                                continue
                            if any(not f.get("source") or not f.get("chunk") or not f["triples"] for f in supporting_facts):
                                continue
                            available_sources = get_sources_from_subgraph(subgraph1["triples"])
                            if not validate_supporting_facts_by_type("simple_fact", supporting_facts, available_sources):
                                continue
                            is_final, final_answer, final_sf = validate_and_fix_answer(q, a, supporting_facts, "simple_fact", available_sources)
                            ref_answer = generate_reference_answer(q, final_sf)
                            if ref_answer and is_answer_consistent(final_answer, ref_answer):
                                qa = {
                                    "id": f"qa_{str(uuid.uuid4())[:8]}",
                                    "hop": 1,
                                    "type": "simple_fact",
                                    "question": q,
                                    "answer": final_answer,
                                    "supporting_facts": final_sf
                                }
                                qa_list.append(qa)
                                simple_count += 1

        # === Complex QA (2~4 hop) ===
        for ent in entity_list:
            if complex_count >= MAX_PER_TYPE:
                break
            ent_id = ent["id"]
            paths = session.execute_read(get_k_hop_paths, ent_id, max_hops=4)
            if not paths:
                continue

            path_chains = []
            all_triples = []
            for path in paths[:5]:
                chain = []
                for i in range(0, len(path), 2):
                    node = path[i][1]
                    chain.append(node.get("name", "Unknown"))
                    if i + 1 < len(path):
                        rel = path[i+1][1]
                        chain.append(rel.get("relation_type", "related_to"))
                        src = node
                        tgt = path[i+2][1] if i+2 < len(path) else {"name": "Unknown"}
                        triple = {
                            "head": src.get("name", ""),
                            "relation": rel.get("relation_type", ""),
                            "tail": tgt.get("name", ""),
                            "source": rel.get("source", filename),
                            "paragraph": rel.get("paragraph", "")
                        }
                        if all(triple.get(k) for k in ["head", "relation", "tail", "source", "paragraph"]):
                            all_triples.append(triple)
                path_chains.append(" → ".join(chain))

            all_triples = deduplicate_triples(all_triples)
            if not path_chains:
                continue

            prompt = f"""
你是一个标准工程师。请基于以下多跳推理链生成一个**复杂逻辑问题及其答案**。
要求：
- 问题主语明确；
- 答案体现完整推理链（2~4跳），主语明确，无代词；
- 仅基于以下内容；
- 输出 JSON: {{"question": "...", "answer": "..."}}

中心实体：
{json.dumps(ent, ensure_ascii=False, indent=2)}

相关文档内容（来自 {filename}）：
{context_paragraphs}

多跳推理链（2~4跳）：
{json.dumps(path_chains, ensure_ascii=False, indent=2)}
"""
            resp = call_llm(prompt, json_mode=True)
            if resp and "question" in resp and "answer" in resp:
                q = resp["question"].strip()
                a = resp["answer"].strip()
                a = re.sub(r'^(根据多跳推理链[^，。]*[，。]?)', '', a).strip()
                a = re.sub(r'^(依据多跳推理链[^，。]*[，。]?)', '', a).strip()
                if len(q) >= 10 and len(a) >= 10 and not is_duplicate_question(q):
                    supporting_facts = get_relevant_facts_by_question_and_answer(q, a, all_triples, top_k=10, qa_type="complex_logic")
                    if not supporting_facts:
                        continue
                    if any(not f.get("source") or not f.get("chunk") or not f["triples"] for f in supporting_facts):
                        continue
                    available_sources = get_sources_from_subgraph(all_triples)
                    if not validate_supporting_facts_by_type("complex_logic", supporting_facts, available_sources):
                        continue
                    is_final, final_answer, final_sf = validate_and_fix_answer(q, a, supporting_facts, "complex_logic", available_sources)
                    ref_answer = generate_reference_answer(q, final_sf)
                    if ref_answer and is_answer_consistent(final_answer, ref_answer):
                        # ✅ 关键修复：Complex QA 必须有至少2个 supporting_facts
                        if len(final_sf) >= 2:
                            estimated_hop = max(2, len(path_chains[0].split(" → ")) // 2)
                            qa = {
                                "id": f"qa_{str(uuid.uuid4())[:8]}",
                                "hop": estimated_hop,
                                "type": "complex_logic",
                                "question": q,
                                "answer": final_answer,
                                "supporting_facts": final_sf
                            }
                            qa_list.append(qa)
                            complex_count += 1

        # === Open-ended QA ===
        paragraphs = list({t["paragraph"] for t in triples_from_file if t["paragraph"].strip()})
        for para in paragraphs[:2]:
            if open_count >= MAX_PER_TYPE:
                break
            similar = find_semantic_similar_chunks(para, top_k=2)
            if similar:
                prompt = f"""
你是一个标准文档分析专家。请基于以下段落及其语义相似内容，生成一个**开放性问题**。
要求：
- 问题有明确主语。
- 答案整合多源信息。
- 不编造。
- 输出 JSON: {{"question": "...", "answer": "..."}}

原始段落（{filename}）：
{para}

相似段落：
{json.dumps(similar, ensure_ascii=False, indent=2)}
"""
                resp = call_llm(prompt, json_mode=True)
                if resp and "question" in resp and "answer" in resp:
                    q = resp["question"].strip()
                    a = resp["answer"].strip()
                    if len(q) >= 10 and len(a) >= 10 and not is_duplicate_question(q):
                        sf = [
                            {"id": i+1, "source": s["source"], "chunk": s["chunk"], "triples": []}
                            for i, s in enumerate(similar)
                            if s.get("source") and s.get("chunk")
                        ]
                        if not sf:
                            continue
                        available_sources = {s["source"] for s in similar if s.get("source")}
                        if not validate_supporting_facts_by_type("open_ended", sf, available_sources):
                            continue
                        is_final, final_answer, final_sf = validate_and_fix_answer(q, a, sf, "open_ended", available_sources)
                        ref_answer = generate_reference_answer(q, final_sf)
                        if ref_answer and is_answer_consistent(final_answer, ref_answer):
                            qa = {
                                "id": f"qa_{str(uuid.uuid4())[:8]}",
                                "hop": 0,
                                "type": "open_ended",
                                "question": q,
                                "answer": final_answer,
                                "supporting_facts": final_sf
                            }
                            qa_list.append(qa)
                            open_count += 1

    return qa_list

# ==================== 主流程 ====================

def main():
    init_dedup_index()
    open(OUTPUT_QA_ALL, "w").close()
    
    md_files = []
    # 临时测试
    # md_files = [("/home/zzm/Project_1/kg-hk/0_mineru_pdf/data_md_final/GB/GBT+34515-2017.md", "GBT+34515-2017.md")]

    # md_files = [("/home/zzm/Project_1/kg-hk/0_mineru_pdf/data_md_final/HB/HB+8748-2023.md", "HB+8748-2023.md")]
    
    # 全量
    for root, _, files in os.walk(MD_ROOT):
        for f in files:
            if f.endswith(".md"):
                md_files.append((os.path.join(root, f), f))

    print(f"🔍 Found {len(md_files)} .md files. Generating QA...")
    
    total_saved = 0

    # 从第 n 个文件开始，第 m 个文件结束
    start_index = 800  # 第 601 个文件的索引
    end_index = 1000  # 第 801 个文件的索引

    md_files = md_files[start_index:end_index]

    for full_path, filename in tqdm(md_files, desc="Processing documents", unit="doc"):
        qa_items = generate_qa_for_md_file(full_path, filename)
        with open(OUTPUT_QA_ALL, "a", encoding="utf-8") as f:
            for qa in qa_items:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        total_saved += len(qa_items)
        tqdm.write(f"   → Saved {len(qa_items)} QA items from {filename}")

    if total_saved == 0:
        print("❌ No valid QA generated.")
        return

    all_qa = []
    with open(OUTPUT_QA_ALL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_qa.append(json.loads(line))

    random.shuffle(all_qa)
    n = len(all_qa)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    splits = {
        "train.json": all_qa[:train_end],
        "val.json": all_qa[train_end:val_end],
        "test.json": all_qa[val_end:],
    }

    for name, data in splits.items():
        with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Total QA saved: {total_saved}")
    for name, data in splits.items():
        print(f"   {name}: {len(data)} items")
    print(f"   Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()