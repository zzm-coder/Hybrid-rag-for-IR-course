# Hybrid-RAG-v1 使用说明

本文档介绍基于航天标准 PDF 的混合 RAG（Hybrid-RAG）方法的完整处理流程（0–5 步），包括数据抽取、知识图谱构建、文本分片与向量库构建、图谱与向量库映射、QA 数据集生成、以及 Hybrid-RAG 的评估与前端可视化部署。data源pdf文件链接：https://pan.quark.cn/s/98bcf01d6197
提取码：4Typ

## 概览

整个项目分五个主要阶段：

0. 使用 `mineru` 将航天标准 PDF 转换为 Markdown（MD）文件并做预处理。
1. 编写并运行抽取 Prompt，对 MD 提取实体与关系，产出三元组和实体表，用于构建知识图谱（KG）。
2. 将 MD 切分为段落/chunk，生成段落级文本并构建向量索引（FAISS/embeddings）。
3. 在 KG 与向量库之间建立双向映射（实体↔段落/文档），并将两者合并用于上下文检索。
4. 基于已构建的 KG 与向量库生成 QA 数据集（支持事实标注），用于训练/评估 Retriever / Reranker / RAG 系统。
5. 使用 Hybrid-RAG 方法进行评估与测试（包含路由分类、按 QA 类型选择检索/推理策略），并最终构建前端可视化展示。

## 目录结构（关键）

- `0_mineru_pdf/`：mineru 相关脚本与原始 PDF/MD 数据
- `1_extract_data/`：MD 抽取脚本与 prompt（实体/关系抽取）
- `2_kg_construction/`：知识图谱构建脚本、三元组缓存与统计
- `3_QA_creation/`：QA 数据集生成脚本与 split 数据
- `4_RAG_method/Hybrid-RAG-v1/`：Hybrid-RAG 实现、评估器、前端后端示例与结果目录
- `5_Hybrid-RAG-show/`：前端可视化与 API 示例

## 环境与依赖

建议在 Conda 环境中安装依赖：

```bash
conda create -n Hybrid-rag python=3.9 -y
conda activate Hybrid-rag
pip install -r 5_Hybrid-RAG-show/requirements.txt
# 若使用训练脚本或 sentence-transformers 相关功能
pip install sentence-transformers faiss-cpu transformers torch neo4j
```

根据硬件（GPU/CPU）可替换 `faiss-cpu` 为 `faiss-gpu`，并选择合适的 `torch` 版本。

## 步骤详述

### 0. PDF → MD（mineru）

使用 `0_mineru_pdf` 下的脚本将原始航天标准 PDF 批量转换为 MD：

```bash
cd 0_mineru_pdf
python total_data.py    # 示例：批处理 PDF → MD（项目中已包含多种处理脚本）
```

输出：`data_GB_md/`、`data_HB_md/` 等按标准编号组织的 MD 文件夹。

### 1. 从 MD 提取实体与关系 → KG

在 `1_extract_data/` 中定义了抽取 prompt 与抽取脚本（`1_extract_data_chunk_final.py`、`prompt.py` 等）。流程：

- 准备抽取 prompt（模板位于 `prompt.py`）。
- 运行抽取脚本，得到实体、关系和三元组（并保存到 `2_kg_construction/kg_cache/`）。

示例：

```bash
cd 1_extract_data
python 1_extract_data_chunk_final.py --input ../0_mineru_pdf/data_md_final --output ../2_kg_construction/kg_cache
```

产物：实体表（nodes）、关系表（edges）、三元组 JSON/CSV。

### 2. 文本分片与向量索引构建

脚本 `2_md_chunk.py`（及 `2_kg_construction` 中的工具）会把 MD 切分为段落/chunk，并为每个 chunk 计算向量（使用 sentence-transformers 或其他 embedding 提供者）。

示例：

```bash
cd 2_kg_construction
python 2_md_chunk.py --md-dir ../0_mineru_pdf/data_md_final --out ../2_kg_construction/kg_vector_db
```

输出：`kg_vector_db/`，包含 `embeddings.npy`、chunks 元信息等。

### 3. KG ↔ 向量库 双向映射

在 `2_kg_construction` 中的脚本会建立实体与段落/文档的映射（例如将实体指向包含该实体的段落 ID，段落也记录所在实体）。映射用于：

- 当 KG 查询找到实体时，能快速定位到对应的段落上下文（用于 LLM 提供证据）；
- 当向量检索命中段落时，也可回溯到相关 KG 实体以增强上下文（补充结构化信息）。

结果保存在 `kg_cache/` 与 `kg_vector_db/` 中，格式为 JSON（包含 `entity_id -> [chunk_ids]` 和 `chunk_id -> [entity_ids]`）。

### 4. QA 数据集构建

在 `3_QA_creation/` 中提供脚本将 KG 与向量库内容组装为 QA 数据集：

- 将支持事实（supporting_facts）标注为 `source`（md 文件名）与 `chunk`（段落文本或 chunk id）。
- 生成 `train.json` / `dev.json` / `test.json`，每条包含 `id, question, answer, supporting_facts, type` 等字段。

示例：

```bash
cd 3_QA_creation
python 3_QA_creat_new.py --kg ../2_kg_construction/kg_cache --chunks ../2_kg_construction/kg_vector_db --out ./3_QA_data/split_datasets
```

### 5. Hybrid-RAG：路由、检索、生成与评估

`4_RAG_method/Hybrid-RAG-v1/` 包含 Hybrid-RAG 的实现：

- `hybrid_rag_system.py`：系统主流程，负责路由分类（将问题分到不同 QA 类型）、并基于类型选择检索策略（仅 KG / 仅向量 / KG+向量）和重排序。
- `vector_retriever.py`：向量检索与重排序（FAISS + CrossEncoder 可选）。
- `kg_retriever.py`：基于 KG 的检索（实体到段落映射）和结构化证据抽取。
- `llm_generator.py`：LLM 调用封装（生成答案与证据解析）。
- `evaluator.py`：评估模块，计算文档级与段落级指标、引用准确率与生成质量，并输出按类型报告。
- `train_retriever.py` / `train_reranker.py`：提供微调检索器与重排序器的脚本（基于 sentence-transformers / CrossEncoder）。

运行示例（快速评估）：

```bash
cd 4_RAG_method/Hybrid-RAG-v1
python main.py    # 启动交互式运行或选择评估/训练/推理模式
# 或直接运行评估脚本（示例）
python -c "from evaluator import RAGEvaluator; from hybrid_rag_system import HybridRAGSystem; re=RAGEvaluator('3_QA_data/split_datasets/test.json'); sys=HybridRAGSystem(); print(re.evaluate_system(sys, sample_size=50))"
```

注意：`main.py` 提供选项用于选择 ablation（KG-only / Vector-only / Hybrid）与是否运行训练流程。

## 前端可视化

`5_Hybrid-RAG-show/` 包含前端与后端示例：

- `backend_api.py`：后端 API 示例，用于接收问题、调用 `HybridRAGSystem` 并返回答案与证据。
- `frontend_app.py`：简单演示页面，用于展示答案、证据片段、KG 可视化（节点/边）与检索到的段落。

部署方式（示例）：

```bash
# 启动后端（示例）
python 5_Hybrid-RAG-show/backend_api.py
# 在另一终端启动前端（或直接打开静态 HTML）
python 5_Hybrid-RAG-show/frontend_app.py
```

## 配置与常见路径

- QA split 数据：`3_QA_creation/3_QA_data/split_datasets/`（`train.json, dev.json, test.json`）
- 向量库：`2_kg_construction/kg_vector_db/`（`embeddings.npy`, `chunks.json`）
- KG 缓存：`2_kg_construction/kg_cache/`
- RAG 结果：`4_RAG_method/Hybrid-RAG-v1/results/` 和 `evaluation_results/`

配置文件或常量位于各模块顶部（例如 `config.py` / `data_types.py`），运行前请确认路径指向正确的目录。

## 运行顺序建议（快速上手）

1. 准备环境并安装依赖。建议先运行小规模样本确认流程。
2. 执行 `0_mineru_pdf` 的 PDF→MD 转换。
3. 执行 `1_extract_data` 的实体关系抽取，检查 `kg_cache`。
4. 执行 `2_md_chunk.py` 切分与向量化，确认 `kg_vector_db`。
5. 运行 `3_QA_creat_new.py` 生成 QA splits。
6. 在 `4_RAG_method/Hybrid-RAG-v1` 运行 `main.py` 或调用 `RAGEvaluator` 进行评估。
7. 启动 `5_Hybrid-RAG-show` 中的后端/前端以做展示。

## 调试提示与注意事项

- 若向量检索返回空结果或命中率极低，检查 `embeddings.npy` 是否与 `chunks.json` 中的顺序一致（索引映射问题最常见）。
- 训练脚本需要 `sentence-transformers` 与 `torch`，确保对应版本安装并与 CUDA 兼容（如使用 GPU）。
- LLM 评估/生成函数会依赖外部 API 或本地模型服务，若不可用，评估器会回退到 SequenceMatcher 等简易相似度度量。
- KG 与向量映射建议保留双向索引（entity->chunk_ids, chunk_id->entity_ids），便于从任一端快速回溯证据。

## 未来改进方向

- 增加硬负采样（hard negative mining）来提升 reranker 训练效果。
- 将 KG 与向量检索联合训练（多任务或对比学习）以提升跨模态召回。
- 提供更强的证据可视化（按段落高亮、文档跳转、KG 子图展示）。
