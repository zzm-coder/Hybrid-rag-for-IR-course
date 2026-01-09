# file: frontend_fixed.py
"""修复版前端 - 防止重复调用和状态管理"""
import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import re
matplotlib.use('Agg')

# 设置页面配置
st.set_page_config(
    page_title="航空航天制造混合RAG系统",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 后端API配置
BACKEND_URL = "http://localhost:8885"

# CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.2rem;
        color: #4B5563;
        background-color: #F3F4F6;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 1rem;
        font-weight: 600;
    }
    .confidence-high {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    .confidence-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    .confidence-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    .kg-node {
        fill: #3B82F6 !important;
        stroke: #1D4ED8 !important;
    }
    .kg-edge {
        stroke: #6B7280 !important;
    }
    .citation-link {
        color: #2563EB;
        text-decoration: underline;
        cursor: pointer;
        font-weight: 500;
    }
    .citation-link:hover {
        color: #1D4ED8;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .metric-card {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    .answer-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #1F2937;
        padding: 1rem;
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        white-space: pre-wrap;
    }
    .entity-tag {
        display: inline-block;
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem;
        font-size: 0.875rem;
    }
    .relation-tag {
        display: inline-block;
        background-color: #FCE7F3;
        color: #9D174D;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem;
        font-size: 0.875rem;
    }
    .query-status {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .status-processing {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
    }
    .status-success {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
    }
    .status-error {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
    }
</style>
""", unsafe_allow_html=True)

class FixedMixRAGFrontend:
    """修复版混合RAG系统前端"""
    
    def __init__(self, backend_url: str = BACKEND_URL):
        self.backend_url = backend_url
        
        # 初始化状态
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
        if 'kg_data' not in st.session_state:
            st.session_state.kg_data = None
        if 'last_query' not in st.session_state:
            st.session_state.last_query = None
        if 'last_query_time' not in st.session_state:
            st.session_state.last_query_time = 0
        if 'query_in_progress' not in st.session_state:
            st.session_state.query_in_progress = False
        if 'query_result_received' not in st.session_state:
            st.session_state.query_result_received = False
    
    def run(self):
        """运行前端应用"""
        # 侧边栏
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/airplane.png", width=80)
            st.markdown("### 航空航天制造混合RAG系统")
            st.markdown("**版本:** 2.0 ")
            st.markdown("---")
            
            # 系统状态
            try:
                response = requests.get(f"{self.backend_url}/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    st.success("✅ 后端连接正常")
                    if health.get("rag_system") == "simulation":
                        st.warning("⚠️ 当前运行在模拟模式")
                    else:
                        st.success("🔧 真实RAG系统已激活")
                    
                    # 显示查询管理器状态
                    with st.expander("系统状态详情"):
                        query_manager = health.get("query_manager", {})
                        st.metric("缓存查询数", query_manager.get("cache_size", 0))
                        st.metric("活动查询数", query_manager.get("active_queries", 0))
                else:
                    st.error("❌ 后端连接异常")
            except:
                st.error("❌ 后端连接失败")
            
            # 查询历史
            if st.session_state.query_history:
                st.markdown("#### 查询历史")
                for i, (query, timestamp, result) in enumerate(st.session_state.query_history[-5:]):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        # 使用唯一key确保不会重复触发
                        if st.button(f"📌 {query[:25]}...", key=f"history_btn_{i}_{hash(query)}", use_container_width=True):
                            self._load_query_result(query, result)
                    with col2:
                        st.caption(timestamp)
            
            st.markdown("---")
            st.markdown("#### 操作")
            
            if st.button("🔄 清除所有状态", use_container_width=True):
                self._clear_all_state()
            
            if st.button("🧹 清除当前结果", use_container_width=True):
                self._clear_current_result()
        
        # 主页面
        st.markdown('<div class="main-header">✈️ 航空航天制造混合RAG系统</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; color: #6B7280; margin-bottom: 2rem;">知识图谱 + 向量检索的可解释智能问答系统</div>', unsafe_allow_html=True)
        
        # 查询状态显示
        if st.session_state.query_in_progress:
            st.markdown('<div class="query-status status-processing">⏳ 查询处理中，请稍候...</div>', unsafe_allow_html=True)
        
        # 示例问题按钮
        st.markdown("### 📚 示例问题")
        example_cols = st.columns(4)
        examples = [
            ("HB 8766标准", "HB 8766-2025标准的发布时间以及负责归口管理的单位分别是什么？"),
            ("雷达罩定义", "雷达罩的定义是什么？"),
            ("试验中断处理", "试验实施单位在电性能试验过程中因仪器设备故障导致试验中断时，应按照怎样的逻辑顺序进行后续处理？"),
            ("功率反射计算", "试验人员在计算功率反射时，在什么条件下应优先选择公式(9)而非公式(10)？")
        ]
        
        for i, (title, question) in enumerate(examples):
            with example_cols[i]:
                if st.button(title, key=f"example_{i}", use_container_width=True):
                    if not st.session_state.query_in_progress:
                        self._process_query(question)
                    else:
                        st.warning("当前有查询正在处理中，请稍候")
        
        # 自定义查询输入
        st.markdown("### 🔍 自定义查询")
        
        # 使用表单防止重复提交
        with st.form(key="query_form", clear_on_submit=False):
            query = st.text_area(
                "问题输入",
                height=100,
                placeholder="例如：HB 8766-2025标准的发布时间以及负责归口管理的单位分别是什么？",
                help="输入关于航空航天制造标准、工艺、材料等方面的问题",
                key="query_input"
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                submit_button = st.form_submit_button("🚀 提交查询", type="primary", use_container_width=True,
                                                    disabled=st.session_state.query_in_progress)
            
            if submit_button and query.strip():
                if not st.session_state.query_in_progress:
                    self._process_query(query)
                else:
                    st.warning("当前有查询正在处理中，请稍候")
            elif submit_button and not query.strip():
                st.warning("请输入有效的问题")
        
        # 自动处理待处理的查询（关键修复！）
        if (st.session_state.query_in_progress and 
            st.session_state.last_query and 
            not st.session_state.query_result_received):
            self._actually_process_query(st.session_state.last_query)


        # 显示当前查询结果
        if st.session_state.current_result and st.session_state.query_result_received:
            self._display_result(st.session_state.current_result)
    
    def _clear_all_state(self):
        """清除所有状态"""
        st.session_state.query_history = []
        st.session_state.current_result = None
        st.session_state.kg_data = None
        st.session_state.last_query = None
        st.session_state.last_query_time = 0
        st.session_state.query_in_progress = False
        st.session_state.query_result_received = False
        st.rerun()
    
    def _clear_current_result(self):
        """清除当前结果"""
        st.session_state.current_result = None
        st.session_state.kg_data = None
        st.session_state.query_result_received = False
        st.rerun()
    
    def _process_query(self, query: str):
        """仅设置查询状态，由主循环触发实际处理"""
        current_time = time.time()
        if (st.session_state.last_query == query and
            current_time - st.session_state.last_query_time < 2 and
            st.session_state.query_result_received):
            st.warning("相同查询最近已处理过")
            return

        # 仅设置状态，不调用 API，不 rerun
        st.session_state.query_in_progress = True
        st.session_state.query_result_received = False
        st.session_state.last_query = query
        st.session_state.last_query_time = current_time
        
        try:
            # 调用后端API
            response = self._call_backend(query)
            
            if response:
                # 检查是否为处理中状态
                if response.get("status") == "processing":
                    st.warning("查询正在处理中，请稍候...")
                    # 等待一段时间后重试
                    time.sleep(2)
                    return
                
                # 保存到历史
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.query_history.append((query, timestamp, response))
                st.session_state.current_result = response
                
                # 提取知识图谱数据
                self._extract_kg_data(response)
                
                # 设置状态
                st.session_state.query_result_received = True
                
            else:
                st.error("无法获取响应，请检查后端服务")
        
        except Exception as e:
            st.error(f"查询处理失败: {str(e)}")
        
        finally:
            # 无论成功失败，都标记查询完成
            st.session_state.query_in_progress = False
            st.rerun()
    
    def _actually_process_query(self, query: str):
        """实际执行后端调用（仅由系统自动触发）"""
        # 防止重复进入
        if not st.session_state.query_in_progress:
            return

        try:
            response = self._call_backend(query)
            if response:
                if response.get("status") == "processing":
                    st.warning("查询正在处理中，请稍候...")
                    time.sleep(2)
                    return  # 不修改状态，等待下次轮询
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.query_history.append((query, timestamp, response))
                st.session_state.current_result = response
                self._extract_kg_data(response)
                st.session_state.query_result_received = True
            else:
                st.error("无法获取响应，请检查后端服务")
        except Exception as e:
            st.error(f"查询处理失败: {str(e)}")
        finally:
            # 关键：先关闭状态，再让 Streamlit 自然重绘（不主动 rerun）
            st.session = st.session_state  # 确保状态已更新
            st.session_state.query_in_progress = False
            # 不调用 st.rerun()！


    def _call_backend(self, query: str) -> Optional[Dict]:
        """调用后端API"""
        try:
            # 实际调用后端API
            response = requests.post(
                f"{self.backend_url}/api/query",
                json={"question": query, "include_context": True, "force_refresh": False},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 409:  # 处理中
                return {"status": "processing", "message": "查询正在处理中"}
            else:
                st.error(f"API响应错误: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            st.error(f"无法连接到后端服务: {self.backend_url}")
            st.info("请确保后端服务正在运行")
            return None
        except requests.exceptions.Timeout:
            st.error("请求超时，后端服务响应时间过长")
            return None
        except Exception as e:
            st.error(f"API调用失败: {e}")
            return None
    
    def _load_query_result(self, query: str, result: Dict):
        """加载历史查询结果"""
        st.session_state.current_result = result
        self._extract_kg_data(result)
        st.session_state.query_result_received = True
        st.rerun()
    
    def _extract_kg_data(self, result: Dict):
        """从结果中提取知识图谱数据用于可视化"""
        kg_nodes = set()
        kg_edges = []
        triples_list = []

        # 从 kg_results -> ke_results -> paragrepa -> triples 提取
        retrieval = result.get("retrieval", {})
        kg_results = retrieval.get("kg_results", {})
        ke_results = kg_results.get("ke_results", [])

        for ke in ke_results:
            paragrepa = ke.get("paragrepa", [])
            for para in paragrepa:
                triples = para.get("triples", [])
                for triple in triples:
                    head = triple.get("head", "").strip()
                    relation = triple.get("relation", "").strip()
                    tail = triple.get("tail", "").strip()
                    if not (head and relation and tail):
                        continue

                    triples_list.append({
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "confidence": float(triple.get("confidence", 0.5)),
                        "source": ke.get("source", "unknown.md")
                    })

                    # 添加节点（去重）
                    kg_nodes.add(head)
                    kg_nodes.add(tail)

                    # 添加边
                    kg_edges.append({
                        "source": head,
                        "target": tail,
                        "label": relation,
                        "weight": float(triple.get("confidence", 0.5))
                    })

        # 构建节点列表（含类型和大小）
        nodes = []
        node_types = {}
        for node in kg_nodes:
            # 简单类型推断（可扩展）
            if "标准" in node or re.match(r"[A-Z]{1,3}\d", node):
                node_type = "standard"
            elif "委员会" in node or "单位" in node:
                node_type = "organization"
            elif re.search(r"\d{4}年", node):
                node_type = "date"
            elif "雷达罩" in node or "天线" in node:
                node_type = "component"
            else:
                node_type = "entity"
            node_types[node] = node_type

            nodes.append({
                "id": node,
                "label": node,
                "type": node_type,
                "size": 15 + len([e for e in kg_edges if e["source"] == node or e["target"] == node]) * 3
            })

        st.session_state.kg_data = {
            "nodes": nodes,
            "edges": kg_edges,
            "triples": triples_list
        }
    
    def _display_result(self, result: Dict):
        """显示查询结果"""
        st.markdown("---")
        
        # 显示警告信息（如果有）
        if "warning" in result:
            st.warning(f"⚠️ {result['warning']}")
        if "error" in result:
            st.error(f"❌ {result['error']}")
        
        # 显示缓存命中信息
        if result.get("cache_hit"):
            st.info("✅ 本次查询结果来自缓存")
        
        # 1. 答案展示
        st.markdown('<div class="sub-header">📝 答案</div>', unsafe_allow_html=True)
        
        # 显示答案
        answer = result.get('generation', {}).get('answer', '')
        citations = result.get('generation', {}).get('citations', [])
        
        # 处理答案中的引用标记
        processed_answer = self._process_answer_with_citations(answer, citations)
        st.markdown(f'<div class="answer-text">{processed_answer}</div>', unsafe_allow_html=True)
        
        # 显示处理时间
        perf = result.get('performance', {})
        processing_time = result.get('processing_time', perf.get('total_time', 0))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总处理时间", f"{processing_time:.2f}s")
        with col2:
            st.metric("检索时间", f"{perf.get('retrieval_time', 0):.2f}s")
        with col3:
            st.metric("生成时间", f"{perf.get('generation_time', 0):.2f}s")
        
        # 2. 可视化部分
        st.markdown('<div class="sub-header">🔍 知识图谱与检索分析</div>', unsafe_allow_html=True)
        
        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs(["知识图谱", "检索结果", "路由分析", "原始数据"])
        
        with tab1:
            self._display_knowledge_graph(result)
        
        with tab2:
            self._display_retrieval_results(result)
        
        with tab3:
            self._display_router_analysis(result)
        
        with tab4:
            self._display_raw_data(result)

    def _process_answer_with_citations(self, answer: str, citations: List[str]) -> str:
        """处理答案中的引用标记，添加样式"""
        import re
        
        # 查找所有引用标记 [数字]
        pattern = r'\[(\d+)\]'
        matches = list(re.finditer(pattern, answer))
        
        if not matches:
            return answer
        
        # 创建替换字典
        parts = []
        last_end = 0
        
        for match in matches:
            # 添加之前的部分
            parts.append(answer[last_end:match.start()])
            
            # 添加引用标记
            citation_num = match.group(1)
            parts.append(f'<span class="citation-link" title="引用{citation_num}">[{citation_num}]</span>')
            
            last_end = match.end()
        
        # 添加剩余部分
        parts.append(answer[last_end:])
        
        return "".join(parts)
    
    def _display_knowledge_graph(self, result: Dict):
        """显示知识图谱可视化"""
        if not st.session_state.kg_data:
            st.info("未找到知识图谱数据")
            return
        
        kg_data = st.session_state.kg_data
        
        # 创建两个列：左侧显示图，右侧显示详情
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 使用Plotly创建交互式知识图谱
            self._create_knowledge_graph_plotly(kg_data)
        
        with col2:
            # 显示三元组详情
            st.markdown('<div class="section-header">📋 知识三元组</div>', unsafe_allow_html=True)
            
            triples = kg_data.get("triples", [])
            if triples:
                for i, triple in enumerate(triples[:10]):  # 只显示前10个
                    with st.expander(f"三元组 {i+1}: {triple['head'][:15]}..."):
                        st.markdown(f"**头实体:** `{triple['head']}`")
                        st.markdown(f"**关系:** `{triple['relation']}`")
                        st.markdown(f"**尾实体:** `{triple['tail']}`")
                        st.markdown(f"**置信度:** {triple.get('confidence', 0.0):.2f}")
                        if triple.get('source'):
                            st.markdown(f"**来源:** {triple['source']}")
            else:
                st.info("无三元组数据")
            
            # 显示统计信息
            st.markdown('<div class="section-header">📊 统计信息</div>', unsafe_allow_html=True)
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("节点数", len(kg_data.get("nodes", [])))
            with col_stat2:
                st.metric("关系数", len(kg_data.get("edges", [])))
    
    def _create_knowledge_graph_plotly(self, kg_data: Dict):
        """使用Plotly创建知识图谱"""
        nodes = kg_data.get("nodes", [])
        edges = kg_data.get("edges", [])
        
        if not nodes or not edges:
            st.info("知识图谱数据不足，无法可视化")
            return
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        for node in nodes:
            G.add_node(node["id"], label=node["label"], type=node.get("type", "entity"), size=node.get("size", 15))
        
        # 添加边
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], label=edge["label"], weight=edge.get("weight", 0.5))
        
        # 使用spring布局
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # 创建边轨迹
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # 边的中点用于显示标签
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            edge_text.append((mid_x, mid_y, edge[2].get('label', '')))
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 创建边标签轨迹
        edge_label_x = []
        edge_label_y = []
        edge_label_text = []
        
        for mid_x, mid_y, label in edge_text:
            edge_label_x.append(mid_x)
            edge_label_y.append(mid_y)
            edge_label_text.append(label)
        
        edge_label_trace = go.Scatter(
            x=edge_label_x, y=edge_label_y,
            mode='text',
            text=edge_label_text,
            textposition='middle center',
            textfont=dict(size=10, color='#555'),
            hoverinfo='none'
        )
        
        # 创建节点轨迹
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        # 节点类型颜色映射
        type_colors = {
            'standard': '#3B82F6',
            'organization': '#10B981',
            'date': '#F59E0B',
            'component': '#8B5CF6',
            'process': '#EF4444',
            'entity': '#6B7280',
            '领域': '#3B82F6',
            '概念': '#10B981',
            '要求': '#F59E0B',
            'default': '#6B7280'
        }
        
        for node in nodes:
            x, y = pos[node["id"]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node["label"])
            node_size.append(node.get("size", 15))
            
            node_type = node.get("type", "entity")
            node_color.append(type_colors.get(node_type, type_colors["default"]))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace],
                       layout=go.Layout(
                           title='知识图谱可视化',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='rgba(0,0,0,0)'
                       ))
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # 图例说明
        with st.expander("📖 图例说明"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**节点颜色:**")
                for type_name, color in list(type_colors.items())[:4]:
                    st.markdown(f'<span style="color:{color}">●</span> {type_name}', unsafe_allow_html=True)
            with col2:
                st.markdown("**节点大小:**")
                st.markdown("● 大小表示节点重要性")
                st.markdown("**连线:**")
                st.markdown("─ 表示实体间关系")
    
    def _display_retrieval_results(self, result: Dict):
        """显示检索结果"""
        retrieval = result.get('retrieval', {})
        
        # 创建两个列：左侧显示KG结果，右侧显示向量结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">🧠 知识图谱检索</div>', unsafe_allow_html=True)
            
            kg_results = retrieval.get('kg_results', {})
            ke_results = kg_results.get('ke_results', [])
            
            if ke_results:
                for i, ke_result in enumerate(ke_results[:3]):  # 只显示前3个来源
                    source = ke_result.get('source', '未知来源')
                    # 构造 .md 文件下载路径
                    md_path = self._get_md_file_path(source)
                    if md_path and md_path.exists():
                        # 提供下载链接（使用 st.download_button 需要读取文件）
                        with open(md_path, "rb") as f:
                            md_content = f.read()
                        st.download_button(
                            label=f"📥 下载 {source}",
                            data=md_content,
                            file_name=source,
                            mime="text/markdown",
                            key=f"download_{i}"
                        )
                    else:
                        st.warning(f"⚠️ 未找到文件: {source}")
                        paragraphs = ke_result.get('paragrepa', [])
                        for para in paragraphs:  # 每个来源显示前2个段落
                            st.markdown(f"**段落:** {para.get('text', '')}...")
                            triples = para.get('triples', [])
                            if triples:
                                st.markdown("**提取的三元组:**")
                                for triple in triples[:3]:
                                    st.code(f"{triple.get('head', '')} → {triple.get('relation', '')} → {triple.get('tail', '')}")
            else:
                st.info("未检索到知识图谱信息")
            
            # KG统计
            st.markdown('<div class="section-header">📊 KG统计</div>', unsafe_allow_html=True)
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                total_triples = sum(len(para.get('triples', [])) for ke in ke_results for para in ke.get('paragrepa', []))
                st.metric("三元组数", total_triples)
            with col_stat2:
                st.metric("来源数", len(ke_results))
        
        with col2:
            st.markdown('<div class="section-header">🔤 向量检索</div>', unsafe_allow_html=True)
            
            vector_results = retrieval.get('vector_results', [])
            
            if vector_results:
                # 按相似度排序
                vector_results_sorted = sorted(vector_results, 
                                             key=lambda x: x.get('similarity_score', 0), 
                                             reverse=True)
                
                for i, chunk in enumerate(vector_results_sorted[:5]):  # 只显示前5个
                    with st.expander(f"文档 {i+1}: {chunk.get('source', '未知')} (相似度: {chunk.get('similarity_score', 0):.3f})"):
                        st.markdown(f"**内容:**")
                        chunk_text = chunk.get('chunk_text', '')
                        if len(chunk_text) > 300:
                            st.text(chunk_text[:300] + "...")
                        else:
                            st.text(chunk_text)
                        
                        metadata = chunk.get('metadata', {})
                        if metadata:
                            st.markdown(f"**元数据:**")
                            for key, value in list(metadata.items())[:3]:
                                st.markdown(f"- {key}: {value}")
            else:
                st.info("未检索到向量文档")
            
            # 向量检索统计
            st.markdown('<div class="section-header">📊 向量检索统计</div>', unsafe_allow_html=True)
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("检索文档数", len(vector_results))
            with col_stat2:
                if vector_results:
                    best_score = max(v.get('similarity_score', 0) for v in vector_results)
                    st.metric("最佳相似度", f"{best_score:.3f}")
                else:
                    st.metric("最佳相似度", 0.0)
    
    def _get_md_file_path(self, filename: str) -> Optional[Path]:
        """根据文件名在 GB/HB 目录下查找 .md 文件"""
        gb_dir = Path("/home/zzm/Project_1/kg-hk/0_mineru_pdf/data_md_final/GB")
        hb_dir = Path("/home/zzm/Project_1/kg-hk/0_mineru_pdf/data_md_final/HB")

        for base_dir in [gb_dir, hb_dir]:
            if base_dir.exists():
                candidate = base_dir / filename
                if candidate.is_file():
                    return candidate
        return None


    def _display_router_analysis(self, result: Dict):
        """显示路由分析"""
        router = result.get('router_analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("问题类型", router.get('question_type', '未知'))
            st.caption(f"类型ID: {router.get('type_id', '未知')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 类型说明
            type_id = router.get('type_id', 1)
            type_descriptions = {
                1: "简单事实型：查询具体的数值、定义、标准编号等可以直接从文档中查找的简单事实信息",
                2: "复杂逻辑型：需要推理、比较、原因分析、工艺参数调整等复杂逻辑处理的问题",
                3: "开放语义型：需要论述、总结、概述、讨论、综合分析等开放性内容的问题"
            }
            st.info(type_descriptions.get(type_id, "未知类型"))
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            entities = router.get('entities', [])
            st.metric("识别实体", len(entities))
            if entities:
                st.markdown("**实体列表:**")
                for entity in entities[:5]:
                    st.markdown(f'<span class="entity-tag">{entity}</span>', unsafe_allow_html=True)
                if len(entities) > 5:
                    st.caption(f"...等 {len(entities)} 个实体")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 用户意图
        st.markdown('<div class="section-header">🎯 用户意图分析</div>', unsafe_allow_html=True)
        st.markdown(f'**核心意图:** {router.get("intent", "未知")}')
        
        # 置信度说明
        st.markdown('<div class="section-header">📈 置信度说明</div>', unsafe_allow_html=True)
        col_conf1, col_conf2, col_conf3 = st.columns(3)
        
        with col_conf1:
            st.markdown('<div class="confidence-high">高置信度</div>', unsafe_allow_html=True)
            st.caption("基于知识图谱直接查询，可靠性高")
        
        with col_conf2:
            st.markdown('<div class="confidence-medium">中置信度</div>', unsafe_allow_html=True)
            st.caption("基于向量检索推理，需验证")
        
        with col_conf3:
            st.markdown('<div class="confidence-low">低置信度</div>', unsafe_allow_html=True)
            st.caption("基于外部知识推断，仅供参考")
    
    def _display_raw_data(self, result: Dict):
        """显示原始数据"""
        with st.expander("📄 查看完整原始数据"):
            st.json(result)

def main():
    """主函数"""
    # 应用标题
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1E3A8A;">✈️ 航空航天制造混合RAG系统</h1>
        <p style="color: #6B7280; font-size: 1.1rem;">
        结合知识图谱推理与向量语义检索的可解释智能问答系统
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化应用
    app = FixedMixRAGFrontend()
    app.run()

if __name__ == "__main__":
    main()