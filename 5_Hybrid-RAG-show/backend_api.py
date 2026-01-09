# file: backend_api.py
"""混合RAG系统后端API"""
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent  # /home/zzm/Project_1/kg-hk
rag_method_dir = project_root / "4_RAG_method" / "Mix-RAG-v1" # RAG方法目录

# 确保路径存在
if str(rag_method_dir) not in sys.path:
    sys.path.insert(0, str(rag_method_dir))

# 导入RAG系统模块
try:
    sys.path.insert(0, "/home/zzm/Project_1/kg-hk/4_RAG_method/Mix-RAG-v1")
    from data_types import SystemConfig
    from hybrid_rag_system import HybridRAGSystem
    print("✅ 使用绝对路径导入成功")
except ImportError as e2:
    print(f"❌ 绝对路径导入也失败: {e2}")
    raise

import time
import hashlib
import threading
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
rag_system: Optional[HybridRAGSystem] = None
system_config: Optional[SystemConfig] = None

# 查询缓存和状态管理
class QueryManager:
    """查询管理器，防止重复处理和提供状态跟踪"""
    
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="RAG_Worker")
        self.active_queries = set()  # 正在处理的查询
        self.max_cache_size = 100
        self.cache_ttl = 300  # 5分钟
    
    def get_query_hash(self, question: str) -> str:
        """获取查询的哈希值"""
        return hashlib.md5(question.encode('utf-8')).hexdigest()
    
    def is_query_active(self, query_hash: str) -> bool:
        """检查查询是否正在处理中"""
        with self.lock:
            return query_hash in self.active_queries
    
    def mark_query_active(self, query_hash: str):
        """标记查询为处理中"""
        with self.lock:
            self.active_queries.add(query_hash)
    
    def mark_query_inactive(self, query_hash: str):
        """标记查询为处理完成"""
        with self.lock:
            if query_hash in self.active_queries:
                self.active_queries.remove(query_hash)
    
    def get_cached_result(self, query_hash: str) -> Optional[Dict]:
        """获取缓存的查询结果"""
        with self.lock:
            if query_hash in self.cache:
                cached_data = self.cache[query_hash]
                # 检查是否过期
                if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl):
                    logger.info(f"缓存命中: {query_hash[:8]}")
                    return cached_data['result']
                else:
                    # 过期删除
                    del self.cache[query_hash]
            return None
    
    def cache_result(self, query_hash: str, result: Dict):
        """缓存查询结果"""
        with self.lock:
            # 清理过期缓存
            current_time = datetime.now()
            expired_hashes = []
            for qh, data in self.cache.items():
                if current_time - data['timestamp'] > timedelta(seconds=self.cache_ttl):
                    expired_hashes.append(qh)
            
            for qh in expired_hashes:
                del self.cache[qh]
            
            # 如果缓存满了，删除最旧的
            if len(self.cache) >= self.max_cache_size:
                oldest_hash = next(iter(self.cache))
                del self.cache[oldest_hash]
            
            # 存储新结果
            self.cache[query_hash] = {
                'result': result,
                'timestamp': current_time
            }

query_manager = QueryManager()

# 数据模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    include_context: bool = True
    force_refresh: bool = False  # 强制刷新缓存

# 使用 lifespan 上下文管理器替代 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理：启动和关闭"""
    # 启动时初始化
    global rag_system, system_config
    try:
        logger.info(f"正在初始化RAG系统...")
        
        # 尝试加载配置
        try:
            system_config = SystemConfig()
            logger.info("✅ 系统配置加载成功")
            logger.info(f"Neo4j URI: {system_config.neo4j_uri}")
            logger.info(f"向量数据库路径: {system_config.vector_db_path}")
            logger.info(f"LLM模型: {system_config.llm_model}")
        except Exception as e:
            logger.error(f"❌ 加载系统配置失败: {e}")
            logger.warning("将继续使用模拟模式运行")
        
        # 尝试初始化RAG系统
        try:
            if system_config:
                rag_system = HybridRAGSystem(system_config)
                logger.info("✅ 混合RAG系统初始化成功")
            else:
                logger.warning("⚠️ 使用模拟模式运行")
        except Exception as e:
            logger.error(f"❌ RAG系统初始化失败: {e}")
            logger.warning("⚠️ 使用模拟模式运行，实际RAG系统不可用")
        
        logger.info("🚀 混合RAG系统API服务启动完成")
        yield  # 应用运行期间
        
    except Exception as e:
        logger.error(f"🔥 系统初始化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        yield  # 即使失败也继续运行
    
    finally:
        # 关闭时清理资源
        if rag_system:
            try:
                rag_system.close()
                logger.info("✅ 系统资源已清理")
            except Exception as e:
                logger.error(f"清理资源时出错: {e}")

# 创建FastAPI应用，使用 lifespan
app = FastAPI(
    title="航空航天制造混合RAG系统API",
    description="结合知识图谱与向量检索的可解释智能问答系统",
    version="2.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_process_query(question: str, timeout: int = 30) -> Dict:
    """安全处理查询，避免无限循环"""
    import threading
    import queue
    
    result_queue = queue.Queue()
    
    def process():
        try:
            result = rag_system.process_query(question)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", str(e)))
    
    # 启动线程处理查询
    thread = threading.Thread(target=process)
    thread.daemon = True
    thread.start()
    
    # 等待结果，超时返回模拟数据
    try:
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            logger.warning(f"查询处理超时: {question[:50]}...")
            raise TimeoutError(f"查询处理超时（{timeout}秒）")
        
        status, data = result_queue.get(timeout=1)
        
        if status == "success":
            return data
        else:
            raise Exception(f"处理失败: {data}")
            
    except queue.Empty:
        logger.error("结果队列为空")
        raise Exception("处理结果为空")
    except TimeoutError as e:
        raise e
    except Exception as e:
        raise e

def get_mock_response(query: str) -> Dict:
    """获取模拟响应数据"""
    from datetime import datetime
    
    # 根据查询内容生成不同类型的响应
    if "HB 8766" in query or "8766" in query:
        return {
            "question": query,
            "router_analysis": {
                "type_id": 1,
                "question_type": "simple_fact",
                "entities": ["HB 8766-2025"],
                "intent": "查询标准发布信息",
                "metadata": {"source": "llm_analysis"}
            },
            "retrieval": {
                "kg_results": {
                    "ke_results": [
                        {
                            "source": "HB 8766-2025.md",
                            "paragrepa": [
                                {
                                    "text": "本标准于2025年1月15日发布，自2025年7月1日起实施。本标准由航空航天标准化委员会提出并归口。",
                                    "triples": [
                                        {"head": "HB 8766-2025", "relation": "发布时间", "tail": "2025年1月15日", "confidence": 0.95},
                                        {"head": "HB 8766-2025", "relation": "实施时间", "tail": "2025年7月1日", "confidence": 0.95},
                                        {"head": "HB 8766-2025", "relation": "归口单位", "tail": "航空航天标准化委员会", "confidence": 0.95}
                                    ]
                                }
                            ]
                        }
                    ],
                    "entities": ["HB 8766-2025", "航空航天标准化委员会", "2025年1月15日", "2025年7月1日"],
                    "query_time": 0.12
                },
                "vector_results": [
                    {
                        "chunk_id": "123",
                        "source": "HB 8766-2025.md",
                        "chunk_text": "本标准规定了雷达罩电性能试验的要求、试验方法、试验设备、试验程序和试验报告等内容。适用于各类飞行器雷达罩的电性能试验。",
                        "similarity_score": 0.85,
                        "retrieval_source": "vector",
                        "metadata": {"file_name": "HB 8766-2025.md", "section": "1.范围"}
                    }
                ],
                "reranked_results": [],
                "retrieval_time": 0.35
            },
            "generation": {
                "answer": "HB 8766-2025标准于2025年1月15日发布[1]，自2025年7月1日起实施[1]。负责归口管理的单位是航空航天标准化委员会[1]。",
                "citations": ["HB 8766-2025.md"],
                "citation_extracted_files": ["HB 8766-2025.md"],
                "generation_time": 2.34,
                "raw_response": "【答案】HB 8766-2025标准于2025年1月15日发布[1]，自2025年7月1日起实施[1]。负责归口管理的单位是航空航天标准化委员会[1]。\n【证据】1. HB 8766-2025.md (本标准于2025年1月15日发布...)"
            },
            "performance": {
                "total_time": 2.89,
                "retrieval_time": 0.35,
                "generation_time": 2.34
            },
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "kg_uri": "bolt://192.168.1.104:7687",
                "vector_db": "/home/zzm/Project_1/kg-hk/2_kg_construction/kg_vector_db",
                "llm_model": "/hdd1/checkpoints/Qwen/Qwen3-32B"
            }
        }
    elif "雷达罩" in query and "定义" in query:
        return {
            "question": query,
            "router_analysis": {
                "type_id": 1,
                "question_type": "simple_fact",
                "entities": ["雷达罩"],
                "intent": "查询定义",
                "metadata": {"source": "llm_analysis"}
            },
            "retrieval": {
                "kg_results": {
                    "ke_results": [
                        {
                            "source": "航空航天术语标准.md",
                            "paragrepa": [
                                {
                                    "text": "雷达罩是安装在雷达天线前方的保护罩，用于保护天线免受环境影响，同时允许电磁波通过。",
                                    "triples": [
                                        {"head": "雷达罩", "relation": "定义", "tail": "安装在雷达天线前方的保护罩", "confidence": 0.92},
                                        {"head": "雷达罩", "relation": "功能", "tail": "保护天线免受环境影响", "confidence": 0.90},
                                        {"head": "雷达罩", "relation": "特性", "tail": "允许电磁波通过", "confidence": 0.95}
                                    ]
                                }
                            ]
                        }
                    ],
                    "entities": ["雷达罩", "雷达天线", "电磁波"],
                    "query_time": 0.10
                },
                "vector_results": [
                    {
                        "chunk_id": "456",
                        "source": "雷达罩设计规范.md",
                        "chunk_text": "雷达罩（Radome）是雷达系统的关键部件，通常由复合材料制成，具有良好的透波性能和结构强度。",
                        "similarity_score": 0.82,
                        "retrieval_source": "vector",
                        "metadata": {"file_name": "雷达罩设计规范.md", "section": "1.定义"}
                    }
                ],
                "reranked_results": [],
                "retrieval_time": 0.28
            },
            "generation": {
                "answer": "雷达罩是安装在雷达天线前方的保护罩[1]，用于保护天线免受环境影响[1]，同时允许电磁波通过[1]。它通常由复合材料制成，具有良好的透波性能和结构强度[2]。",
                "citations": ["航空航天术语标准.md", "雷达罩设计规范.md"],
                "citation_extracted_files": ["航空航天术语标准.md", "雷达罩设计规范.md"],
                "generation_time": 1.98,
                "raw_response": "【答案】雷达罩是安装在雷达天线前方的保护罩[1]，用于保护天线免受环境影响[1]，同时允许电磁波通过[1]。它通常由复合材料制成，具有良好的透波性能和结构强度[2]。\n【证据】1. 航空航天术语标准.md (雷达罩是安装在雷达天线前方的保护罩...)\n2. 雷达罩设计规范.md (雷达罩通常由复合材料制成...)"
            },
            "performance": {
                "total_time": 2.36,
                "retrieval_time": 0.28,
                "generation_time": 1.98
            },
            "timestamp": datetime.now().isoformat()
        }
    else:
        # 通用响应
        return {
            "question": query,
            "router_analysis": {
                "type_id": 1,
                "question_type": "simple_fact",
                "entities": ["航空航天", "标准"],
                "intent": "查询信息",
                "metadata": {"source": "llm_analysis"}
            },
            "retrieval": {
                "kg_results": {
                    "ke_results": [],
                    "entities": [],
                    "query_time": 0.05
                },
                "vector_results": [
                    {
                        "chunk_id": "789",
                        "source": "航空航天标准总览.md",
                        "chunk_text": "航空航天制造涉及大量国家标准（GB）、行业标准（HB）和企业标准，确保产品质量和安全。",
                        "similarity_score": 0.75,
                        "retrieval_source": "vector",
                        "metadata": {"file_name": "航空航天标准总览.md", "section": "1.概述"}
                    }
                ],
                "reranked_results": [],
                "retrieval_time": 0.22
            },
            "generation": {
                "answer": "根据您的问题，我找到了一些相关信息：航空航天制造涉及大量国家标准（GB）、行业标准（HB）和企业标准，这些标准确保了产品质量和安全[1]。如果您有具体标准编号或问题，请提供更多细节。",
                "citations": ["航空航天标准总览.md"],
                "citation_extracted_files": ["航空航天标准总览.md"],
                "generation_time": 1.75,
                "raw_response": "【答案】根据您的问题，我找到了一些相关信息：航空航天制造涉及大量国家标准（GB）、行业标准（HB）和企业标准，这些标准确保了产品质量和安全[1]。如果您有具体标准编号或问题，请提供更多细节。\n【证据】1. 航空航天标准总览.md (航空航天制造涉及大量国家标准...)"
            },
            "performance": {
                "total_time": 2.02,
                "retrieval_time": 0.22,
                "generation_time": 1.75
            },
            "timestamp": datetime.now().isoformat()
        }

# API路由
@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "航空航天制造混合RAG系统",
        "version": "2.0",
        "status": "running" if rag_system else "simulation",
        "timestamp": datetime.now().isoformat(),
        "mode": "真实RAG模式" if rag_system else "模拟模式"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_system": "active" if rag_system else "simulation",
        "query_manager": {
            "cache_size": len(query_manager.cache),
            "active_queries": len(query_manager.active_queries)
        }
    }

@app.post("/api/query")
async def process_query(request: QueryRequest):
    """处理单个查询"""
    start_time = time.time()
    query_hash = query_manager.get_query_hash(request.question)
    
    # 检查是否正在处理相同查询
    if query_manager.is_query_active(query_hash):
        logger.warning(f"查询已在处理中: {request.question[:50]}...")
        return JSONResponse(
            status_code=409,
            content={
                "status": "processing",
                "message": "相同的查询正在处理中，请稍候",
                "query_hash": query_hash,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # 检查缓存（除非强制刷新）
    if not request.force_refresh:
        cached_result = query_manager.get_cached_result(query_hash)
        if cached_result:
            processing_time = time.time() - start_time
            cached_result["processing_time"] = processing_time
            cached_result["cache_hit"] = True
            return cached_result
    
    try:
        # 标记查询为处理中
        query_manager.mark_query_active(query_hash)
        logger.info(f"开始处理查询: {request.question[:50]}...")
        
        result = None
        
        # 如果RAG系统已初始化且可用，使用真实处理
        if rag_system:
            try:
                # 使用安全处理，避免无限循环
                result = safe_process_query(request.question, timeout=30)
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                result["cache_hit"] = False
                logger.info(f"真实查询处理完成，耗时: {processing_time:.2f}秒")
                
                # 缓存结果
                query_manager.cache_result(query_hash, result)
                
            except TimeoutError as e:
                logger.warning(f"真实处理超时，使用模拟数据: {e}")
                # 超时后使用模拟数据
                result = get_mock_response(request.question)
                result["warning"] = "真实处理超时，已返回模拟数据"
                result["processing_time"] = time.time() - start_time
                result["cache_hit"] = False
                
            except Exception as e:
                logger.error(f"真实处理失败: {e}")
                # 失败后使用模拟数据
                result = get_mock_response(request.question)
                result["error"] = str(e)
                result["warning"] = "真实处理失败，已返回模拟数据"
                result["processing_time"] = time.time() - start_time
                result["cache_hit"] = False
        else:
            # 使用模拟数据
            result = get_mock_response(request.question)
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["mode"] = "simulation"
            result["cache_hit"] = False
            logger.info(f"模拟查询处理完成，耗时: {processing_time:.2f}秒")
        
        # 标记查询完成
        query_manager.mark_query_inactive(query_hash)
        
        return result if result else {
            "question": request.question,
            "error": "处理失败，无结果",
            "timestamp": datetime.now().isoformat()
        }
            
    except Exception as e:
        logger.error(f"查询处理失败: {e}", exc_info=True)
        # 确保标记查询为完成
        query_manager.mark_query_inactive(query_hash)
        
        # 发生错误时返回基本模拟数据
        error_response = {
            "question": request.question,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "generation": {
                "answer": "抱歉，处理查询时发生错误。请稍后重试或检查系统状态。",
                "citations": [],
                "generation_time": 0.0
            },
            "retrieval": {
                "kg_results": {"ke_results": [], "entities": [], "query_time": 0.0},
                "vector_results": [],
                "retrieval_time": 0.0
            },
            "performance": {
                "total_time": time.time() - start_time,
                "retrieval_time": 0.0,
                "generation_time": 0.0
            }
        }
        return error_response

@app.get("/api/test_connection")
async def test_connection():
    """测试各个组件连接状态"""
    components = {
        "neo4j": "unknown",
        "vector_db": "unknown",
        "llm": "unknown"
    }
    
    if system_config:
        # 测试Neo4j连接
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                system_config.neo4j_uri,
                auth=(system_config.neo4j_user, system_config.neo4j_password),
                connection_timeout=5
            )
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    components["neo4j"] = "connected"
            driver.close()
        except Exception as e:
            components["neo4j"] = f"error: {str(e)[:50]}"
        
        # 测试向量数据库
        vector_path = Path(system_config.vector_db_path)
        if vector_path.exists():
            components["vector_db"] = "exists"
        else:
            components["vector_db"] = "not_found"
        
        # 测试LLM服务
        try:
            import requests
            response = requests.get(system_config.llm_service_url, timeout=5)
            if response.status_code < 500:
                components["llm"] = "reachable"
            else:
                components["llm"] = f"error: {response.status_code}"
        except Exception as e:
            components["llm"] = f"error: {str(e)[:50]}"
    
    return {
        "timestamp": datetime.now().isoformat(),
        "rag_system_initialized": rag_system is not None,
        "components": components,
        "mode": "真实模式" if rag_system else "模拟模式"
    }

if __name__ == "__main__":
    # 启动服务器
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8885,
        reload=False,  # 关闭reload避免警告
        log_level="info"
    )