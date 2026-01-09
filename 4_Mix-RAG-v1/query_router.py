"""查询路由器（完全使用大模型分析）"""
import json
import requests
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import re

from config import SystemConfig
from data_types import QuestionType

logger = logging.getLogger(__name__)

@dataclass
class RouterResponse:
    """路由器响应"""
    question_type: QuestionType
    type_id: int
    entities: List[str]
    intent: str
    metadata: Dict[str, Any]

class QueryRouter:
    """查询路由器（完全使用大模型分析）"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.service_url = config.llm_service_url
        self.api_key = config.llm_api_key
        self.model = config.llm_model  # 使用Qwen-32B
    
    def analyze_query(self, question: str) -> RouterResponse:
        """
        使用大模型分析查询
        
        Returns:
            RouterResponse: 包含问题类型、实体列表、用户意图
        """
        try:
            # 调用大模型进行分析
            result = self._analyze_with_llm(question)
            
            if result:
                type_id, entities, intent = result
                question_type = QuestionType.from_int(type_id)
                
                logger.info(f"查询分析成功: 类型={question_type.value}(ID={type_id}), 实体={len(entities)}个, 意图={intent}")
                
                return RouterResponse(
                    question_type=question_type,
                    type_id=type_id,
                    entities=entities,
                    intent=intent,
                    metadata={"source": "llm_analysis"}
                )
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
        
        # 失败时返回默认值
        return RouterResponse(
            question_type=QuestionType.SIMPLE_FACT,
            type_id=1,
            entities=[],
            intent="查询事实信息",
            metadata={"source": "fallback", "error": "分析失败"}
        )
    
    def _analyze_with_llm(self, question: str) -> Optional[Tuple[int, List[str], str]]:
        """使用大模型分析查询"""
        try:
            prompt = self._build_analysis_prompt(question)
            response = self._call_llm(prompt)
            
            if response:
                return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"大模型分析失败: {e}")
        
        return None
    
    def _build_analysis_prompt(self, question: str) -> str:
        """构建分析提示词"""
        return f"""请分析以下航空航天制造领域的问题：

问题：{question}

请提供以下分析结果：
1. 问题类型（从以下3种中选择）：
   - 1（简单事实型）：查询具体的数值、定义、标准编号等可以直接从文档中查找的简单事实信息
   - 2（复杂逻辑型）：需要推理、比较、原因分析、工艺参数调整等复杂逻辑处理的问题
   - 3（开放语义型）：需要论述、总结、概述、讨论、综合分析等开放性内容的问题

2. 关键实体列表：从问题中提取的关键实体，如标准编号、材料牌号、工艺参数、专业术语等，禁止生成无关实体

3. 用户核心意图：用一句话概括用户想要了解的核心内容

请严格按照以下JSON格式输出：
{{
    "type_id": 1或2或3,
    "entities": ["实体1", "实体2", ...],
    "intent": "用户的核心意图描述"
}}

请确保分析准确、简洁，并严格按JSON格式输出。
/no_think"""
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用大模型"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "你是一个严谨的航空航天制造领域专家，请严格按照要求分析问题并提供准确、专业的回答。"
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,  # 低温度确保稳定性
                "max_tokens": 1024,
                "top_p": 0.9
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.service_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=20
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                logger.info(f"大模型调用成功，耗时: {elapsed_time:.2f}s")
                return content
            else:
                logger.error(f"大模型API错误: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("大模型调用超时")
            return None
        except Exception as e:
            logger.error(f"调用大模型失败: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Tuple[int, List[str], str]:
        """解析大模型响应"""
        try:
            # 提取JSON部分
            json_match = response.strip()
            
            # 尝试直接解析整个响应
            try:
                result = json.loads(json_match)
            except json.JSONDecodeError:
                # 尝试从文本中提取JSON
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.error(f"无法从响应中提取JSON: {response[:200]}...")
                    raise ValueError("无效的响应格式")
            
            # 提取和分析数据
            type_id = int(result.get("type_id", 1))
            entities = result.get("entities", [])
            intent = result.get("intent", "查询信息")
            
            # 验证和清理数据
            if type_id not in [1, 2, 3]:
                logger.warning(f"type_id超出范围: {type_id}，调整为1")
                type_id = 1
            
            if not isinstance(entities, list):
                entities = []
            
            # 清理实体列表
            entities = [str(e).strip() for e in entities if e and str(e).strip()]
            
            return type_id, entities, intent
            
        except Exception as e:
            logger.error(f"解析大模型响应失败: {e}")
            # 返回默认值
            return 1, [], "查询事实信息"