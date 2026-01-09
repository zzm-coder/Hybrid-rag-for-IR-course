"""LLM生成器"""
import re
import requests
import logging
import time
from typing import Tuple, List

from config import SystemConfig
from data_types import LLMResponse

logger = logging.getLogger(__name__)

class LLMGenerator:
    """LLM生成器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.service_url = config.llm_service_url
    
    def generate_answer(self, question: str, context: str) -> LLMResponse:
        """
        生成答案
        """
        start_time = time.time()
        
        try:
            prompt = self._build_prompt(question, context)
            response = self._call_llm(prompt)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            answer, citations = self._parse_response(response)
            
            return LLMResponse(
                answer=answer,
                evidence_citations=citations,
                raw_response=response,
                generation_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return LLMResponse(
                answer="生成答案时发生错误，请稍后重试。",
                evidence_citations=[],
                raw_response="",
                generation_time=time.time() - start_time
            )
    
    def _build_prompt(self, question: str, context: str) -> str:
        """构建提示词"""
        prompt_template = f"""
你是一个严谨的航空航天标准文档问答系统。请严格按以下规则回答：

## 任务要求
1. **基于证据回答**：必须严格基于提供的上下文信息生成答案
2. **诚实可信**：如果提供的上下文不足以回答问题，必须明确声明"无法根据提供的信息回答"
3. **精确引用**：回答中必须标注引用来源，使用中括号标注编号，格式为[编号]
4. **避免幻觉**：禁止编造、推测或添加上下文以外的信息

## 上下文信息
{context}

## 用户问题
{question}

## 输出格式
请严格按照以下格式输出：
【答案】[你的答案，必须包含引用标注如[1][2]]
【证据】[列出引用的证据来源，格式：1. 文档名称 (相关段落摘要)]

## 注意事项
- 如果上下文中有矛盾信息，以最新或更权威的来源为准
- 数值参数必须精确引用
- 工艺流程必须按顺序描述
- 标准要求必须注明标准编号
"""
        return prompt_template
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM服务"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.llm_api_key}"
            }
            
            payload = {
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content": "你是一个严谨的航空航天制造领域专家。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000,
                "top_p": 0.9
            }
            
            response = requests.post(
                f"{self.service_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"LLM API错误: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise Exception("LLM服务响应超时")
        except Exception as e:
            raise Exception(f"调用LLM失败: {e}")
    
    def _parse_response(self, response: str) -> Tuple[str, List[str]]:
        """解析LLM响应"""
        answer_match = re.search(r'【答案】\s*(.*?)(?=\n【证据】|\Z)', response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else response
        # 解析证据区，尝试提取编号->文档名的映射
        evidence_match = re.search(r'【证据】\s*(.*)', response, re.DOTALL)
        index_to_md = {}
        evidence_md_list = []
        if evidence_match:
            evidence_text = evidence_match.group(1)
            # 每行可能为: 1. 文档名称 (段落...)：描述
            for line in evidence_text.splitlines():
                # 先尝试解析行首的编号
                idx_m = re.match(r'\s*(\d+)[\.)）]?\s*(.*)', line)
                content_part = line
                idx = None
                if idx_m:
                    idx = idx_m.group(1)
                    content_part = idx_m.group(2)

                # 在行中寻找第一个以 .md 结尾的文件名（允许包含多种字符）
                md_m = re.search(r'([^\s\(\)\[\]]+?\.md)', content_part, re.IGNORECASE)
                if md_m:
                    md = md_m.group(1).split('/')[-1]
                    evidence_md_list.append(md)
                    if idx:
                        index_to_md[idx] = md

            # 作为补充：如果没有行级编号映射，但整体有 md 列表，可使用其顺序作为候选
            # 去重并保持出现顺序
            seen_md = set()
            evidence_md_list = [m for m in evidence_md_list if not (m in seen_md or seen_md.add(m))]

        else:
            evidence_text = ""

        # 从答案中提取编号引用并只映射为 md 文件名（忽略未映射的纯数字引用）
        citations = []
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        for m in matches:
            if m in index_to_md:
                md_name = index_to_md[m]
                if md_name and md_name.lower().endswith('.md'):
                    citations.append(md_name)
            else:
                # 忽略未在证据区解析到的纯编号引用
                continue

        # 如果答案没有引用编号，但证据区包含 md，作为回退将证据中的 md 全部加入（按出现顺序）
        if not citations and evidence_md_list:
            citations = list(evidence_md_list)

        # 去重并保持顺序
        seen = set()
        final = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                final.append(c)

        return answer, final

    def evaluate_accuracy(self, question: str, answer: str, ground_truth: str, timeout: int = 10) -> float:
        """使用大模型评估答案与参考答案的准确率（0.0-1.0）。
        返回浮点数，失败时抛出异常或返回 -1.0 标识不可用。
        """
        try:
            prompt = f"评估任务：请判断下面生成答案的正确率（0.0 到 1.0 之间）。问题：{question}\n\n参考答案：{ground_truth}\n\n生成答案：{answer}\n\n要求：\n\n1.判断生成答案的意图与参考答案的意图是否一致;\n\n2.判断生成答案是否正确的回答了问题;\n\n3.如果生成答案中存在无法回答，则输出0.000;\n\n4.正确率只输出一个数值，范围 [0.0,1.0]，保留三位小数，例如 0.750。不要添加其他文字。/no_think"
            # 直接调用 _call_llm 以复用配置
            resp = self._call_llm(prompt)
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL | re.IGNORECASE)
            # 提取第一个 0-1 浮点数
            m = re.search(r'(?:(?:1(?:\.0+)?)|(?:0(?:\.\d+)?))', resp)
            if m:
                val = float(m.group(0))
                return max(0.0, min(1.0, val))
            # 退回到寻找小数点表示
            m2 = re.search(r'(0?\.\d+|1(?:\.0+)?)', resp)
            if m2:
                val = float(m2.group(0))
                return max(0.0, min(1.0, val))
            raise Exception(f"无法从模型响应解析出数值: {resp}")
        except Exception as e:
            logger.warning(f"通过LLM评估准确率失败: {e}")
            return -1.0