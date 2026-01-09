"""知识图谱检索器 - 使用增强的实体匹配"""
import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from neo4j import GraphDatabase

from data_types import SystemConfig, KGResult, KGTriple, QuestionType
from entity_matcher import EntityMatcher

logger = logging.getLogger(__name__)

class KnowledgeGraphRetriever:
    """增强版知识图谱检索器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.cache_path = Path(config.kg_cache_path)
        
        # 初始化Neo4j连接
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        
        # 初始化实体匹配器
        self.entity_matcher = EntityMatcher(config)
        
        # 加载三联缓存
        self.triplet_cache = self._load_triplet_cache()
        
        logger.info("知识图谱检索器初始化完成")
    
    def _load_triplet_cache(self) -> Dict[str, List[Dict]]:
        """加载三联缓存"""
        cache_file = self.cache_path / "triplet_cache.json"
        if not cache_file.exists():
            logger.error(f"三联缓存文件不存在: {cache_file}")
            return {}
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载三联缓存失败: {e}")
            return {}
    
    def query_kg(self, question: str, entities: List[str], question_type: QuestionType) -> KGResult:
        """
        查询知识图谱
        
        Args:
            question: 问题文本
            entities: 提取的实体列表
            question_type: 问题类型
            
        Returns:
            KGResult: 知识图谱查询结果
        """
        start_time = time.time()
        
        try:
            # 1. 提取关键词
            keywords = self.entity_matcher.extract_keywords_from_question(question)
            
            # 2. 匹配实体
            matched_entities = self.entity_matcher.match_entities(question, keywords + entities)
            
            # 3. 从缓存中获取三联
            cache_triples = self._get_triples_from_cache(matched_entities, question_type)
            
            # 4. 从Neo4j查询三联（补充）
            neo4j_triples = self._query_triples_from_neo4j(matched_entities, question_type)
            
            # 5. 合并并去重
            all_triples = cache_triples + neo4j_triples
            unique_triples = self._deduplicate_triples(all_triples)
            
            # 6. 提取所有相关实体
            all_kg_entities = set()
            for triple in unique_triples:
                all_kg_entities.add(triple.head)
                all_kg_entities.add(triple.tail)
            
            # 添加匹配的实体名称
            for entity_info in matched_entities:
                all_kg_entities.add(entity_info["name"])
            
            result = KGResult(
                triples=unique_triples,
                entities=list(all_kg_entities),
                query_time=time.time() - start_time
            )
            
            logger.info(f"KG查询完成: 问题='{question[:50]}...'，找到 {len(unique_triples)} 个三元组，{len(all_kg_entities)} 个实体")
            
            return result
            
        except Exception as e:
            logger.error(f"KG查询失败: {e}")
            return KGResult([], [], time.time() - start_time)
    
    def _get_triples_from_cache(self, matched_entities: List[Dict], question_type: QuestionType) -> List[KGTriple]:
        """从缓存中获取三元组"""
        triples = []
        
        if not matched_entities or not self.triplet_cache:
            return triples
        
        entity_names = [entity["name"] for entity in matched_entities[:10]]
        entity_ids = [entity["id"] for entity in matched_entities[:10]]
        
        # 遍历所有三联缓存
        for source_file, triple_list in self.triplet_cache.items():
            for triple_data in triple_list:
                head_data = triple_data.get("head", {})
                tail_data = triple_data.get("tail", {})
                
                head_name = head_data.get("name", "")
                tail_name = tail_data.get("name", "")
                head_id = head_data.get("node_id", "")
                tail_id = tail_data.get("node_id", "")
                
                relation = triple_data.get("relation", "")
                paragraph = triple_data.get("paragraph", "")
                confidence = triple_data.get("confidence", 1.0)
                
                # 检查是否与任何实体匹配
                entity_match = False
                
                # 检查名称匹配
                for entity_name in entity_names:
                    if entity_name in head_name or entity_name in tail_name:
                        entity_match = True
                        break
                
                # 检查ID匹配
                if not entity_match:
                    for entity_id in entity_ids:
                        if entity_id == head_id or entity_id == tail_id:
                            entity_match = True
                            break
                
                # 检查关系是否与问题相关
                relation_relevant = self._is_relation_relevant(relation, question_type)
                
                if entity_match and relation_relevant:
                    triples.append(KGTriple(
                        head=head_name,
                        relation=relation,
                        tail=tail_name,
                        source=source_file,
                        paragraph=paragraph,
                        confidence=confidence
                    ))
        
        # 根据问题类型调整返回数量
        max_triples = {
            QuestionType.SIMPLE_FACT: 10,
            QuestionType.COMPLEX_LOGIC: 15,
            QuestionType.OPEN_SEMANTIC: 20
        }.get(question_type, 10)
        
        return triples[:max_triples]
    
    def _query_triples_from_neo4j(self, matched_entities: List[Dict], question_type: QuestionType) -> List[KGTriple]:
        """从Neo4j查询三元组"""
        triples = []
        
        if not matched_entities or not self.driver:
            return triples
        
        entity_names = [entity["name"] for entity in matched_entities[:5]]
        
        try:
            with self.driver.session() as session:
                for entity_name in entity_names:
                    # 根据问题类型调整查询
                    if question_type == QuestionType.SIMPLE_FACT:
                        query = """
                        MATCH (n)-[r]->(m)
                        WHERE n.name CONTAINS $entity OR n.id CONTAINS $entity
                        RETURN n.name as head, type(r) as relation, m.name as tail, 
                               properties(r) as rel_props
                        LIMIT 5
                        """
                    elif question_type == QuestionType.COMPLEX_LOGIC:
                        query = """
                        MATCH path = (n)-[r*1..2]->(m)
                        WHERE n.name CONTAINS $entity
                        UNWIND relationships(path) as rel
                        RETURN startNode(rel).name as head, type(rel) as relation, 
                               endNode(rel).name as tail, properties(rel) as rel_props
                        LIMIT 10
                        """
                    else:  # OPEN_SEMANTIC
                        query = """
                        MATCH (n)-[r]->(m)
                        WHERE n.name CONTAINS $entity
                        RETURN n.name as head, type(r) as relation, m.name as tail,
                               properties(r) as rel_props
                        LIMIT 3
                        """
                    
                    result = session.run(query, {"entity": entity_name})
                    for record in result:
                        # 尝试从关系属性中提取段落信息
                        rel_props = record.get("rel_props", {})
                        paragraph = rel_props.get("paragraph", "") if rel_props else ""
                        
                        triples.append(KGTriple(
                            head=record["head"] or "",
                            relation=record["relation"] or "",
                            tail=record["tail"] or "",
                            source="neo4j",
                            paragraph=paragraph,
                            confidence=0.9
                        ))
                        
        except Exception as e:
            logger.error(f"Neo4j查询失败: {e}")
        
        return triples
    
    def _is_relation_relevant(self, relation: str, question_type: QuestionType) -> bool:
        """检查关系是否与问题类型相关"""
        if not relation:
            return False
        
        # 关系类型分类（扩展版，包含常见事实、逻辑与语义类关系）
        fact_relations = [
            "is_a", "has", "has_parameter", "has_value", "has_standard", "has_property",
            "has_method", "has_test", "has_test_method", "has_requirement", "has_component",
            "has_part", "has_material", "has_condition", "has_version", "has_result",
            "has_output", "has_input", "has_function", "has_subsystem", "has_subcomponent",
            "has_definition", "has_structure", "has_type", "has_formula", "has_format",
            "has_step", "has_process", "has_service", "has_equipment", "has_behavior",
            "provided_by", "produced_by", "published_by", "issued_by", "organized_by",
            "developed_by", "drafted_by", "administered_by", "responsible_for", "provided_by"
        ]
        logic_relations = [
            "requires", "require", "depends_on", "depends", "affects", "affect", "causes",
            "leads_to", "influences", "interfere_with", "prevent", "protect", "replace",
            "combine", "combine_with", "combined_with", "follow", "should_follow", "must_follow",
            "not_require", "not_required", "not_recommended", "prefer", "allow", "access",
            "connect_to", "connects_to", "connected_to", "connects_with", "call_service",
            "transmit_by", "transmit", "output_to", "output", "return_value", "return_type",
            "calculate_by", "calculate", "calculated_by", "calculate"
        ]
        semantic_relations = [
            "related_to", "similar_to", "part_of", "belongs_to", "associated_with", "compatible_with",
            "used_with", "used_by", "used_in", "used_for", "representation", "representation_of",
            "representation_in", "representation_to", "equivalent", "matches_with", "equal_to",
            "based_on", "constitute", "contribute_to", "related", "belongs", "associated"
        ]
        
        relation_lower = relation.lower()
        
        if question_type == QuestionType.SIMPLE_FACT:
            return any(r in relation_lower for r in fact_relations)
        elif question_type == QuestionType.COMPLEX_LOGIC:
            return any(r in relation_lower for r in fact_relations + logic_relations)
        else:  # OPEN_SEMANTIC
            return any(r in relation_lower for r in fact_relations + logic_relations + semantic_relations)
    
    def _deduplicate_triples(self, triples: List[KGTriple]) -> List[KGTriple]:
        """去重三元组"""
        seen = set()
        unique_triples = []
        
        for triple in triples:
            key = (triple.head, triple.relation, triple.tail)
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
        
        # 按置信度排序
        unique_triples.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_triples
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()