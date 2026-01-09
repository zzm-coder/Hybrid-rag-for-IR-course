import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from datetime import datetime
from neo4j import GraphDatabase
import pandas as pd
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class Neo4jCacheBuilder:
    """从Neo4j构建缓存文件 - 精简适配版"""
    
    def __init__(self, uri: str, user: str, password: str, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        初始化Neo4j连接和嵌入模型
        
        Args:
            uri: Neo4j数据库URI，如"bolt://localhost:7687"
            user: 用户名
            password: 密码
            embedding_model: 嵌入模型名称，默认为'all-MiniLM-L6-v2'
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.entity_cache = {}
        self.triplet_cache = {}
        self.node_labels = set()
        self.relationship_types = set()
        
        # 初始化嵌入模型
        print(f"加载嵌入模型: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.entity_embeddings = None
        self.entity_id_to_index = {}
        print("嵌入模型加载完成")
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
    
    def build_entity_cache(self, batch_size: int = 1000) -> Dict:
        """
        构建实体缓存 - 适配您的图谱结构
        
        Args:
            batch_size: 批量处理大小
            
        Returns:
            实体缓存字典
        """
        print("开始构建实体缓存...")
        
        # 查询所有节点（使用您的图谱结构）
        query = """
        MATCH (n)
        RETURN 
          id(n) as internal_id,
          n.node_id as node_id,
          n.name as name,
          n.type as type,
          n.category as category,
          n.subcategory as subcategory,
          n.created_at as created_at,
          labels(n) as labels
        ORDER BY id(n)
        """
        
        entity_cache = {}
        total_nodes = 0
        
        with self.driver.session() as session:
            result = session.run(query)
            
            for record in result:
                internal_id = record["internal_id"]
                node_id = record.get("node_id", f"internal_{internal_id}")
                name = record.get("name", "")
                entity_type = record.get("type", "Unknown")
                category = record.get("category", "")
                subcategory = record.get("subcategory", "")
                created_at = record.get("created_at", "")
                labels = record.get("labels", [])
                
                if labels:
                    self.node_labels.update(labels)
                
                # 构建实体信息 - 适配您的图谱结构
                entity_info = {
                    "internal_id": internal_id,
                    "node_id": node_id,
                    "name": name,
                    "type": entity_type,
                    "category": category,
                    "subcategory": subcategory,
                    "created_at": str(created_at) if created_at else "",
                    "labels": labels,
                    "primary_label": labels[0] if labels else entity_type
                }
                
                # 使用node_id作为主键，如果没有则使用internal_id
                cache_key = node_id if node_id and node_id != f"internal_{internal_id}" else f"internal_{internal_id}"
                entity_cache[cache_key] = entity_info
                
                total_nodes += 1
                if total_nodes % 1000 == 0:
                    print(f"  已处理 {total_nodes} 个实体...")
        
        self.entity_cache = entity_cache
        print(f"实体缓存构建完成，共 {len(entity_cache)} 个实体")
        return entity_cache
    
    def generate_entity_embeddings(self, batch_size: int = 512) -> np.ndarray:
        """
        为所有实体生成嵌入向量
        
        Args:
            batch_size: 批处理大小
            
        Returns:
            嵌入矩阵: shape (num_entities, embedding_dim)
        """
        if not self.entity_cache:
            raise ValueError("请先调用build_entity_cache()构建实体缓存")
        
        print("\n开始生成实体嵌入...")
        entity_ids = list(self.entity_cache.keys())
        num_entities = len(entity_ids)
        
        # 创建ID到索引的映射
        self.entity_id_to_index = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
        
        # 准备文本内容用于嵌入
        texts = []
        for entity_id in entity_ids:
            entity = self.entity_cache[entity_id]
            # 构建描述性文本，包含关键信息
            text_parts = []
            if entity.get("name"):
                text_parts.append(f"名称: {entity['name']}")
            if entity.get("type") and entity["type"] != "Unknown":
                text_parts.append(f"类型: {entity['type']}")
            if entity.get("category"):
                text_parts.append(f"类别: {entity['category']}")
            if entity.get("subcategory"):
                text_parts.append(f"子类: {entity['subcategory']}")
            if entity.get("primary_label"):
                text_parts.append(f"标签: {entity['primary_label']}")
            
            text = " | ".join(text_parts) if text_parts else entity_id
            texts.append(text)
        
        print(f"  共 {num_entities} 个实体，嵌入维度: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # 批量生成嵌入
        embeddings = []
        for i in range(0, num_entities, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            embeddings.append(batch_embeddings)
            
            if (i + batch_size) % 1000 == 0:
                print(f"  已处理 {i + len(batch_texts)}/{num_entities} 个实体...")
        
        # 合并所有嵌入
        self.entity_embeddings = np.vstack(embeddings)
        print(f"实体嵌入生成完成，形状: {self.entity_embeddings.shape}")
        
        return self.entity_embeddings
    
    def build_triplet_cache(self, batch_size: int = 1000) -> Dict:
        """
        构建三元组缓存 - 适配您的图谱结构
        
        Args:
            batch_size: 批量处理大小
            
        Returns:
            三元组缓存字典
        """
        print("开始构建三元组缓存...")
        
        # 查询所有关系（使用您的图谱结构）
        query = """
        MATCH (a)-[r]->(b)
        RETURN 
          id(r) as relation_id,
          id(a) as head_internal_id,
          id(b) as tail_internal_id,
          type(r) as relation_type,
          r.confidence as confidence,
          r.paragraph as paragraph,
          r.source as source,
          r.created_at as created_at,
          a.node_id as head_node_id,
          a.name as head_name,
          b.node_id as tail_node_id,
          b.name as tail_name
        ORDER BY id(r)
        """
        
        triplet_cache = defaultdict(list)
        total_relationships = 0
        
        with self.driver.session() as session:
            result = session.run(query)
            
            for record in result:
                relation_id = record["relation_id"]
                relation_type = record["relation_type"]
                confidence = record.get("confidence", 1.0)
                paragraph = record.get("paragraph", "")
                source = record.get("source", "unknown")
                created_at = record.get("created_at", "")
                
                # 头实体信息
                head_internal_id = record["head_internal_id"]
                head_node_id = record.get("head_node_id", f"internal_{head_internal_id}")
                head_name = record.get("head_name", "")
                
                # 尾实体信息
                tail_internal_id = record["tail_internal_id"]
                tail_node_id = record.get("tail_node_id", f"internal_{tail_internal_id}")
                tail_name = record.get("tail_name", "")
                
                # 记录关系类型
                self.relationship_types.add(relation_type)
                
                # 构建三元组信息 - 适配您的图谱结构
                triplet = {
                    "relation_id": relation_id,
                    "head": {
                        "internal_id": head_internal_id,
                        "node_id": head_node_id,
                        "name": head_name
                    },
                    "relation": relation_type,
                    "tail": {
                        "internal_id": tail_internal_id,
                        "node_id": tail_node_id,
                        "name": tail_name
                    },
                    "confidence": float(confidence) if confidence is not None else 1.0,
                    "paragraph": paragraph,
                    "source": source,  # 注意：这里使用source，而不是source_doc
                    "created_at": str(created_at) if created_at else ""
                }
                
                # 按source分组存储
                triplet_cache[source].append(triplet)
                
                total_relationships += 1
                if total_relationships % 1000 == 0:
                    print(f"  已处理 {total_relationships} 个关系...")
        
        self.triplet_cache = dict(triplet_cache)
        print(f"三元组缓存构建完成，共 {total_relationships} 个关系，分布在 {len(triplet_cache)} 个文档中")
        return self.triplet_cache
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            "total_entities": len(self.entity_cache),
            "total_relationships": sum(len(triplets) for triplets in self.triplet_cache.values()),
            "node_labels": list(self.node_labels),
            "relationship_types": list(self.relationship_types),
            "documents_with_triplets": len(self.triplet_cache),
            "embedding_dimension": self.entity_embeddings.shape[1] if self.entity_embeddings is not None else 0,
            "generated_at": datetime.now().isoformat()
        }
        
        # 按类型统计实体数量
        type_counts = defaultdict(int)
        for entity_info in self.entity_cache.values():
            entity_type = entity_info.get("type", "Unknown")
            type_counts[entity_type] += 1
        
        stats["entity_counts_by_type"] = dict(type_counts)
        
        # 按分类统计实体数量
        category_counts = defaultdict(int)
        for entity_info in self.entity_cache.values():
            category = entity_info.get("category", "Unknown")
            category_counts[category] += 1
        
        stats["entity_counts_by_category"] = dict(category_counts)
        
        # 按关系类型统计三元组数量
        rel_counts = defaultdict(int)
        for triplets in self.triplet_cache.values():
            for triplet in triplets:
                rel_type = triplet.get("relation", "")
                if rel_type:
                    rel_counts[rel_type] += 1
        
        stats["triplet_counts_by_relation"] = dict(rel_counts)
        
        # 按文档统计三元组数量
        doc_counts = {}
        for doc_name, triplets in self.triplet_cache.items():
            doc_counts[doc_name] = len(triplets)
        
        stats["triplet_counts_by_document"] = doc_counts
        
        return stats
    
    def save_caches(self, output_dir: str):
        """
        保存所有缓存文件
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存实体缓存
        entity_cache_file = output_path / "entity_cache.json"
        with open(entity_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.entity_cache, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"实体缓存已保存到: {entity_cache_file}")
        
        # 保存三元组缓存
        triplet_cache_file = output_path / "triplet_cache.json"
        with open(triplet_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.triplet_cache, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"三元组缓存已保存到: {triplet_cache_file}")
        
        # 保存实体嵌入
        if self.entity_embeddings is not None:
            embeddings_file = output_path / "entity_embeddings.npy"
            np.save(embeddings_file, self.entity_embeddings)
            print(f"实体嵌入已保存到: {embeddings_file}")
            
            # 保存实体ID到索引的映射
            mapping_file = output_path / "entity_id_to_index.json"
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.entity_id_to_index, f, ensure_ascii=False, indent=2)
            print(f"实体ID映射已保存到: {mapping_file}")
        
        # 保存统计信息
        stats = self.get_statistics()
        stats_file = output_path / "cache_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"统计信息已保存到: {stats_file}")
        
        # 生成CSV格式的统计摘要
        self.generate_statistics_csv(stats, output_path)
        
        print(f"\n所有缓存文件已保存到目录: {output_path}")
    
    def generate_statistics_csv(self, stats: Dict, output_path: Path):
        """生成CSV格式的统计摘要"""
        # 实体按类型统计
        if "entity_counts_by_type" in stats:
            entity_df = pd.DataFrame([
                {"type": etype, "count": count}
                for etype, count in stats["entity_counts_by_type"].items()
            ])
            entity_df = entity_df.sort_values("count", ascending=False)
            entity_csv = output_path / "entity_counts_by_type.csv"
            entity_df.to_csv(entity_csv, index=False, encoding='utf-8-sig')
        
        # 实体按分类统计
        if "entity_counts_by_category" in stats:
            category_df = pd.DataFrame([
                {"category": category, "count": count}
                for category, count in stats["entity_counts_by_category"].items()
            ])
            category_df = category_df.sort_values("count", ascending=False)
            category_csv = output_path / "entity_counts_by_category.csv"
            category_df.to_csv(category_csv, index=False, encoding='utf-8-sig')
        
        # 三元组按关系类型统计
        if "triplet_counts_by_relation" in stats:
            triplet_df = pd.DataFrame([
                {"relation_type": rel_type, "count": count}
                for rel_type, count in stats["triplet_counts_by_relation"].items()
            ])
            triplet_df = triplet_df.sort_values("count", ascending=False)
            triplet_csv = output_path / "triplet_counts_by_relation.csv"
            triplet_df.to_csv(triplet_csv, index=False, encoding='utf-8-sig')
        
        # 三元组按文档统计（取前20个）
        if "triplet_counts_by_document" in stats:
            doc_counts = stats["triplet_counts_by_document"]
            top_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            doc_df = pd.DataFrame([
                {"document": doc, "triplet_count": count}
                for doc, count in top_docs
            ])
            doc_csv = output_path / "top_documents_by_triplet_count.csv"
            doc_df.to_csv(doc_csv, index=False, encoding='utf-8-sig')
        
        print(f"统计CSV文件已生成")
    
    def validate_cache_consistency(self) -> Dict[str, List[str]]:
        """
        验证缓存的一致性
        
        Returns:
            包含问题和警告的字典
        """
        print("验证缓存一致性...")
        
        issues = {
            "warnings": [],
            "errors": []
        }
        
        # 检查是否有实体在三元组中引用但不在实体缓存中
        referenced_node_ids = set()
        for triplets in self.triplet_cache.values():
            for triplet in triplets:
                head_node_id = triplet["head"]["node_id"]
                tail_node_id = triplet["tail"]["node_id"]
                referenced_node_ids.add(head_node_id)
                referenced_node_ids.add(tail_node_id)
        
        # 检查实体缓存中的节点
        cached_node_ids = set(self.entity_cache.keys())
        
        # 检查三元组中引用的节点是否都在实体缓存中
        missing_nodes = referenced_node_ids - cached_node_ids
        if missing_nodes:
            issues["warnings"].append(f"三元组中引用了 {len(missing_nodes)} 个不在实体缓存中的节点")
            # 显示前5个缺失的节点
            for node_id in list(missing_nodes)[:5]:
                issues["warnings"].append(f"  缺失节点: {node_id}")
        
        # 检查实体名称的唯一性
        name_counts = defaultdict(int)
        for entity_info in self.entity_cache.values():
            name = entity_info.get("name", "")
            if name:
                name_counts[name] += 1
        
        duplicate_names = {name: count for name, count in name_counts.items() if count > 1}
        if duplicate_names:
            issues["warnings"].append(f"发现 {len(duplicate_names)} 个重复的实体名称")
            for name, count in list(duplicate_names.items())[:5]:
                issues["warnings"].append(f"  名称 '{name}' 出现 {count} 次")
        
        # 检查空名称
        empty_name_count = sum(1 for entity_info in self.entity_cache.values() 
                              if not entity_info.get("name", "").strip())
        if empty_name_count:
            issues["warnings"].append(f"发现 {empty_name_count} 个实体没有名称")
        
        print(f"一致性检查完成: {len(issues['warnings'])} 个警告, {len(issues['errors'])} 个错误")
        return issues


# 快速构建缓存的工具函数
def build_all_caches(neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                     output_dir: str = "./kg_cache",
                     embedding_model: str = 'all-MiniLM-L6-v2') -> Neo4jCacheBuilder:
    """
    快速构建所有缓存
    
    Args:
        neo4j_uri: Neo4j数据库URI
        neo4j_user: 用户名
        neo4j_password: 密码
        output_dir: 输出目录
        embedding_model: 嵌入模型名称
        
    Returns:
        Neo4jCacheBuilder实例
    """
    print("=== 开始构建Neo4j知识图谱缓存 ===")
    
    # 初始化构建器
    cache_builder = Neo4jCacheBuilder(neo4j_uri, neo4j_user, neo4j_password, embedding_model)
    
    try:
        # 构建实体缓存
        cache_builder.build_entity_cache()
        
        # 生成实体嵌入
        cache_builder.generate_entity_embeddings()
        
        # 构建三元组缓存
        cache_builder.build_triplet_cache()
        
        # 验证一致性
        issues = cache_builder.validate_cache_consistency()
        
        # 保存所有缓存
        cache_builder.save_caches(output_dir)
        
        # 打印统计信息
        stats = cache_builder.get_statistics()
        print("\n=== 缓存统计信息 ===")
        print(f"总实体数: {stats['total_entities']}")
        print(f"总关系数: {stats['total_relationships']}")
        print(f"文档数: {stats['documents_with_triplets']}")
        print(f"节点标签: {len(stats['node_labels'])} 种")
        print(f"关系类型: {len(stats['relationship_types'])} 种")
        print(f"嵌入维度: {stats['embedding_dimension']}")
        
        # 打印实体按类型统计
        print("\n实体按类型统计:")
        if 'entity_counts_by_type' in stats:
            for etype, count in stats['entity_counts_by_type'].items():
                print(f"  {etype}: {count}")
        
        # 打印关系按类型统计（前10个）
        print("\n关系按类型统计（前10个）:")
        if 'triplet_counts_by_relation' in stats:
            rel_counts = stats['triplet_counts_by_relation']
            sorted_rels = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for rel_type, count in sorted_rels:
                print(f"  {rel_type}: {count}")
        
        return cache_builder
        
    except Exception as e:
        print(f"构建缓存时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        cache_builder.close()


# 主函数
def main():
    """主函数：从Neo4j构建缓存"""
    # 配置参数
    NEO4J_URI = "URL"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "PASSWORD"
    OUTPUT_DIR = "/home/zzm/Project_1/kg-hk/2_kg_construction/kg_cache"
    EMBEDDING_MODEL = "/hdd1/checkpoints/sentence-transformers/text2vec-base-chinese"  # 可替换为其他模型，如"all-mpnet-base-v2"
    
    # 构建所有缓存
    cache_builder = build_all_caches(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        output_dir=OUTPUT_DIR,
        embedding_model=EMBEDDING_MODEL
    )
    
    if cache_builder:
        print(f"\n缓存构建完成！文件保存在: {OUTPUT_DIR}")
        print(f"嵌入文件: {OUTPUT_DIR}/entity_embeddings.npy")
        print(f"实体ID映射: {OUTPUT_DIR}/entity_id_to_index.json")


if __name__ == "__main__":
    # 运行主函数
    main()