import json
from pathlib import Path
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from datetime import datetime
import os
from collections import defaultdict
import re

# === Neo4j 配置 ===
NEO4J_URI = "URL"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "PASSWORD"

class SimplifiedKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        """初始化知识图谱构建器"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.entity_cache = {}  # 节点缓存：node_id -> 节点信息
        self.processed_files = set()  # 已处理文件
        self.new_nodes_this_run = set()  # 本次运行新增的节点
        self.new_relations_this_run = set()  # 本次运行新增的关系
        self.stats = {
            "total_nodes": 0,
            "total_relations": 0,
            "nodes_by_type": defaultdict(int),
            "nodes_by_category": defaultdict(int),
            "nodes_by_subcategory": defaultdict(int),
            "relations_by_type": defaultdict(int)
        }
    
    def sanitize_label(self, label: str) -> str:
        """清理标签，移除特殊字符"""
        # 移除空格和特殊字符，保留字母、数字、中文和下划线
        sanitized = re.sub(r'[^\w\u4e00-\u9fff]', '_', label)
        # 如果以数字开头，添加前缀
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        return sanitized
    
    # ==================== 分类映射 ====================
    def categorize_entity(self, entity_type: str) -> tuple:
        """根据实体类型映射到对应的分类和子分类"""
        category_mapping = {
            # 标准文档层
            "Standard": ("标准文档层", "标准编号"),
            "Title": ("标准文档层", "标准标题"),
            "Organization": ("标准文档层", "组织关系"),
            
            # 技术工艺层 - 设计类
            "Component": ("技术工艺层", "设计类"),
            "Requirement": ("技术工艺层", "设计类"),
            "Parameter": ("技术工艺层", "设计类"),
            "Value": ("技术工艺层", "设计类"),
            
            # 技术工艺层 - 材料类
            "Material": ("技术工艺层", "材料类"),
            
            # 技术工艺层 - 工艺类
            "Process": ("技术工艺层", "工艺类"),
            
            # 技术工艺层 - 试验类
            "Test": ("技术工艺层", "试验类"),
            "Equipment": ("技术工艺层", "试验类"),

            # 技术工艺层 - 保障类
            "Defect": ("技术工艺层", "保障类")
        }

        # 默认映射到设计类
        return category_mapping.get(entity_type, ("技术工艺层", "设计类"))
    
    # ==================== 核心构建功能 ====================
    def create_hierarchy(self):
        """创建分层分类体系"""
        with self.driver.session() as session:
            # 检查是否已存在分类节点
            result = session.run("""
                MATCH (c:MainCategory)
                RETURN COUNT(c) as count
            """)
            
            count = result.single()["count"]
            if count > 0:
                print("✅ 分类体系已存在，跳过创建")
                return
            
            # 清空现有分类
            session.run("MATCH (c:Category) DETACH DELETE c")
            
            # 创建主分类层
            main_categories = ["标准文档层", "技术工艺层"]
            for main_cat in main_categories:
                session.run("""
                    CREATE (mc:MainCategory {
                        name: $name,
                        level: 'main',
                        created_at: datetime()
                    })
                """, name=main_cat)
            
            # 创建子分类并关联
            sub_categories = {
                "标准文档层": ["标准编号", "标准标题", "组织关系"],
                "技术工艺层": ["设计类", "材料类", "工艺类", "试验类", "保障类"]
            }
            
            for main_cat, subs in sub_categories.items():
                for sub_cat in subs:
                    session.run("""
                        CREATE (sc:SubCategory {
                            name: $sub_name,
                            level: 'sub',
                            created_at: datetime()
                        })
                        WITH sc
                        MATCH (mc:MainCategory {name: $main_name})
                        CREATE (sc)-[:BELONGS_TO]->(mc)
                    """, sub_name=sub_cat, main_name=main_cat)
            
            print("✅ 分层分类体系创建完成")
    
    def get_node_id(self, entity_name: str, entity_type: str) -> str:
        """生成节点唯一ID（基于小写名称和类型）"""
        # 使用小写名称和类型组合生成MD5哈希作为唯一ID
        combined = f"{entity_name.strip().lower()}:{entity_type.strip().lower()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def node_exists(self, node_id: str) -> Tuple[bool, Optional[Dict]]:
        """检查节点是否已存在，返回是否存在和节点信息"""
        # 先检查缓存
        if node_id in self.entity_cache:
            return True, self.entity_cache[node_id]
        
        # 再检查数据库
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity {node_id: $node_id})
                RETURN n.name as name, n.type as type, 
                       n.category as category, n.subcategory as subcategory
                LIMIT 1
            """, node_id=node_id)
            
            record = result.single()
            if record:
                node_info = {
                    "name": record["name"],
                    "type": record["type"],
                    "category": record.get("category", ""),
                    "subcategory": record.get("subcategory", "")
                }
                self.entity_cache[node_id] = node_info
                return True, node_info
        
        return False, None
    
    def create_node(self, entity: Dict[str, Any]) -> Tuple[Optional[str], bool]:
        """创建实体节点，返回（节点ID，是否新建）"""
        entity_name = entity.get("name", "").strip()
        entity_type = entity.get("type", "Unknown").strip()
        
        if not entity_name:
            return None, False
        
        # 生成节点ID
        node_id = self.get_node_id(entity_name, entity_type)
        
        # 检查节点是否已存在
        exists, node_info = self.node_exists(node_id)
        if exists:
            return node_id, False  # 节点已存在，不是新建
        
        # 获取分类
        category, subcategory = self.categorize_entity(entity_type)
        
        # 清理标签
        clean_entity_type = self.sanitize_label(entity_type)
        
        # 创建节点
        with self.driver.session() as session:
            try:
                # 创建实体节点，使用实体类型作为标签
                query = f"""
                    CREATE (n:Entity:{clean_entity_type} {{
                        name: $name,
                        type: $type,
                        category: $category,
                        subcategory: $subcategory,
                        node_id: $node_id,
                        created_at: datetime()
                    }})
                    RETURN n.node_id as node_id
                """
                
                result = session.run(query, 
                                     name=entity_name, 
                                     type=entity_type,
                                     category=category, 
                                     subcategory=subcategory,
                                     node_id=node_id)
                
                record = result.single()
                if record:
                    actual_node_id = record["node_id"]
                    
                    # 更新缓存
                    self.entity_cache[actual_node_id] = {
                        "name": entity_name,
                        "type": entity_type,
                        "category": category,
                        "subcategory": subcategory
                    }
                    
                    # 添加到本次运行的新节点集合
                    self.new_nodes_this_run.add(actual_node_id)
                    
                    # 更新统计
                    self.stats["total_nodes"] += 1
                    self.stats["nodes_by_type"][entity_type] += 1
                    self.stats["nodes_by_category"][category] += 1
                    self.stats["nodes_by_subcategory"][subcategory] += 1
                    
                    # 关联到分类节点
                    self.link_to_category(actual_node_id, subcategory)
                    
                    return actual_node_id, True
                    
            except Exception as e:
                # 如果创建失败，可能是因为节点已存在（并发问题）
                print(f"⚠️  创建节点失败: {e}")
                # 再次检查
                exists, _ = self.node_exists(node_id)
                if exists:
                    return node_id, False
        
        return None, False
    
    def link_to_category(self, node_id: str, subcategory: str):
        """将节点关联到子分类"""
        with self.driver.session() as session:
            try:
                session.run("""
                    MATCH (n:Entity {node_id: $node_id})
                    MATCH (sc:SubCategory {name: $subcategory})
                    MERGE (n)-[:CLASSIFIED_AS]->(sc)
                """, node_id=node_id, subcategory=subcategory)
            except Exception as e:
                print(f"⚠️  关联分类失败: {e}")
    
    def relation_exists(self, head_id: str, tail_id: str, rel_type: str) -> bool:
        """检查关系是否已存在"""
        clean_rel_type = self.sanitize_label(rel_type)
        rel_key = f"{head_id}->{tail_id}->{clean_rel_type}"
        
        # 检查是否已在本次运行中创建
        if rel_key in self.new_relations_this_run:
            return True
        
        # 检查数据库
        with self.driver.session() as session:
            query = f"""
                MATCH (h:Entity {{node_id: $head_id}})-[r:`{clean_rel_type}`]->(t:Entity {{node_id: $tail_id}})
                RETURN COUNT(r) as count
                LIMIT 1
            """
            
            result = session.run(query, head_id=head_id, tail_id=tail_id)
            count = result.single()["count"]
            return count > 0
    
    def create_relation(self, head_id: str, tail_id: str, rel_data: Dict[str, Any]) -> Tuple[bool, bool]:
        """创建关系，返回（是否成功，是否新建）"""
        if not head_id or not tail_id:
            return False, False
        
        rel_type = rel_data.get("relation", "").strip()
        if not rel_type:
            return False, False
        
        # 清理关系类型
        clean_rel_type = self.sanitize_label(rel_type)
        rel_key = f"{head_id}->{tail_id}->{clean_rel_type}"
        
        # 检查关系是否已存在
        if self.relation_exists(head_id, tail_id, rel_type):
            return True, False  # 关系已存在，不是新建
        
        try:
            with self.driver.session() as session:
                # 使用动态关系类型
                query = f"""
                    MATCH (h:Entity {{node_id: $head_id}})
                    MATCH (t:Entity {{node_id: $tail_id}})
                    MERGE (h)-[r:`{clean_rel_type}`]->(t)
                    ON CREATE SET r.confidence = $confidence,
                                 r.source = $source,
                                 r.paragraph = $paragraph,
                                 r.created_at = datetime(),
                                 r.relation_type = $rel_type
                    RETURN COUNT(r) as count
                """
                
                result = session.run(query, 
                    head_id=head_id, 
                    tail_id=tail_id,
                    confidence=float(rel_data.get("confidence", 0.5)),
                    source=rel_data.get("source", ""),
                    paragraph=rel_data.get("paragraph", ""),
                    rel_type=rel_type)
                
                if result.single()["count"] > 0:
                    # 添加到本次运行的新关系集合
                    self.new_relations_this_run.add(rel_key)
                    
                    # 更新统计
                    self.stats["total_relations"] += 1
                    self.stats["relations_by_type"][rel_type] += 1
                    
                    return True, True  # 关系创建成功，是新建
                    
        except Exception as e:
            print(f"❌ 创建关系失败: {e}")
        
        return False, False
    
    # ==================== 文件处理 ====================
    def process_json_file(self, file_path: Path) -> Dict[str, int]:
        """处理单个JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")
            return {"nodes": 0, "relations": 0}
        
        # 确保是列表
        if not isinstance(data, list):
            data = [data]
        
        nodes_created = 0
        relations_created = 0
        total_triples = len(data)
        
        print(f"  处理 {total_triples} 个三元组")
        
        for i, item in enumerate(data):
            # 创建节点（或获取现有节点）
            head_entity = item.get("head", {})
            tail_entity = item.get("tail", {})
            
            head_id, head_is_new = self.create_node(head_entity)
            tail_id, tail_is_new = self.create_node(tail_entity)
            
            # 统计新创建的节点
            if head_is_new:
                nodes_created += 1
            if tail_is_new:
                nodes_created += 1
            
            # 创建关系
            if head_id and tail_id:
                rel_created, rel_is_new = self.create_relation(head_id, tail_id, item)
                if rel_created and rel_is_new:
                    relations_created += 1
            
            # 显示进度
            if (i + 1) % 100 == 0 or (i + 1) == total_triples:
                print(f"    进度: {i+1}/{total_triples} (新节点: {nodes_created}, 新关系: {relations_created})")
        
        # 记录已处理文件（使用规范化的路径）
        normalized_path = str(file_path.absolute())
        self.processed_files.add(normalized_path)
        
        return {"nodes": nodes_created, "relations": relations_created, "triples": total_triples}
    
    def build_from_dir(self, json_dir: Path, clear_first: bool = False):
        """从目录构建知识图谱"""
        # 查找JSON文件
        json_files = list(json_dir.rglob("*.json"))
        if not json_files:
            print(f"❌ 未找到JSON文件: {json_dir}")
            return
        
        print(f"📁 找到 {len(json_files)} 个文件")
        
        # 重置本次运行的统计
        self.new_nodes_this_run = set()
        self.new_relations_this_run = set()
        
        # 清空数据库（如果需要）
        if clear_first:
            self.clear_database()
        else:
            # 增量更新时，加载缓存
            self.load_cache_from_db()
        
        # 创建分类体系
        self.create_hierarchy()
        
        # 处理文件
        total_stats = {"nodes": 0, "relations": 0, "files_processed": 0}
        
        for json_file in json_files:
            print(f"\n处理: {json_file.name}")
            
            # 检查是否已处理过（规范化路径）
            normalized_path = str(json_file.absolute())
            if normalized_path in self.processed_files:
                print(f"  ⏭️  文件已处理过，跳过")
                continue
            
            stats = self.process_json_file(json_file)
            total_stats["nodes"] += stats["nodes"]
            total_stats["relations"] += stats["relations"]
            total_stats["files_processed"] += 1
            print(f"  → 新增 {stats['nodes']} 节点, {stats['relations']} 关系")
        
        # 保存统计
        self.save_statistics()
        
        print(f"\n{'='*60}")
        print(f"✅ 构建完成!")
        print(f"📊 本次运行统计:")
        print(f"   新增节点: {total_stats['nodes']}")
        print(f"   新增关系: {total_stats['relations']}")
        print(f"   处理文件数: {total_stats['files_processed']}")
        print(f"   跳过文件数: {len(json_files) - total_stats['files_processed']}")
        print(f"\n📊 累计统计:")
        print(f"   总节点数: {self.stats['total_nodes']}")
        print(f"   总关系数: {self.stats['total_relations']}")
    
    # ==================== 数据库操作 ====================
    def clear_database(self):
        """清空数据库"""
        confirm = input("⚠️  清空数据库？输入 'YES' 确认: ")
        if confirm != "YES":
            print("操作取消")
            return
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        # 重置状态
        self.entity_cache = {}
        self.processed_files = set()
        self.new_nodes_this_run = set()
        self.new_relations_this_run = set()
        self.stats = {
            "total_nodes": 0,
            "total_relations": 0,
            "nodes_by_type": defaultdict(int),
            "nodes_by_category": defaultdict(int),
            "nodes_by_subcategory": defaultdict(int),
            "relations_by_type": defaultdict(int)
        }
        
        print("✅ 数据库已清空")
    
    def load_cache_from_db(self):
        """从数据库加载缓存"""
        with self.driver.session() as session:
            # 加载节点
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.node_id as node_id, n.name as name, n.type as type,
                       n.category as category, n.subcategory as subcategory
            """)
            
            node_count = 0
            for record in result:
                node_id = record["node_id"]
                self.entity_cache[node_id] = {
                    "name": record["name"],
                    "type": record["type"],
                    "category": record.get("category", ""),
                    "subcategory": record.get("subcategory", "")
                }
                node_count += 1
            
            # 加载统计
            node_stats = session.run("""
                MATCH (n:Entity)
                RETURN n.type as type, COUNT(n) as count
            """)
            
            # 重置统计
            self.stats = {
                "total_nodes": node_count,
                "total_relations": 0,
                "nodes_by_type": defaultdict(int),
                "nodes_by_category": defaultdict(int),
                "nodes_by_subcategory": defaultdict(int),
                "relations_by_type": defaultdict(int)
            }
            
            for record in node_stats:
                entity_type = record["type"]
                count = record["count"]
                self.stats["nodes_by_type"][entity_type] = count
            
            # 加载关系统计
            rel_stats = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, COUNT(r) as count
            """)
            
            total_relations = 0
            for record in rel_stats:
                rel_type = record["rel_type"]
                count = record["count"]
                self.stats["relations_by_type"][rel_type] = count
                total_relations += count
            
            self.stats["total_relations"] = total_relations
            
            print(f"✅ 从数据库加载 {node_count} 个节点, {total_relations} 个关系到缓存")
    
    def update_single_file(self, file_path: Path):
        """增量更新单个文件"""
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return
        
        # 规范化路径
        normalized_path = str(file_path.absolute())
        
        # 检查是否已处理过
        if normalized_path in self.processed_files:
            print(f"⏭️  文件已处理过: {file_path.name}")
            return
        
        # 重置本次运行的统计
        self.new_nodes_this_run = set()
        self.new_relations_this_run = set()
        
        # 加载缓存
        self.load_cache_from_db()
        
        print(f"\n🔄 增量更新: {file_path.name}")
        stats = self.process_json_file(file_path)
        
        # 保存统计
        self.save_statistics()
        
        print(f"\n✅ 更新完成:")
        print(f"   新增节点: {stats['nodes']}")
        print(f"   新增关系: {stats['relations']}")
        print(f"   处理三元组: {stats['triples']}")
        print(f"   累计总节点: {self.stats['total_nodes']}")
        print(f"   累计总关系: {self.stats['total_relations']}")
    
    # ==================== 查询统计 ====================
    def query_statistics(self, save_path: Optional[Path] = None):
        """查询统计信息"""
        print("\n📊 知识图谱统计信息")
        print("=" * 60)
        
        with self.driver.session() as session:
            # 1. 总体统计
            result = session.run("""
                MATCH (n:Entity)
                RETURN 
                    COUNT(n) as total_nodes,
                    COUNT(DISTINCT n.type) as node_types,
                    COUNT(DISTINCT n.category) as categories,
                    COUNT(DISTINCT n.subcategory) as subcategories
            """)
            total = result.single()
            
            print(f"总节点数: {total['total_nodes']}")
            print(f"节点类型数: {total['node_types']}")
            print(f"主分类数: {total['categories']}")
            print(f"子分类数: {total['subcategories']}")
            
            # 2. 关系统计
            result = session.run("""
                MATCH ()-[r]->()
                RETURN 
                    COUNT(r) as total_relations,
                    COUNT(DISTINCT type(r)) as relation_types
            """)
            rels = result.single()
            print(f"总关系数: {rels['total_relations']}")
            print(f"关系类型数: {rels['relation_types']}")
            
            # 3. 分类分布
            print("\n📈 分类分布:")
            result = session.run("""
                MATCH (n:Entity)
                WHERE n.category IS NOT NULL AND n.subcategory IS NOT NULL
                RETURN 
                    n.category as category,
                    n.subcategory as subcategory,
                    COUNT(n) as count
                ORDER BY category, count DESC
            """)
            
            categories = defaultdict(list)
            for record in result:
                category = record["category"]
                subcategory = record["subcategory"]
                count = record["count"]
                categories[category].append((subcategory, count))
            
            for category, subcats in categories.items():
                print(f"\n  {category}:")
                for subcat, count in subcats:
                    print(f"    {subcat}: {count} 节点")
            
            # 4. 节点类型分布（前10）
            print("\n🔤 节点类型分布（前10）:")
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.type as type, COUNT(n) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            # 5. 关系类型分布（前10）
            print("\n🔗 关系类型分布（前10）:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, COUNT(r) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            # 6. 节点标签统计（修复语法错误）
            print("\n🏷️  节点标签统计:")
            result = session.run("""
                MATCH (n)
                UNWIND labels(n) as label
                WITH label
                WHERE label <> 'Entity'
                RETURN label, COUNT(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['label']}: {record['count']}")
            
            # 7. 文件处理统计
            print(f"\n📄 文件处理统计:")
            print(f"   已处理文件数: {len(self.processed_files)}")
            
            # 保存结果
            if save_path:
                self._save_query_results(total, rels, categories, save_path)
    
    def _save_query_results(self, total, rels, categories, save_path: Path):
        """保存查询结果到文件"""
        results = {
            "query_time": datetime.now().isoformat(),
            "total_nodes": total["total_nodes"],
            "node_types": total["node_types"],
            "categories": total["categories"],
            "subcategories": total["subcategories"],
            "total_relations": rels["total_relations"],
            "relation_types": rels["relation_types"],
            "category_distribution": {
                cat: dict(subcats) for cat, subcats in categories.items()
            },
            "entity_cache_size": len(self.entity_cache),
            "processed_files_count": len(self.processed_files)
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 统计结果已保存到: {save_path}")
    
    # ==================== 状态管理 ====================
    def save_statistics(self):
        """保存统计信息"""
        cache_dir = Path("/home/zzm/Project_1/kg-hk/2_kg_construction/kg_statistics")
        cache_dir.mkdir(exist_ok=True)
        
        # 保存统计
        stats_path = cache_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_nodes": self.stats["total_nodes"],
                "total_relations": self.stats["total_relations"],
                "nodes_by_type": dict(self.stats["nodes_by_type"]),
                "nodes_by_category": dict(self.stats["nodes_by_category"]),
                "nodes_by_subcategory": dict(self.stats["nodes_by_subcategory"]),
                "relations_by_type": dict(self.stats["relations_by_type"]),
                "processed_files": list(self.processed_files),
                "cache_size": len(self.entity_cache)
            }, f, ensure_ascii=False, indent=2)
    
    def load_statistics(self):
        """加载统计信息"""
        stats_path = Path("/home/zzm/Project_1/kg-hk/2_kg_construction/kg_statistics") / "statistics.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 恢复统计
                self.stats["total_nodes"] = data.get("total_nodes", 0)
                self.stats["total_relations"] = data.get("total_relations", 0)
                
                # 恢复defaultdict
                for key in ["nodes_by_type", "nodes_by_category", "nodes_by_subcategory", "relations_by_type"]:
                    if key in data:
                        self.stats[key] = defaultdict(int, data[key])
                
                # 恢复已处理文件
                self.processed_files = set(data.get("processed_files", []))
                
                print(f"✅ 已加载缓存状态: {len(self.processed_files)} 个已处理文件")
                
            except Exception as e:
                print(f"⚠️  加载缓存失败: {e}")
    
    def close(self):
        """关闭连接"""
        self.save_statistics()
        self.driver.close()
        print("🔌 连接已关闭")


# ==================== 主程序 ====================
def main():
    """主程序"""
    print("=" * 60)
    print("分层知识图谱管理系统")
    print("=" * 60)
    
    # 配置路径
    DATA_DIR = Path(r"/home/zzm/Project_1/kg-hk/1_extract_data/kg_data/GB")
    OUTPUT_DIR = Path(r"/home/zzm/Project_1/kg-hk/2_kg_construction/kg_statistics")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 初始化
    kg = SimplifiedKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # 加载缓存
        kg.load_statistics()
        
        # 用户选择
        print("\n请选择操作:")
        print("1. 全新构建（清空现有数据）")
        print("2. 增量更新单个文件")
        print("3. 增量更新目录")
        print("4. 查询统计信息")
        
        choice = input("\n请输入选择 (1/2/3/4): ").strip()
        
        if choice == "1":
            # 全新构建
            print("\n全新构建图谱...")
            kg.build_from_dir(DATA_DIR, clear_first=True)
            
        elif choice == "2":
            # 增量更新单个文件
            file_name = input("请输入文件名（包含路径）: ").strip()
            file_path = Path(file_name)
            if not file_path.is_absolute():
                file_path = DATA_DIR / file_name
            
            kg.update_single_file(file_path)
            
        elif choice == "3":
            # 增量更新目录
            dir_name = input("请输入文件夹名（包含路径）: ").strip()
            dir_path = Path(dir_name)
            if not dir_path.is_absolute():
                dir_path = DATA_DIR / dir_name
            
            kg.build_from_dir(dir_path, clear_first=False)
            
        elif choice == "4":
            # 查询统计
            save_path = OUTPUT_DIR / f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            kg.query_statistics(save_path=save_path)
            
        else:
            print("❌ 无效选择")
            
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        kg.close()


if __name__ == "__main__":
    main()