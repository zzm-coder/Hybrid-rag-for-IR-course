"""航空航天制造混合RAG系统 - 主程序入口"""
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import SystemConfig
from hybrid_rag_system import HybridRAGSystem
from evaluator import RAGEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aerospace_rag.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主程序"""
    try:
        # 1. 初始化配置
        config = SystemConfig()
        logger.info("系统配置加载完成")
        
        # 2. 选择运行模式并初始化混合RAG系统（支持消融实验）
        print("请选择运行模式:")
        print("  1) KG-only (仅使用知识图谱检索)")
        print("  2) Vector-only (仅使用向量检索)")
        print("  3) Both (混合，默认)")
        print("  4) Adaptive-RAG (自动路由)")
        mode_choice = input("请输入模式 (1/2/3/4, 回车为3): ").strip()

        if mode_choice == "1":
            rag_system = HybridRAGSystem(config, use_kg=True, use_vector=False)
            mode_name = "KG-only"
        elif mode_choice == "2":
            rag_system = HybridRAGSystem(config, use_kg=False, use_vector=True)
            mode_name = "Vector-only"
        elif mode_choice == "4":
            # 使用全新 Adaptive-RAG 系统（不修改原类）
            from compare_rag.Adapitve_rag.adaptive_rag_system import AdaptiveRAGSystem
            rag_system = AdaptiveRAGSystem(config)
            mode_name = "Adaptive-RAG"
        else:
            rag_system = HybridRAGSystem(config, use_kg=True, use_vector=True)
            mode_name = "Hybrid (Both)"

        logger.info(f"系统初始化完成: {mode_name}")
        
        # 3. 示例查询
        test_questions = [
            "HB 8766－2025标准的发布时间以及负责归口管理的单位分别是什么？",
            "雷达罩的定义是什么？",
            "试验实施单位在电性能试验过程中因仪器设备故障导致试验中断时，应按照怎样的逻辑顺序进行后续处理才能确保试验件安全和试验有效性？",
            "试验人员在计算功率反射时，在什么条件下应优先选择公式(9)而非公式(10)，这种选择背后的技术逻辑是什么？",
            "HB 8766－2025标准规定的试验有效性要求如何协同适航要求、人员资质要求以及设备管理要求，共同构建起雷达罩电性能试验的质量控制体系？"
        ]
        
        print("=== 航空航天制造混合RAG系统 ===")
        print("版本: 2.0 (简化路由版)")
        print("=" * 50)
        
        while True:
            print("\n请选择操作模式:")
            print("1. 运行示例查询")
            print("2. 输入自定义查询")
            print("3. 评估系统性能")
            print("4. 退出系统")
            print("5. 训练并评估 (训练检索器/重排序器，然后在 test 集评估)")
            
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == "1":
                run_example_queries(rag_system, test_questions)
            elif choice == "2":
                run_custom_query(rag_system)
            elif choice == "3":
                evaluate_system(rag_system)
            elif choice == "5":
                run_train_and_eval_interactive()
            elif choice == "4":
                break
            else:
                print("无效选择，请重新输入")
        
        # 4. 清理资源
        rag_system.close()
        logger.info("系统已关闭")
        
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        print(f"系统运行失败: {e}")

def run_example_queries(rag_system, questions):
    """运行示例查询"""
    print(f"\n开始运行 {len(questions)} 个示例查询...")
    
    for i, question in enumerate(questions):
        print(f"\n【示例 {i+1}】")
        print(f"问题: {question}")
        print("-" * 80)
        
        try:
            result = rag_system.process_query(question)
            
            if "error" not in result:
                print("✓ 查询成功")
                
                # 显示路由分析
                router_info = result.get("router_analysis", {})
                print(f"问题类型: {router_info.get('question_type', '未知')}")
                print(f"类型ID: {router_info.get('type_id', '未知')}")
                print(f"提取实体: {', '.join(router_info.get('entities', []))}")
                print(f"用户意图: {router_info.get('intent', '未知')}")
                
                # 显示答案
                answer = result.get("generation", {}).get("answer", "")
                print(f"\n答案: {answer}")
                
                # 显示检索统计
                retrieval = result.get("retrieval", {})
                print(f"\n检索统计:")
                print(f"  • KG三元组: {retrieval.get('kg_results', {}).get('triples_count', 0)}个")
                print(f"  • 向量检索: {len(retrieval.get('vector_results', []))}个")
                print(f"  • 重排序后: {len(retrieval.get('reranked_results', []))}个")
                
                # 显示性能
                perf = result.get("performance", {})
                print(f"\n性能:")
                print(f"  • 检索时间: {perf.get('retrieval_time', 0):.2f}秒")
                print(f"  • 生成时间: {perf.get('generation_time', 0):.2f}秒")
                print(f"  • 总时间: {perf.get('total_time', 0):.2f}秒")
            else:
                print(f"✗ 查询失败: {result['error']}")
                
        except Exception as e:
            print(f"✗ 处理失败: {e}")
    
    print(f"\n示例查询完成")

def run_custom_query(rag_system):
    """运行自定义查询"""
    print("\n请输入您的问题 (输入'退出'返回主菜单):")
    
    while True:
        question = input("\n> ").strip()
        
        if question.lower() in ['退出', 'exit', 'quit']:
            break
        
        if not question:
            print("问题不能为空，请重新输入")
            continue
        
        try:
            print("正在处理，请稍候...")
            result = rag_system.process_query(question)
            
            if "error" not in result:
                print("\n" + "=" * 80)
                
                # 显示路由分析
                router_info = result.get("router_analysis", {})
                print(f"路由分析:")
                print(f"  • 类型: {router_info.get('question_type', '未知')} (ID: {router_info.get('type_id', '未知')})")
                print(f"  • 实体: {', '.join(router_info.get('entities', [])) or '无'}")
                print(f"  • 意图: {router_info.get('intent', '未知')}")
                
                # 显示答案
                answer = result.get("generation", {}).get("answer", "")
                print(f"\n答案:")
                print(answer)
                
                # 显示引用
                citations = result.get("generation", {}).get("citations", [])
                if citations:
                    print(f"\n引用: {', '.join(citations)}")
                
                print("=" * 80)
            else:
                print(f"处理失败: {result['error']}")
                
        except Exception as e:
            print(f"处理失败: {e}")

def evaluate_system(rag_system):
    """评估系统性能"""
    print("\n=== 系统评估 ===")
    print("请选择要评估的数据集分割:")
    print("  1) train")
    print("  2) dev (验证集)")
    print("  3) test")
    print("  4) custom path")
    split_choice = input("请输入选择 (1/2/3/4, 回车为2): ").strip()

    base_splits_dir = "/home/zzm/Project_1/kg-hk/3_QA_creation/3_QA_data/split_datasets"
    if split_choice == "1":
        qa_dataset_path = f"{base_splits_dir}/train.json"
    elif split_choice == "3":
        qa_dataset_path = f"{base_splits_dir}/test.json"
    elif split_choice == "4":
        qa_dataset_path = input("请输入自定义 QA 数据集路径: ").strip()
    else:
        qa_dataset_path = f"{base_splits_dir}/dev.json"
    try:
        print(f"正在加载QA数据集: {qa_dataset_path}")
        evaluator = RAGEvaluator(qa_dataset_path)

        sample_size_input = input("请输入评估样本大小 (建议3-50，直接回车使用默认值3): ").strip()
        sample_size = int(sample_size_input) if sample_size_input else 3
        
        if sample_size <= 0:
            print("样本大小必须大于0，使用默认值3")
            sample_size = 3

        print(f"开始评估，样本大小: {sample_size}")
        print("评估可能需要一些时间，请耐心等待...")
        
        evaluation_result = evaluator.evaluate_system(rag_system, sample_size)
        
        print("\n" + "=" * 80)
        print("评估结果摘要:")
        print("=" * 80)
        
        print(f"\n检索性能:")
        # 文档级检索
        print(f"  • Hit@1: {evaluation_result.hit_at_1:.3f}")
        print(f"  • Hit@3: {evaluation_result.hit_at_3:.3f}")
        print(f"  • Hit@5: {evaluation_result.hit_at_5:.3f}")
        print(f"  • Recall@1: {evaluation_result.recall_at_1:.3f}")
        print(f"  • Recall@3: {evaluation_result.recall_at_3:.3f}")
        print(f"  • Recall@5: {evaluation_result.recall_at_5:.3f}")
        # 段落级检索
        print(f"  • Para Hit@1: {evaluation_result.para_hit_at_1:.3f}")
        print(f"  • Para Hit@5: {evaluation_result.para_hit_at_5:.3f}")
        print(f"  • Para Hit@10: {evaluation_result.para_hit_at_10:.3f}")
        print(f"  • Para Recall@1: {evaluation_result.para_recall_at_1:.3f}")
        print(f"  • Para Recall@5: {evaluation_result.para_recall_at_5:.3f}")
        print(f"  • Para Recall@10: {evaluation_result.para_recall_at_10:.3f}")
        print(f"  • MRR: {evaluation_result.mrr:.3f}")
        # 上下文质量
        print(f"\n上下文质量:")
        print(f"  • 精准率: {evaluation_result.context_precision:.3f}")
        print(f"  • 召回率: {evaluation_result.context_recall:.3f}")
        print(f"  • 相关性: {evaluation_result.context_relevance:.3f}")
        # 生成质量
        print(f"\n生成质量:")
        print(f"  • EM分数: {evaluation_result.em_score:.3f}")
        print(f"  • F1分数: {evaluation_result.f1_score:.3f}")
        print(f"  • 准确率: {evaluation_result.accuracy:.3f}")
        print(f"  • 忠实度: {evaluation_result.faithfulness:.3f}")
        print(f"  • 答案相关性: {evaluation_result.answer_relevance:.3f}")
        # 引用与幻觉
        print(f"  • 引用准确率: {evaluation_result.citation_accuracy:.3f}")
        print(f"  • 幻觉率: {evaluation_result.hallucination_rate:.3f}")
        # 性能指标
        print(f"\n性能指标:")
        print(f"  • 平均检索时间: {evaluation_result.retrieval_time:.3f}秒")
        print(f"  • 平均生成时间: {evaluation_result.generation_time:.3f}秒")
        
        print("\n详细评估结果已保存到 /home/zzm/Project_1/kg-hk/4_RAG_method/results/ 目录")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请检查QA数据集路径是否正确")
    except Exception as e:
        print(f"评估失败: {e}")


def run_train_and_eval_interactive():
    """交互式触发训练并评估脚本"""
    import subprocess, sys
    print("\n=== 训练并评估 ===")
    base = "/home/zzm/Project_1/kg-hk/3_QA_creation/3_QA_data/split_datasets"
    train_dir = input(f"训练集目录 (回车使用 {base}/train): ").strip() or f"{base}/train.json"
    val_dir = input(f"验证集目录 (回车使用 {base}/val): ").strip() or f"{base}/val.json"
    test_dir = input(f"测试集目录 (回车使用 {base}/test): ").strip() or f"{base}/test.json"
    epochs = input("训练轮数 epochs (回车为1): ").strip() or "1"
    batch = input("batch size (回车为16): ").strip() or "16"
    sample = input("评估样本大小 (回车为50): ").strip() or "50"
    out_model = input("输出模型目录 (回车为 /hdd1/checkpoints/sentence-transformers/zzm_sbert_finetuned): ").strip() or "/hdd1/checkpoints/sentence-transformers/zzm_sbert_finetuned"
    save_cfg = input("是否将训练后配置保存为 JSON (输入路径或默认): ").strip() or "/home/zzm/Project_1/kg-hk/4_RAG_method/Mix-RAG-v1/system_config_trained.json"

    cmd = [sys.executable, '/home/zzm/Project_1/kg-hk/4_RAG_method/Mix-RAG-v1/run_train_and_eval.py', '--train-dir', train_dir, '--val-dir', val_dir, '--test-dir', test_dir, '--out-model-dir', out_model, '--epochs', epochs, '--batch-size', batch, '--sample-size', sample]
    if save_cfg:
        cmd += ['--save-config', save_cfg]

    print("执行命令:", ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
        print("训练并评估完成，结果保存在 results 目录")
    except Exception as e:
        print(f"执行失败: {e}")

if __name__ == "__main__":
    main()