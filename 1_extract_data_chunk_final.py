import json
from pathlib import Path
from openai import OpenAI
import re
from tqdm import tqdm
import math

# ========== 配置 vLLM ==========
VLLM_BASE_URL = "URL"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "MODEL_NAME"

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

# ========== 路径配置 ==========
INPUT_ROOT = Path(r"/home/zzm/Project_1/kg-hk/1_extract_data/empty")
OUTPUT_ROOT = Path(r"/home/zzm/Project_1/kg-hk/1_extract_data/kg_data")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ========== 处理参数配置 ==========
CHUNK_SIZE = 1500  # 每块5000字符 # 6000
CHUNK_OVERLAP = 100  # 重叠100字符
START_FILE_INDEX = 16  # 从第几个文件开始处理（从1开始计数）
END_FILE_INDEX = 35   # 到第几个文件结束（包含）

def split_text_into_chunks(text: str, chunk_size: int = 5000, overlap: int = 100) -> list:
    """将文本分割成固定大小的块，带有重叠部分"""
    chunks = []
    start = 0
    
    while start < len(text):
        # 计算块的结束位置
        end = start + chunk_size
        
        # 如果块结束在句子中间，尽量在句号、分号或换行处断开
        if end < len(text):
            # 寻找合适的断点
            for break_char in ['\n', '。', '；', '. ', '; ']:
                break_pos = text.rfind(break_char, start, end)
                if break_pos != -1 and break_pos > start + chunk_size // 2:
                    end = break_pos + 1
                    break
        
        # 获取当前块
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 移动起始位置，考虑重叠
        start = end - overlap
        
        # 防止无限循环
        if start <= 0:
            break
            
    return chunks

def extract_standard_id(md_filename: str):
    """从文件名提取标准ID"""
    return Path(md_filename).stem.replace("_", " ").replace("+", " ")

# 更新后的prompt，要求输出JSON数组格式
build_prompt = f"""
你是航空航天制造知识抽取专家。请从markdown标准文档中抽取结构化三元组知识。

【任务】
1. 文档元信息：标准编号、标题、归口/起草单位
2. 技术实体关系：部件、材料、工艺、参数等关系

【实体类型】(type字段标注):
- Standard(标准文档), Title(标题), Component(结构部件), Material(材料), Process(工艺)
- Equipment(设备), Parameter(参数), Value(值), Organization(机构)
- Defect(缺陷), Requirement(要求), Test(试验)

【关系类型】:
- 层级: part_of(A是B部分), is_a(A是B类型)
- 属性: has_parameter(A有参数B), parameter_value(参数A值为B)
- 约束: must_follow(A遵循B), reference_to(A参考B), applicable_to(A适用于B)
- 因果: cause(A导致B), prevent(A防止B)
- 时序: precede(A在B前), follow(A在B后)
- 验证: verify_by(A通过B验证), test_method(A测试方法为B)
- 文档: title(A标题为B), issued_by(A由B发布), drafted_by(A由B起草), replace(A替代B), reference(A引用B)

【规则】:
1. 主实体确定: 标准编号或文档标题
2. 代词处理: "本文件"/"本标准"等代词替换为主实体
3. 逐句分析: 每句话独立抽取
4. 表格公式: 抽取关键信息
5. 关系强度: 根据"必须"/"应"/"宜"/"可"判断

【输出格式】JSON数组:
[
  {{
    "head": {{"name": "实体名", "type": "实体类型"}},
    "relation": "关系类型",
    "tail": {{"name": "实体名", "type": "实体类型"}},
    "paragraph": "原文句子",
    "source": "",
    "confidence": 0.0-1.0
  }}
]

【示例】:
输入: "HB 8768-2025《民用飞机复合材料雷达罩修理通用要求》发布。本标准规定了复合材料雷达罩修理要求。固化升温速率不得超过1.5°C/min。"

输出:
[
  {{
    "head": {{"name": "HB 8768-2025", "type": "Standard"}},
    "relation": "title",
    "tail": {{"name": "民用飞机复合材料雷达罩修理通用要求", "type": "Title"}},
    "paragraph": "HB 8768-2025《民用飞机复合材料雷达罩修理通用要求》发布。",
    "source": "",
    "confidence": 1.0
  }},
  {{
    "head": {{"name": "HB 8768-2025", "type": "Standard"}},
    "relation": "applicable_to",
    "tail": {{"name": "复合材料雷达罩", "type": "Component"}},
    "paragraph": "本标准规定了复合材料雷达罩修理要求。",
    "source": "",
    "confidence": 0.9
  }}
]

【重要】:
1. 只输出JSON数组
2. 无三元组时输出: []
3. source留空
4. 置信度基于关系明确性

请从以下文档抽取三元组:
/no_think
"""

def call_vllm(text_chunk: str, md_filename: str):
    """调用vLLM API处理文本块"""
    try:
        # 准备完整提示
        full_prompt = build_prompt + "\n\n文档块内容：\n" + text_chunk
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个结构化信息抽取专家。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0,  # 降低温度以获得更稳定输出
            max_tokens=12000
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # 清理输出，移除可能的多余文本
        cleaned = re.sub(r'(\<think\>.*?\<\/think\>)', '', raw_output, flags=re.DOTALL | re.IGNORECASE)
        cleaned = cleaned.strip()
        print(f"📝 模型输出片段: {cleaned}...")
        
        # 查找JSON数组的开始和结束
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        
        if start == -1 or end == -1 or start >= end:
            print(f"⚠️  无法在输出中找到有效的JSON数组，输出长度：{len(cleaned)}")
            return []
        
        json_str = cleaned[start:end+1]
        
        # 尝试解析JSON
        try:
            triples = json.loads(json_str)
            if not isinstance(triples, list):
                print(f"⚠️  解析结果不是列表，类型：{type(triples)}")
                return []
            
            # 处理每个三元组，添加source字段
            processed_triples = []
            for triple in triples:
                if isinstance(triple, dict):
                    # 确保所有必需字段都存在
                    processed_triple = {
                        "head": triple.get("head", {"name": "", "type": ""}),
                        "relation": triple.get("relation", ""),
                        "tail": triple.get("tail", {"name": "", "type": ""}),
                        "paragraph": triple.get("paragraph", ""),
                        "source": md_filename,  # 添加文档名称
                        "confidence": triple.get("confidence", 0.5)
                    }
                    processed_triples.append(processed_triple)
            
            return processed_triples
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            print(f"JSON字符串片段: {json_str[:200]}...")
            return []
            
    except Exception as e:
        print(f"⚠️  调用vLLM API时出错: {e}")
        return []

def process_single_file(md_path: Path, output_path: Path):
    """处理单个文件"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ 读取失败: {md_path} - {e}")
        return 0
    
    if not text.strip():
        print(f"⚠️ 跳过空文件: {md_path.name}")
        return 0
    
    # 分块处理
    chunks = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"📄 文件 '{md_path.name}' 分割为 {len(chunks)} 块，每块为{len(text)/len(chunks)}字符")
    
    all_triples = []
    
    # 处理每个块
    for i, chunk in enumerate(tqdm(chunks, desc=f"处理 {md_path.name}", leave=False)):
        # 处理转义字符
        cleaned_chunk = chunk.replace('\\', '\\\\')
        
        # 调用模型
        triples = call_vllm(cleaned_chunk, md_path.name)
        
        # 添加当前块的信息（可选，用于调试）
        for triple in triples:
            triple["chunk_index"] = i
            triple["chunk_size"] = len(chunk)
        
        all_triples.extend(triples)
    
    # 去重：基于head、relation、tail和paragraph去重
    seen = set()
    unique_triples = []
    
    for triple in all_triples:
        # 创建唯一标识符
        head_name = triple.get("head", {}).get("name", "") if isinstance(triple.get("head"), dict) else str(triple.get("head", ""))
        tail_name = triple.get("tail", {}).get("name", "") if isinstance(triple.get("tail"), dict) else str(triple.get("tail", ""))
        key = (head_name, 
               triple.get("relation", ""), 
               tail_name,
               triple.get("paragraph", ""))
        
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON文件（推荐使用JSON，因为结构统一）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_triples, f, ensure_ascii=False, indent=2)
    
    return len(unique_triples)

def main():
    """主函数"""
    # 收集所有 .md 文件
    md_files_info = []
    
    # 获取所有子文件夹
    subfolders = [f for f in INPUT_ROOT.iterdir() if f.is_dir()]
    
    if not subfolders:
        print("❌ 未找到任何子文件夹，尝试直接查找md文件...")
        # 如果没有子文件夹，直接在根目录查找
        md_files = list(INPUT_ROOT.glob("*.md"))
        for md_file in md_files:
            out_file = OUTPUT_ROOT / (md_file.stem + ".json")
            md_files_info.append((md_file, out_file))
    else:
        # 遍历所有子文件夹
        for folder in subfolders:
            md_files = list(folder.glob("*.md"))
            for md_file in md_files:
                # 保持文件夹结构
                relative_path = md_file.relative_to(INPUT_ROOT)
                out_file = OUTPUT_ROOT / relative_path.with_suffix('.json')
                md_files_info.append((md_file, out_file))
    
    if not md_files_info:
        print("❌ 未找到任何 .md 文件！")
        return
    
    # 按文件名排序，确保处理顺序一致
    md_files_info.sort(key=lambda x: str(x[0]))
    
    total_files = len(md_files_info)
    print(f"📁 共发现 {total_files} 个 .md 文件")
    
    # 应用文件范围限制（转换为0-based索引）
    start_idx = max(0, START_FILE_INDEX - 1)
    end_idx = min(total_files, END_FILE_INDEX)
    
    files_to_process = md_files_info[start_idx:end_idx]
    
    print(f"🔧 将处理从第{START_FILE_INDEX}到第{END_FILE_INDEX}个文件，共 {len(files_to_process)} 个文件...")
    
    total_triples = 0
    successful = 0
    
    # 处理文件
    for i, (md_file, out_file) in enumerate(tqdm(files_to_process, desc="总体进度"), 1):
        try:
            print(f"\n{'='*60}")
            print(f"🔍 处理第 {i+start_idx}/{total_files} 个文件: {md_file.name}")
            
            count = process_single_file(md_file, out_file)
            successful += 1
            total_triples += count
            
            print(f"✅ {md_file.name} → 提取 {count} 个三元组")
            print(f"💾 保存至: {out_file}")
            
        except Exception as e:
            print(f"❌ {md_file.name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"🎉 处理完成！")
    print(f"📊 统计:")
    print(f"   成功处理: {successful}/{len(files_to_process)} 个文件")
    print(f"   总共提取: {total_triples} 个三元组")
    print(f"   平均每个文件: {total_triples/max(1, successful):.1f} 个三元组")
    print(f"📁 结果保存至: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()