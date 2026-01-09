import subprocess
from pathlib import Path

# 配置输入和输出目录（使用正斜杠或原始字符串均可）
INPUT_DIR = Path("/home/zzm/Project_1/kg-hk/0_mineru_pdf/data_pdf")
OUTPUT_DIR = Path("/home/zzm/Project_1/kg-hk/0_mineru_pdf/data_md")

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 遍历所有 PDF 文件
for pdf_path in INPUT_DIR.glob("*.pdf"):
    if pdf_path.is_file():
        print(f"Processing: {pdf_path}")
        # 调用 mineru 命令
        result = subprocess.run([
            "mineru",
            "-p", str(pdf_path),
            "-o", str(OUTPUT_DIR),
            "--source", "local"
        ], capture_output=True, text=True)

        # 可选：检查命令是否成功
        if result.returncode != 0:
            print(f"Error processing {pdf_path}: {result.stderr}")
        else:
            print(f"Successfully processed: {pdf_path}")