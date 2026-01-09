#!/bin/bash
# file: start_system_fixed.sh
# 修复版启动脚本

echo "=========================================="
echo "启动航空航天制造混合RAG系统（修复版）"
echo "=========================================="

# 设置环境变量
export PYTHONPATH="/home/zzm/Project_1/kg-hk/4_RAG_method:$PYTHONPATH"

echo ""
echo "1. 启动后端API服务..."
echo "   访问地址: http://localhost:8886"
echo "   API文档: http://localhost:8886/docs"
echo "   健康检查: http://localhost:8886/health"
echo ""

# 启动后端
cd /home/zzm/Project_1/kg-hk/5_Mix-RAG-show
python backend_api.py &

echo "等待后端启动（5秒）..."
sleep 5

echo ""
echo "2. 启动前端界面..."
echo "   访问地址: http://localhost:8887"
echo ""

# 启动前端
cd /home/zzm/Project_1/kg-hk/5_Mix-RAG-show
streamlit run frontend_app_enhanced.py --server.port 8887 --server.address 0.0.0.0


# # 手动操作
# # 终端1：启动后端
# cd /home/zzm/Project_1/kg-hk/5_Mix-RAG-show
# python backend_api.py

# # 终端2：启动前端
# cd /home/zzm/Project_1/kg-hk/5_Mix-RAG-show
# streamlit run frontend_app.py