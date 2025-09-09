#!/bin/bash
# run_full_pipeline.sh - 完整的增强理论生成流程
# ================================================
# 
# 此脚本执行完整的理论生成流程：
# 1. 从原始文献提取概念和公式
# 2. 构建知识图谱
# 3. 运行增强的unified理论生成
# 4. 评估生成的理论
#
# 使用方法：
#   ./run_full_pipeline.sh [选项]
#
# 选项：
#   --skip-extraction    跳过概念提取（使用现有数据）
#   --test-mode         测试模式（处理较少文档）
#   --max-generations N  最大演进代数（默认3）

# 设置默认值
SKIP_EXTRACTION=false
TEST_MODE=false
MAX_GENERATIONS=3
MODEL_SOURCE="google"
MODEL_NAME="gemini-2.0-flash-exp"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        --max-generations)
            MAX_GENERATIONS="$2"
            shift 2
            ;;
        --model-source)
            MODEL_SOURCE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 [--skip-extraction] [--test-mode] [--max-generations N]"
            exit 1
            ;;
    esac
done

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output_enhanced_full_${TIMESTAMP}"

echo "=========================================="
echo "🚀 完整增强理论生成流程"
echo "=========================================="
echo "时间戳: ${TIMESTAMP}"
echo "输出目录: ${OUTPUT_DIR}"
echo "跳过提取: ${SKIP_EXTRACTION}"
echo "测试模式: ${TEST_MODE}"
echo "最大代数: ${MAX_GENERATIONS}"
echo "模型: ${MODEL_SOURCE}/${MODEL_NAME}"
echo "=========================================="

# 阶段1：概念和公式提取
if [ "$SKIP_EXTRACTION" = false ]; then
    echo ""
    echo "=========================================="
    echo "📚 阶段1: 从文献提取概念和公式"
    echo "=========================================="
    
    PREP_CMD="python prepare_enhanced_concepts.py --model_source ${MODEL_SOURCE} --model_name ${MODEL_NAME}"
    
    if [ "$TEST_MODE" = true ]; then
        PREP_CMD="${PREP_CMD} --test_mode"
    fi
    
    echo "执行命令: ${PREP_CMD}"
    eval ${PREP_CMD}
    
    if [ $? -ne 0 ]; then
        echo "❌ 概念提取失败"
        exit 1
    fi
    
    echo "✅ 概念和公式提取完成"
else
    echo ""
    echo "⏭️  跳过概念提取阶段"
fi

# 阶段2：理论生成和演进
echo ""
echo "=========================================="
echo "🧬 阶段2: 运行增强的unified理论生成"
echo "=========================================="

# 构建演进命令
EVO_CMD="python run_clean_evolution.py \
    --initial_theories_dir data/theories_v2.1 \
    --output_root ${OUTPUT_DIR} \
    --experiment_dir demo/experiments \
    --synthesis_method unified \
    --max_generations ${MAX_GENERATIONS} \
    --max_pairs_to_analyze 5 \
    --variants_per_contradiction 2 \
    --synthesis_model_source ${MODEL_SOURCE} \
    --synthesis_model_name ${MODEL_NAME} \
    --evaluation_model_source ${MODEL_SOURCE} \
    --evaluation_model_name ${MODEL_NAME} \
    --top_n_survivors 3 \
    --promotion_min_score 0.6 \
    --use_raw_literature \
    --literature_concepts_dir data/enhanced_concepts"

echo "执行命令: ${EVO_CMD}"
eval ${EVO_CMD}

if [ $? -ne 0 ]; then
    echo "❌ 理论生成失败"
    exit 1
fi

echo "✅ 理论生成和演进完成"

# 阶段3：生成总结报告
echo ""
echo "=========================================="
echo "📊 阶段3: 生成总结报告"
echo "=========================================="

# 查找最新的运行目录
RUN_DIR=$(find ${OUTPUT_DIR} -name "run_*" -type d | sort | tail -1)

if [ -z "$RUN_DIR" ]; then
    echo "❌ 未找到运行目录"
    exit 1
fi

echo "运行目录: ${RUN_DIR}"

# 生成演进摘要
if [ -f "${RUN_DIR}/run_manifest.json" ]; then
    echo ""
    echo "📈 演进统计："
    python -c "
import json
with open('${RUN_DIR}/run_manifest.json', 'r') as f:
    manifest = json.load(f)
    stats = manifest.get('statistics', {})
    print(f'  总理论数: {stats.get(\"total_theories\", 0)}')
    print(f'  晋级理论数: {stats.get(\"promoted_theories\", 0)}')
    print(f'  最高分数: {stats.get(\"best_score\", 0):.3f}')
    print(f'  平均分数: {stats.get(\"avg_score\", 0):.3f}')
"
fi

# 列出生成的理论
echo ""
echo "🎯 生成的理论："
find ${RUN_DIR} -name "*.json" -path "*/eval_ready_theories/*" | while read f; do
    theory_name=$(python -c "import json; print(json.load(open('$f'))['name'])" 2>/dev/null || echo "未知")
    echo "  - ${theory_name}"
done

# 检查是否有基准对比报告
BENCHMARK_DIR="${RUN_DIR}/benchmark_comparison"
if [ -d "$BENCHMARK_DIR" ]; then
    echo ""
    echo "📊 基准对比报告: ${BENCHMARK_DIR}"
fi

echo ""
echo "=========================================="
echo "✅ 完整流程执行成功！"
echo "=========================================="
echo "所有结果保存在: ${OUTPUT_DIR}"
echo ""
echo "后续步骤："
echo "1. 查看详细结果: ${RUN_DIR}"
echo "2. 查看评估报告: ${RUN_DIR}/benchmark_comparison"
echo "3. 查看生成的理论: ${RUN_DIR}/generation_*/synthesis/eval_ready_theories/"
echo ""

# 可选：启动可视化服务器
read -p "是否启动可视化服务器查看结果？[y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "启动可视化服务器..."
    python -m http.server 8000 --directory ${OUTPUT_DIR} &
    SERVER_PID=$!
    echo "可视化服务器已启动，PID: ${SERVER_PID}"
    echo "请在浏览器中访问: http://localhost:8000"
    echo "按Ctrl+C停止服务器"
    wait ${SERVER_PID}
fi
