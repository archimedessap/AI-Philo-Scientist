#!/bin/bash
# run_with_notification.sh - 带完成通知的理论生成运行脚本
# =========================================================
# 
# 使用方法：
#   ./run_with_notification.sh [fast|normal|full]
#
# 参数说明：
#   fast   - 快速模式（1代，少量理论）
#   normal - 标准模式（2代，中等数量）
#   full   - 完整模式（3代，大量理论）

# 设置默认模式
MODE=${1:-fast}

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output_notified_${MODE}_${TIMESTAMP}"
LOG_FILE="output_${MODE}_run_${TIMESTAMP}.log"

echo "=========================================="
echo "🚀 UniversalTheoryGen 运行脚本"
echo "=========================================="
echo "模式: ${MODE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

# 根据模式设置参数
case $MODE in
    fast)
        GENERATIONS=1
        PAIRS=2
        VARIANTS=1
        MODEL="gemini-2.5-flash"
        echo "使用快速参数..."
        ;;
    normal)
        GENERATIONS=2
        PAIRS=3
        VARIANTS=2
        MODEL="gemini-2.5-flash"
        echo "使用标准参数..."
        ;;
    full)
        GENERATIONS=3
        PAIRS=5
        VARIANTS=3
        MODEL="gemini-2.5-pro"
        echo "使用完整参数..."
        ;;
    *)
        echo "未知模式: $MODE"
        echo "使用方法: $0 [fast|normal|full]"
        exit 1
        ;;
esac

# 构建运行命令
RUN_CMD="python run_clean_evolution.py \
    --initial_theories_dir data/theories_v2.1 \
    --output_root ${OUTPUT_DIR} \
    --experiment_dir demo/experiments \
    --max_generations ${GENERATIONS} \
    --synthesis_method unified \
    --use_raw_literature \
    --literature_concepts_dir data/enhanced_concepts \
    --synthesis_model_source google \
    --synthesis_model_name ${MODEL} \
    --evaluation_model_source google \
    --evaluation_model_name ${MODEL} \
    --max_pairs_to_analyze ${PAIRS} \
    --variants_per_contradiction ${VARIANTS} \
    --promotion_min_score 0.3 \
    --top_n_survivors 3 \
    --role_success_threshold 0.3 \
    --use_instrument_correction"

# 定义通知函数
notify_completion() {
    local status=$1
    local message=""
    local title="UniversalTheoryGen"
    
    if [ $status -eq 0 ]; then
        message="✅ 理论生成完成！模式: ${MODE}"
        echo "=========================================="
        echo "$message"
        echo "输出目录: ${OUTPUT_DIR}"
        echo "日志文件: ${LOG_FILE}"
        echo "=========================================="
    else
        message="❌ 运行失败！请检查日志: ${LOG_FILE}"
        echo "=========================================="
        echo "$message"
        echo "=========================================="
    fi
    
    # 终端输出
    echo "$message"
    
    # macOS 系统通知
    if command -v osascript &> /dev/null; then
        osascript -e "display notification \"$message\" with title \"$title\""
    fi
    
    # 播放系统声音（可选）
    if [ $status -eq 0 ]; then
        # 成功音效
        afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
    else
        # 失败音效
        afplay /System/Library/Sounds/Basso.aiff 2>/dev/null || true
    fi
}

# 在后台运行主命令，并在完成后发送通知
echo "开始运行..."
echo "命令: ${RUN_CMD}"
echo ""
echo "日志输出到: ${LOG_FILE}"
echo "你可以使用以下命令查看进度："
echo "  tail -f ${LOG_FILE}"
echo ""

# 使用 nohup 在后台运行，完成后发送通知
nohup bash -c "${RUN_CMD} && notify_completion 0 || notify_completion 1" > "${LOG_FILE}" 2>&1 &

# 获取后台进程PID
BG_PID=$!
echo "后台进程 PID: ${BG_PID}"
echo ""
echo "使用以下命令检查状态："
echo "  ps -p ${BG_PID}"
echo "  kill ${BG_PID}  # 如需停止"
echo ""

# 导出通知函数以便后台进程可以使用
export -f notify_completion
export MODE OUTPUT_DIR LOG_FILE