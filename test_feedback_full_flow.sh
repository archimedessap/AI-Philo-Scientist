#!/bin/bash
# test_feedback_full_flow.sh - 完整的评估反馈流程测试
# ====================================================
# 这个脚本演示了如何使用所有主要参数运行完整的理论生成和反馈改进流程

echo "====================================================="
echo "UniversalTheoryGen 完整评估反馈流程测试"
echo "====================================================="
echo ""

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output_feedback_test_${TIMESTAMP}"

echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 方案1: 使用 run_clean_evolution.py 运行完整演进流程（包含反馈）
echo "========== 方案1: 完整演进流程（推荐） =========="
echo ""
echo "这个命令将运行多代演进，每代都包含评估和反馈改进："
echo ""

cat << 'EOF'
python run_clean_evolution.py \
    --initial_theories_dir data/theories_test \
    --output_root output_evolution_with_feedback \
    --max_generations 2 \
    --synthesis_method unified \
    --use_raw_literature \
    --literature_concepts_dir data/enhanced_concepts \
    --synthesis_model_source google \
    --synthesis_model_name gemini-2.5-flash \
    --evaluation_model_source google \
    --evaluation_model_name gemini-2.5-flash \
    --max_pairs_to_analyze 3 \
    --variants_per_contradiction 2 \
    --promotion_min_score 0.3 \
    --top_n_survivors 3 \
    --run_role_evaluation \
    --role_evaluation_threshold 0.4 \
    --skip_experiments_for_standard_qm \
    --test_mode
EOF

echo ""
echo "参数说明："
echo "  --max_generations 2: 运行2代演进"
echo "  --synthesis_method unified: 使用统一生成方法（支持文献概念）"
echo "  --use_raw_literature: 使用原始文献概念增强生成"
echo "  --run_role_evaluation: 启用三角色评估"
echo "  --skip_experiments_for_standard_qm: 跳过标准QM理论的实验评估"
echo "  --test_mode: 测试模式，减少生成数量"
echo ""

# 方案2: 使用 full_evaluation_feedback_loop.py 单独运行反馈循环
echo "========== 方案2: 单独的反馈循环测试 =========="
echo ""
echo "如果你已有理论文件，可以单独运行反馈改进循环："
echo ""

cat << 'EOF'
python full_evaluation_feedback_loop.py \
    --theory data/theories_test/T_MW_many-worlds_(everett)_interpretation.json \
    --output_dir output_feedback_loop_test \
    --iterations 2 \
    --model_source google \
    --model_name gemini-2.5-flash \
    --experiment_dir demo/experiments \
    --include_experiments \
    --role_threshold 0.3
EOF

echo ""
echo "参数说明："
echo "  --theory: 输入理论文件路径"
echo "  --iterations 2: 运行2轮反馈改进"
echo "  --include_experiments: 包含实验评估"
echo "  --role_threshold 0.3: 角色评估的及格线"
echo ""

# 方案3: 简化的角色评估反馈（不含实验）
echo "========== 方案3: 仅角色评估的快速反馈 =========="
echo ""
echo "更快的测试，只使用角色评估："
echo ""

cat << 'EOF'
python simple_feedback_loop_v2.py \
    --theory data/theories_test/T_CPH_copenhagen_interpretation.json \
    --output output_role_feedback_test \
    --iterations 1 \
    --model_source google \
    --model_name gemini-2.5-flash
EOF

echo ""
echo "参数说明："
echo "  这是最快的反馈测试，只运行角色评估"
echo "  适合快速验证反馈机制是否正常工作"
echo ""

# 方案4: 带反馈感知的理论生成
echo "========== 方案4: 反馈感知的理论生成 =========="
echo ""
echo "基于先前评估结果生成新理论："
echo ""

cat << 'EOF'
# 首先需要有评估结果目录
python run_clean_evolution.py \
    --initial_theories_dir data/theories_test \
    --output_root output_feedback_aware \
    --max_generations 1 \
    --synthesis_method feedback_aware \
    --feedback_dir previous_evaluation_output \
    --synthesis_model_source google \
    --synthesis_model_name gemini-2.5-pro \
    --use_concept_weight_adjustment \
    --pattern_recognition_mode aggressive
EOF

echo ""
echo "参数说明："
echo "  --synthesis_method feedback_aware: 使用反馈感知生成器"
echo "  --feedback_dir: 包含先前评估结果的目录"
echo "  --use_concept_weight_adjustment: 根据反馈调整概念权重"
echo "  --pattern_recognition_mode aggressive: 积极识别反馈模式"
echo ""

# 准备测试
echo "========== 准备测试 =========="
echo ""
echo "运行前请确保："
echo "1. 已配置 .env 文件中的 API keys"
echo "2. 存在 data/theories_test 目录和理论文件"
echo "3. 如使用文献概念，需先运行："
echo "   python prepare_enhanced_concepts.py --literature_dir data/raw_literature"
echo ""
echo "建议先运行方案3进行快速测试，确认系统正常后再运行完整流程。"
echo ""
echo "选择一个方案，复制相应的命令到终端运行即可。"
echo "====================================================="