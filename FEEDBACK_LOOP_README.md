# 评估反馈循环系统

## 概述

本系统实现了一个简单但有效的评估反馈循环，将理论评估的结果反馈给生成器，从而产生改进的理论版本。

**重要说明**：系统支持两种评估模式：
1. **完整评估模式**（默认）：包括实验评估（5个标准量子实验）+ 三角色评估（物理学家、哲学家、数学家）
2. **仅角色评估模式**（快速模式）：只进行三角色评估，适合快速迭代

## 核心组件

### 1. SimpleFeedbackLoop (`simple_feedback_loop.py`)

基础版本的反馈循环（仅角色评估）：
- 三角色评估（物理学家、哲学家、数学家）
- 反馈提取
- 改进生成
- 迭代优化

### 2. ImprovedFeedbackLoop (`simple_feedback_loop_v2.py`)

改进版本的反馈循环（推荐使用）：
- 更准确的角色评估实现
- 更全面的反馈分类（数学、实验、概念等）
- 详细的改进报告
- 分数对比追踪

### 3. FeedbackAwareGenerator (`theory_generation/methods/feedback_aware_generator.py`)

基于反馈的理论生成器，特点：
- 分析评估反馈中的模式
- 动态调整概念权重
- 生成针对性改进的理论

## 使用方法

### 方法1：运行完整的反馈循环

```bash
# 对已有理论进行反馈循环优化
python simple_feedback_loop.py \
    --theory output_clean_evolution/run_xxx/generation_0/synthesis/eval_ready_theories \
    --output output_feedback_loop \
    --iterations 2 \
    --model_name gemini-2.5-pro
```

工作流程：
1. 加载原始理论
2. 进行三角色评估
3. 提取改进建议
4. 生成改进版本
5. 重复迭代

### 方法2：使用反馈感知生成器

```python
# 在代码中使用
from theory_generation.methods.feedback_aware_generator import FeedbackAwareGenerator

# 从评估结果创建生成器
generator = FeedbackAwareGenerator.from_evaluation_results(
    evaluation_dir="path/to/evaluation/results",
    theories_dir="data/theories_test",
    output_dir="output_feedback_aware"
)

# 生成基于反馈的新理论
result = await generator._async_generate()
```

### 方法3：集成到演进系统

```bash
# 使用feedback_aware作为生成方法
python run_clean_evolution.py \
    --initial_theories_dir data/theories_test \
    --synthesis_method feedback_aware \
    --use_raw_literature \
    --max_generations 1
```

## 反馈类型

系统能够识别和处理以下类型的反馈：

1. **数学形式化缺陷**
   - 缺乏方程
   - 数学框架不完整
   - 形式化描述不足

2. **实验预测需求**
   - 缺少可验证预测
   - 定量预测不明确
   - 实验设计缺失

3. **概念澄清问题**
   - 定义不清
   - 概念模糊
   - 术语混淆

4. **成功方面（需保持）**
   - 创新概念
   - 优雅解释
   - 哲学深度

## 示例输出

### 原始理论评估
```json
{
  "physicist": {
    "weaknesses": [
      "缺乏C-Field的具体数学描述",
      "Reality Weight的量化方法不明确"
    ],
    "improvement_suggestions": "需要建立C-Field的拉格朗日量..."
  }
}
```

### 改进后的理论
```json
{
  "name": "Coherent Reality Interpretation v2",
  "mathematical_formalism": "C-Field拉格朗日量：L_C = -1/4 F_{μν}F^{μν} + ...",
  "improvements_made": "添加了C-Field的完整数学描述，定义了Reality Weight演化方程"
}
```

## 优势

1. **自动化**：减少人工干预，自动从评估中学习
2. **针对性**：根据具体问题生成改进
3. **迭代优化**：支持多轮改进
4. **保持创新**：在改进的同时保持原理论的核心创新

## 测试

运行测试脚本：
```bash
python test_feedback_loop.py
```

这将演示：
- 简单反馈循环的完整流程
- 基于反馈的理论生成

## 注意事项

1. 反馈循环需要较多的API调用，建议使用较快的模型（如gemini-2.5-flash）进行测试
2. 迭代次数不宜过多（建议2-3次），避免过度优化
3. 每次迭代都会保存中间结果，便于分析优化过程

## 未来改进

1. 更智能的反馈解析（使用NLP技术）
2. 自动评估改进效果
3. 跨理论的改进模式学习
4. 集成到主演进循环中