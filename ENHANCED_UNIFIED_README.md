# 增强的Unified理论生成方法使用指南

## 概述

增强的Unified方法通过整合原始文献中的概念和公式，构建更丰富的概念空间，从而生成更高质量的理论。

## 主要改进

1. **文献概念提取**：从预处理的科学文献中自动提取概念和关系
2. **公式提取和分类**：识别并分类数学公式，建立公式之间的关系
3. **知识图谱构建**：构建包含概念、公式和理论的多层知识网络
4. **物理领域嵌入**：使用领域特定的嵌入方法，更好地捕捉物理概念的语义

## 使用方法

### 方法1：完整流程（推荐）

运行完整的自动化流程：

```bash
# 基本用法
./run_full_pipeline.sh

# 测试模式（快速验证）
./run_full_pipeline.sh --test-mode

# 跳过概念提取（使用已有数据）
./run_full_pipeline.sh --skip-extraction

# 自定义参数
./run_full_pipeline.sh \
    --max-generations 5 \
    --model-source google \
    --model-name gemini-2.0-flash-exp
```

### 方法2：分步执行

#### 步骤1：准备概念空间

```bash
# 从文献提取概念和公式
python prepare_enhanced_concepts.py \
    --preprocessed_dir data/preprocessed \
    --model_source google \
    --model_name gemini-2.0-flash-exp

# 测试模式（只处理2个文档）
python prepare_enhanced_concepts.py --test_mode
```

#### 步骤2：运行理论生成

```bash
# 使用增强的unified方法
python run_clean_evolution.py \
    --initial_theories_dir data/theories_v2.1 \
    --output_root output_enhanced \
    --synthesis_method unified \
    --use_raw_literature \
    --max_generations 3 \
    --synthesis_model_source google \
    --synthesis_model_name gemini-2.0-flash-exp
```

### 方法3：直接测试unified方法

```bash
# 快速测试
python run_theory_generation.py \
    --method unified \
    --theories_dir data/theories_v2.1 \
    --output_dir output_unified_test \
    --num_theories 3 \
    --model_source google \
    --model_name gemini-2.0-flash-exp
```

## 关键参数说明

- `--synthesis_method unified`: 使用增强的unified方法
- `--use_raw_literature`: 启用文献概念增强
- `--test_mode`: 测试模式，处理较少文档
- `--skip-extraction`: 跳过概念提取，使用现有数据
- `--max_generations`: 演进的最大代数

## 输出文件

- `data/enhanced_concepts/`: 提取的概念和关系
- `data/extracted_formulas/`: 提取的公式
- `data/knowledge_graph/`: 知识图谱文件
- `output_*/run_*/generation_*/synthesis/eval_ready_theories/`: 生成的理论
- `output_*/run_*/benchmark_comparison/`: 与先验理论的对比报告

## 监控进度

```bash
# 查看日志
tail -f output_*/run_*/generation_*/synthesis/unified_generator.log

# 查看manifest状态
watch -n 10 'cat output_*/run_*/run_manifest.json | jq ".statistics"'
```

## 常见问题

1. **内存不足**：减少`--max_documents`参数或使用`--test_mode`
2. **API限制**：调整批处理大小`--batch_size`
3. **找不到概念文件**：先运行`prepare_enhanced_concepts.py`

## 性能优化建议

1. 首次运行建议使用`--test_mode`验证流程
2. 概念提取可以单独运行并保存，后续使用`--skip-extraction`
3. 对于大规模运行，建议分批处理文档