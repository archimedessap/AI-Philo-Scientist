# 评估错误修复说明

## 问题描述

用户在运行增强的unified理论生成方法时遇到了评估失败的问题：

1. **可视化警告**：`setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions`
2. **评估失败**：`Evaluation completed but no results found in output_enhanced_evolution/run_20250724_221824/generation_0/evaluation/results`
3. **理论生成成功但评估失败**：生成了2个理论但评估阶段无法找到结果文件

## 根本原因

经过分析，发现了以下几个问题：

### 1. 理论文件格式不匹配
- unified生成器生成的理论使用了不同的字段名：
  - `core_assumptions` 而非 `core_principles`
  - `mathematical_formalism` 而非 `formalism`
  - `empirical_predictions` 而非 `predictions_and_verifiability`
- demo_1.py期望的字段不存在，导致LLM看到的是"No principles provided"等占位文本

### 2. LLM响应解析问题
- Gemini模型返回的JSON被包裹在markdown代码块中（```json ... ```）
- demo_1.py的解析逻辑无法处理这种格式，导致无法提取预测值
- 当无法提取预测值时，评估函数返回None，不生成评估文件

### 3. 评估流程中断
- 当所有理论评估都失败时，demo_1.py不会生成`final_evaluation_summary.json`
- run_clean_evolution.py找不到评估结果文件，导致整个流程失败

## 解决方案

### 1. 修复demo_1.py的JSON解析
```python
# 添加了对markdown代码块的处理
cleaned_response = response.strip()
if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
    cleaned_response = cleaned_response[7:-3].strip()
```

### 2. 修复理论字段映射
```python
# 添加了字段回退逻辑
{json.dumps(theory.get("core_principles", theory.get("core_assumptions", "No principles provided.")), indent=2)}
{json.dumps(theory.get("formalism", theory.get("mathematical_formalism", "No formalism provided.")), indent=2)}
{json.dumps(theory.get("predictions_and_verifiability", theory.get("empirical_predictions", "No predictions provided.")), indent=2)}
```

### 3. 确保生成评估文件
- 即使LLM无法提供预测值，也生成一个标记为失败的评估文件
- 即使所有评估失败，也生成`final_evaluation_summary.json`

### 4. 创建理论格式转换器
创建了`utils/theory_format_converter.py`，可以自动转换理论格式：
```bash
# 转换单个文件
python utils/theory_format_converter.py input.json -o output.json

# 批量转换目录
python utils/theory_format_converter.py theories_dir/ -o converted_dir/
```

### 5. 集成到评估流程
修改了run_clean_evolution.py，在评估前自动转换理论格式。

## 使用建议

1. **重新运行评估**：修复后的代码应该能够正确处理unified生成的理论
2. **检查生成的理论**：确保理论内容不为空，包含实际的原理和公式
3. **监控LLM响应**：查看`*_response_raw.txt`文件，了解LLM的实际输出

## 测试命令

```bash
# 测试评估修复
python demo/demo_1.py \
    --theory_path output_enhanced_evolution/run_*/generation_0/synthesis/eval_ready_theories \
    --experiment_dir data/experiments \
    --output_dir test_eval_output \
    --model_source google \
    --model_name gemini-2.0-flash-exp

# 重新运行完整流程
./run_full_pipeline.sh --skip-extraction
```

## 注意事项

1. 可视化警告可能来自于numpy数组形状不一致，但不影响评估结果的生成
2. 如果理论内容确实为空，需要检查unified生成器的输出
3. 建议在生成理论时就使用正确的字段名，避免后续转换