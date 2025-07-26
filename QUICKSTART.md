# 🚀 UniversalTheoryGen 快速开始指南

欢迎使用 UniversalTheoryGen - 基于高维概念空间的理论生成系统！

## 📋 目录

1. [系统要求](#系统要求)
2. [快速安装](#快速安装)
3. [环境配置](#环境配置)
4. [第一次运行](#第一次运行)
5. [常用命令](#常用命令)
6. [故障排除](#故障排除)

## 🔧 系统要求

- Python 3.8+
- 至少 8GB RAM
- 至少一个 LLM API 密钥（OpenAI、Anthropic、Google、DeepSeek 等）

## ⚡ 快速安装

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/UniversalTheoryGen_0.1.git
cd UniversalTheoryGen_0.1
```

### 2. 检查依赖

```bash
python check_dependencies.py
```

这将自动检查所有必需的依赖，并生成 `requirements.txt` 文件（如果不存在）。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 🔑 环境配置

### 1. 创建 `.env` 文件

在项目根目录创建 `.env` 文件：

```bash
# 选择一个或多个 API 密钥配置
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 2. 验证配置

```bash
python check_dependencies.py
```

确保至少有一个 API 密钥配置正确。

## 🎯 第一次运行

### 示例 1: 基于矛盾分析生成新理论

```bash
python run_theory_generator.py \
    --theories_dir data/theories_v2.1 \
    --output_dir output/my_first_run \
    --model_source google \
    --model_name gemini-2.5-pro \
    --max_theories_to_analyze 3 \
    --num_output_theories 2
```

### 示例 2: 运行完整的理论演进

```bash
python run_clean_evolution.py \
    --initial_theories_dir data/theories_v2.1 \
    --max_generations 2 \
    --synthesis_model_source google \
    --synthesis_model_name gemini-2.5-pro
```

### 示例 3: 使用统一理论生成器

```bash
python run_unified_theory_generation.py \
    --literature_dirs data/literature \
    --prior_theories_dir data/theories_v2.1 \
    --output_dir unified_output \
    --demo_mode
```

## 🛠️ 常用命令

### 生成理论

```bash
# 使用不同的生成方法
--synthesis_method direct_synthesis  # 基于矛盾分析（默认）
--synthesis_method multi_level      # 多级创新
--synthesis_method unified_generator # 高维概念空间
--synthesis_method concept_relaxation # 概念松弛
```

### 评估理论

```bash
# 评估生成的理论
python run_theory_evaluation.py \
    --theories_file output/theories.json \
    --output_dir evaluation_results \
    --manifest_path output/run_manifest.json  # 自动更新评分
```

### 断点续跑

```bash
# 如果运行中断，使用续跑功能
python run_clean_evolution.py \
    --initial_theories_dir data/theories_v2.1 \
    --resume run_20250724_123456  # 使用之前的运行ID
```

### 查看日志

日志文件保存在 `logs/` 目录下：

```bash
# 查看最新的日志
ls -la logs/run_*/
tail -f logs/run_*/unified_theory_gen.log
```

## 🔍 故障排除

### 1. API 调用失败

**问题**: `ConnectionError` 或 `API rate limit exceeded`

**解决方案**:
- 检查 API 密钥是否正确
- 检查网络连接
- 降低并发请求数
- 使用不同的模型源

### 2. 内存不足

**问题**: `MemoryError` 或系统变慢

**解决方案**:
- 减少 `--max_theories_to_analyze` 参数
- 使用 `--test_mode` 进行小规模测试
- 清理缓存: `rm -rf cache/`

### 3. 缓存版本冲突

**问题**: 缓存版本不匹配警告

**解决方案**:
```bash
# 清理旧版本缓存
rm -rf cache/unified_theory/v1.0.0/
# 或使用新版本
```

### 4. 运行中断恢复

**问题**: 运行过程中意外中断

**解决方案**:
```bash
# 查看可恢复的运行
ls output_clean_evolution/

# 使用断点续跑
python run_clean_evolution.py --resume run_20250724_xxxxx [其他原始参数]
```

## 📚 进阶使用

### 1. 自定义理论生成配置

创建 `custom_config.json`:

```json
{
    "target_innovation_levels": ["PARAMETER_EXTENSION", "INTERPRETATION"],
    "synthesis_mode": "fusion",
    "innovation_intensity": 0.8
}
```

### 2. 批量评估

```bash
# 评估整个目录的理论
python run_theory_evaluation.py \
    --theories_file "output/*/theories/*.json" \
    --output_dir batch_evaluation \
    --top_n 10  # 只评估前10个
```

### 3. 概念空间可视化

```bash
# 运行可视化演示
python demo_concept_space_visualization.py --all

# 查看特定演示
python demo_concept_space_visualization.py --demo 1  # 统一理论空间
python demo_concept_space_visualization.py --demo 2  # 自定义概念空间
python demo_concept_space_visualization.py --demo 3  # 概念演化

# 在理论生成时自动生成可视化
python run_unified_theory_generation.py \
    --enable_visualization \
    --output_dir output_with_viz
```

可视化结果包括：
- PCA/t-SNE 二维投影
- 概念聚类分析
- 概念相似度网络
- 概念空间密度热力图
- 概念空白区域标记

### 4. 性能优化

```bash
# 使用测试模式快速验证流程
python run_unified_theory_generation.py --demo_mode

# 启用缓存加速
export CACHE_TTL_HOURS=168  # 缓存7天
```

## 🤝 获取帮助

- 查看详细文档: `docs/`
- 提交问题: [GitHub Issues](https://github.com/yourusername/UniversalTheoryGen/issues)
- 查看示例: `demo/`
- 运行测试: `pytest tests/`

## 🎉 下一步

恭喜！你已经成功运行了 UniversalTheoryGen。接下来可以：

1. 探索不同的理论生成方法
2. 调整参数以获得更好的结果
3. 添加自己的文献数据
4. 扩展评估指标
5. 贡献代码改进

祝你在理论探索之旅中有所收获！🚀