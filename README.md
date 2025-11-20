# UniversalTheoryGen - 精简版（论文附件入口）

本分支只保留与“第一个全流程”直接相关的代码与论文引用资产，删去了体量很大的中间产物与旧流水线。开箱即可看到核心目录与最终排名结果。

## 核心结构
- `run_direct_synthesis.py`：直接矛盾分析 → 假说生成主入口（支持 `--model_source google --model_name gemini-2.5-pro`）。
- `pipelines/`：短卡 RAG + 矛盾表构建 + 新理论生成的一键流程（`pipelines/run_full_pipeline.py`）。
- `schemas/`：理论、短卡、矛盾表、新诠释的 JSON Schema v2.1。
- `cards/`：短卡知识库；`data/theories_v2.1/`：原始理论 JSON。
- `theory_generation/`：LLM 调用与直接合成、矛盾分析、假说生成逻辑。
- `theory_evaluation/`、`theory_validation/`：评估与验证组件。
- `results/`：论文引用的精简评估汇总（见下文“全局排名结果”）。

## 快速运行（direct 模式）
最小复现（随机抽 1 对理论、每对 1 个候选）：
```bash
python run_direct_synthesis.py \
  --model_source google \
  --model_name gemini-2.5-pro \
  --generation_method direct \
  --max_pairs 1 \
  --variants_per_contradiction 1 \
  --output_dir runs/direct_gemini
```
生成物会放在 `runs/direct_gemini/synthesis_*/`，其中 `eval_ready_theories/` 即评估输入。

## 快速运行（短卡全链路）
```bash
python pipelines/run_full_pipeline.py \
  --query "Joint analysis of all prior theories" \
  --topk 50 \
  --model_source google \
  --model_name gemini-2.5-pro \
  --skip-evaluation
```
输出：`runs/full_pipeline/<timestamp>/`，含矛盾表、机器摘要、可评估的理论 JSON。

## 全局排名结果（论文引用）
- `results/prior_eval_paper/run_20251007_prior_eval/final_evaluation_summary.json`
- `results/prior_eval_paper/run_20251008_prior_eval_multi/final_evaluation_summary.json`

以上是经过标准化的最终排名摘要，已按理论名排序便于审阅与 diff。更详细的逐条评测输出可在 `results/prior_theories_evaluation/` 查看。

## 依赖
```bash
pip install -r requirements.txt
```
