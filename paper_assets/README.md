# paper_assets 使用说明

本目录包含论文数据资产（CSV）与可视化图（PNG），以及案例卡片（Markdown）。

生成命令：

```bash
bash scripts/paper/build_paper_assets.sh
```

默认输出：

- `leaderboard_topN.csv`：全局理论空间 Top-N 榜单（按综合分排序）
- `generation_curves_latest.csv`：最近一次运行的代际曲线数据（每代总数、晋级数、最佳/均值分）
- `fig_leaderboard_topN.png`：榜单条形图
- `fig_generation_curves_latest.png`：代际曲线图
- `case_cards/`：Top-N 案例卡片（可直接用于论文案例章节）

可调环境变量：

- `TOP_N`：榜单 Top-N（默认 20）
- `CASE_N`：案例卡片数量（默认 3）

数据来源：

- 注册库索引：`global_theory_registry/theory_index.json`
- 运行清单：`output_*/*/run_manifest.json`

注意：

- 若 `.env` 中有真实密钥，请勿提交/公开；论文产物与复现实验不依赖密钥内容。
- 生成图表依赖 `matplotlib`，已在 `requirements.txt` 中列出。

