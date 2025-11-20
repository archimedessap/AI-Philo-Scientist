# AI-Philo-Scientist: Paper Submission Artifacts

> **Branch Overview**: This branch (`AI-Philo-paper`) contains the streamlined code, data, and results for the paper **"AI-Philo-Scientist: Automated Theory Generation & Evolution"**. It has been cleaned to facilitate reproducibility and review.

## 1. Workflow Overview (核心工作流)

The following diagram illustrates the automated theory generation and evolution process implemented in this project:

![AI-Philo-Scientist Workflow](project_workflow_detailed.svg)

## 2. Repository Contents (包含内容)

This branch retains only the essential components required to run the full workflow and verify the results:

- **`run_full_cycle.py`**: The main entry point for the complete generation-evaluation-evolution loop.
- **`run_evolution_cycle.py`**: The orchestrator for the multi-generational evolutionary process.
- **`theory_generation/`**: Core logic for theory synthesis (Direct Synthesis, Short Card RAG).
- **`demo/`**: Evaluation modules, including experimental validation and role-based peer review.
- **`data/theories_v2.1/`**: Initial seed theories used in the experiments.
- **`paper/tables/`**: Final results and rankings (CSV/TeX).

## 3. Reproduction (复现指南)

To reproduce the full evolutionary cycle:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full cycle (Generation -> Evaluation -> Evolution)
python run_full_cycle.py --max_generations 3
```

## 4. Results (结果)

The final global ranking of generated theories can be found in:
- [CSV Table](paper/tables/global_ranking_with_models.csv)
- [LaTeX Table](paper/tables/global_ranking_with_models.tex)
