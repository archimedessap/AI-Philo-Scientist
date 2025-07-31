# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniversalTheoryGen is a quantum mechanics theory generation and evaluation framework that uses high-dimensional concept spaces and LLMs to explore novel interpretations. The system implements AI-Philo1.0, a contradiction-based theory generation method.

## Key Commands

### Theory Generation

```bash
# Basic theory generation with direct synthesis (AI-Philo1.0 core method)
python run_clean_evolution.py \
    --initial_theories_dir data/theories_test \
    --synthesis_method direct_synthesis \
    --max_generations 1 \
    --model_source google \
    --model_name gemini-2.5-flash

# Enhanced unified method using literature concepts
python prepare_enhanced_concepts.py \
    --literature_dir data/raw_literature \
    --output_dir data/enhanced_concepts

python run_clean_evolution.py \
    --synthesis_method unified \
    --use_raw_literature \
    --force_load_literature

# Full pipeline with evaluation feedback
bash run_full_pipeline.sh
```

### Theory Evaluation

```bash
# Experimental evaluation (5 quantum experiments)
python demo/demo_1.py \
    --theory_path output_theories \
    --experiment_dir demo/experiments \
    --output_dir evaluation_results \
    --model_source google \
    --model_name gemini-2.5-flash

# Role-based evaluation for high-scoring theories
python demo/auto_role_evaluation.py \
    evaluation_results/theory_rankings.json \
    --theories_dir data/theories_test \
    --threshold 0.6
```

### Feedback Loop

```bash
# Simple feedback loop (role evaluation only)
python simple_feedback_loop_v2.py \
    --theory theory.json \
    --output improved_theories \
    --iterations 2

# Full evaluation feedback loop
python full_evaluation_feedback_loop.py \
    --theory theory.json \
    --experiment_dir demo/experiments \
    --iterations 1 \
    --skip_experiments  # for faster testing
```

### Testing

```bash
# Check dependencies
python check_dependencies.py

# Run visualization demos
python demo_concept_space_visualization.py --all
```

## Architecture Overview

### Core Theory Generation Flow

```
1. Contradiction Detection (theory_generation/methods/contradiction_detector.py)
   - Identifies philosophical contradictions between theories
   - Extracts conflicting assumptions and principles

2. Theory Synthesis (theory_generation/methods/direct_synthesis_generator.py)
   - Generates new theories by resolving contradictions
   - Multiple methods: direct_synthesis, unified, multi_level, concept_relaxation

3. Concept Space Construction (theory_generation/methods/unified_generator_adapter.py)
   - Loads concepts from prior theories and literature
   - Builds high-dimensional embedding space
   - Identifies conceptual gaps for theory generation

4. Evaluation Pipeline (demo/demo_1.py + theory_validation/)
   - Experimental validation against 5 quantum experiments
   - Role-based evaluation (physicist, philosopher, mathematician)
   - Combined scoring: 60% experimental + 40% role evaluation

5. Feedback Integration (full_evaluation_feedback_loop.py)
   - Extracts improvement suggestions from evaluations
   - Generates enhanced theories based on feedback
   - Supports iterative refinement
```

### Key Data Structures

**Theory Format (Schema 2.1)**:
```json
{
  "metadata": {"schema_version": "2.1"},
  "name": "Theory Name",
  "description": "...",
  "core_assumptions": [...],
  "mathematical_formalism": "...",
  "empirical_predictions": [...],
  "philosophy": {
    "ontology": "...",
    "measurement": "..."
  }
}
```

**Evaluation Results**:
- Experimental: success_rate, average_chi2, per-experiment results
- Role-based: scores, strengths, weaknesses, suggestions per role
- Combined: weighted ranking integrating both evaluations

### Critical Files

- `theory_generation/generation_hub.py`: Central registry for all generation methods
- `theory_generation/methods/unified_space_based_generator.py`: Core high-dimensional space logic
- `demo/demo_1.py`: Main evaluation entry point - handles both experimental and role evaluation
- `theory_validation/agent_validation/theory_evaluator.py`: Role-based evaluation implementation
- `utils/theory_format_converter.py`: Handles theory format conversions between schemas

### Common Issues

1. **Schema Version Mismatch**: Theories must have `metadata.schema_version = "2.1"`
2. **API Method Names**: Use `query_async()` not `generate()` or `call_llm_api()`
3. **Experiment Directory**: Always required even with `--skip_experiments`, use `demo/experiments`
4. **JSON in LLM Responses**: Strip markdown code blocks with `response[7:-3].strip()`
5. **Concurrent API Calls**: Batch multiple tool calls for performance

### Environment Setup

Required API keys in `.env`:
```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
ANTHROPIC_API_KEY=...
```

At least one API key must be configured. Google Gemini models are recommended for cost/performance.

### Logging

Logs are automatically created in `logs/run_YYYYMMDD_HHMMSS/`. Key log files:
- `unified_theory_gen.log`: Main generation logs
- `synthesis.log`: Theory synthesis details
- `evaluation.log`: Evaluation process logs

### Resume Capability

If interrupted, use `--resume run_ID` to continue from checkpoint:
```bash
python run_clean_evolution.py --resume run_20250725_123456 [original parameters]
```

### Performance Tips

- Use `--test_mode` for quick validation with reduced data
- Enable caching with `--use_cache` for unified methods
- Use `gemini-2.5-flash` for development/testing (faster, cheaper)
- Use `gemini-2.5-pro` or `gpt-4` for production quality