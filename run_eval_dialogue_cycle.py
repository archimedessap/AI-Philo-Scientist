#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_dialogue_cycle.py (Manifest-driven)
============================================
Orchestrates a multi-generational evaluation and refinement cycle for
scientific theories using a central 'manifest' to track all state,
eliminating reliance on fragile file system paths and conventions.
"""

from __future__ import annotations
import subprocess
import sys
import argparse
from pathlib import Path
import glob
import json
import shutil
import time
from typing import List
import os
from datetime import datetime

# NEW: Import our manifest tools
import manifest_tools


# -----------------------------------------------------------------------------
# 工具函数 (简化)
# -----------------------------------------------------------------------------

def run_cmd(cmd: list[str], **kwargs):
    """Wraps subprocess.run for convenience."""
    print("\n$ " + " ".join(cmd))
    # Using check=True to automatically raise an exception on non-zero exit codes.
    subprocess.run(cmd, check=True, **kwargs)

def find_unique(pattern: str) -> Path:
    """Finds a unique file or directory, prioritizing the most recently modified."""
    matches = sorted(list(Path().glob(pattern)), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No matches found for pattern: {pattern}")
    if len(matches) > 1:
        print(f"[WARN] Found multiple matches for {pattern}, selecting most recent: {matches[0]}")
    return matches[0]

# OBSOLETE FUNCTIONS `collect_best_versions` etc. are now removed.

# -----------------------------------------------------------------------------
# 主逻辑 (最终重构版)
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Manifest-based Evaluation Cycle")
    parser.add_argument('--generations', type=int, default=3, help='Number of generations to run.')
    parser.add_argument('--initial_theories_dir', type=str, required=True, help='Directory with initial theories.')
    parser.add_argument("--output_root", default="data/dialog_cycle_runs_manifest", help="Output root")
    parser.add_argument("--top_n", type=int, default=1, help="Top N theories to refine")
    parser.add_argument("--max_iters", type=int, default=2, help="Max refinement iterations")
    parser.add_argument("--min_improve", type=float, default=0.03)
    parser.add_argument("--synthesis_model_source", default="google")
    parser.add_argument("--synthesis_model_name", default="gemini-1.5-pro-latest")
    parser.add_argument("--evaluation_model_source", default="google")
    parser.add_argument("--evaluation_model_name", default="gemini-2.5-pro")
    parser.add_argument("--dialog_model_source", default="google")
    parser.add_argument("--dialog_model_name", default="gemini-2.5-pro")
    parser.add_argument("--role_model_source", default="openai")
    parser.add_argument("--role_model_name", default="gpt-4o-mini")
    parser.add_argument("--role_eval_threshold", type=float, default=0.6, help="Minimum score for a role to be considered valid.")
    parser.add_argument("--promotion_min_score", type=float, default=0.5, help="Minimum score for a theory to be promoted to the next generation.")
    parser.add_argument("--use_instrument_correction", action="store_true", help="Use instrument correction for evaluation.")
    
    # New arguments for controlling theory generation
    parser.add_argument("--max_pairs_to_analyze", type=int, default=10, help="Max pairs of theories to analyze for contradictions during initial generation.")
    parser.add_argument("--variants_per_contradiction", type=int, default=1, help="Number of new theory variants to generate per contradiction during initial generation.")

    args = parser.parse_args()

    # --- 1. Initialization ---
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.output_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    
    manifest = manifest_tools.initialize_manifest(run_id, args)
    manifest_path = run_root / "run_manifest.json"

    # --- 2. Generation 0 ---
    print("\n" + "="*80 + "\nGeneration 0: Creation and Initial Evaluation\n" + "="*80)
    gen0_dir = run_root / "generation_0"
    
    # Phase A: Create and evaluate initial theories
    full_cycle_dir = gen0_dir / "full_cycle"
    cmd_full = [
        "python", "run_full_cycle.py",
        "--existing_theories_dir", args.initial_theories_dir,
        "--base_output_dir", str(full_cycle_dir),
        "--synthesis_model_source", args.synthesis_model_source,
        "--synthesis_model_name", args.synthesis_model_name,
        "--evaluation_model_source", args.evaluation_model_source,
        "--evaluation_model_name", args.evaluation_model_name,
        "--role_eval_threshold", str(args.role_eval_threshold),
        "--max_pairs_to_analyze", str(args.max_pairs_to_analyze),
        "--variants_per_contradiction", str(args.variants_per_contradiction),
    ]
    if args.use_instrument_correction:
        cmd_full.append("--use_instrument_correction")
    
    print(f"$ {' '.join(cmd_full)}")
    run_cmd(cmd_full)
    
    # Phase B: Ingest the results into the manifest
    try:
        # 1. 智能地寻找新理论所在的目录 (这部分之前是正确的)
        theories_root_path = find_unique(f"{full_cycle_dir}/run_*/1_synthesis_output/*/eval_ready_theories")
        
        # 2. 将新理论注册到清单中
        for theory_file in theories_root_path.glob("*.json"):
            with open(theory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manifest_tools.register_theory_in_manifest(manifest, data, theory_file, generation=0)

        # 3. [核心修复] 使用通配符智能地寻找包含最终综合评分的总结文件
        #    这个路径现在可以正确处理由子脚本创建的、任意名称的带时间戳的子目录
        summary_path = find_unique(f"{full_cycle_dir}/run_*/2_evaluation_output/run_*/role_evaluations/combined_rankings.json")
        
        # 4. 用找到的评估分数更新清单
        manifest_tools.update_manifest_with_evaluation(manifest, str(summary_path))
        manifest_tools.save_manifest(manifest, manifest_path)
        
    except FileNotFoundError as e:
        print(f"[FATAL] 无法找到第0代周期的关键输出文件。这很可能是脚本间的路径不匹配导致。错误: {e}")
        sys.exit(1)

    print("\n[MANIFEST] Selecting best theories from Generation 0...")
    promoted_ids = manifest_tools.select_best_theories_for_next_gen(
        manifest, 
        current_gen=0,
        top_n=args.top_n,
        min_score=args.promotion_min_score
    )

    if not promoted_ids:
        print(f"[FATAL] Gen 0 produced no theories that passed the promotion threshold of {args.promotion_min_score:.2f}. Exiting.")
        manifest_tools.save_manifest(manifest, manifest_path)
        return

    # --- 3. Subsequent Generations Loop ---
    for gen in range(1, args.generations):
        print("\n" + "="*80 + f"\nGeneration {gen}: Refinement and Re-evaluation\n" + "="*80)
        gen_dir = run_root / f"generation_{gen}"
        gen_dir.mkdir(exist_ok=True)
        
        # Phase A: Refine theories from the previous generation
        if not promoted_ids:
            print(f"[INFO] No theories to refine for Gen {gen}. Ending run."); break
            
        print(f"\n--- Refining {len(promoted_ids)} theories ---")
        refinement_dir = gen_dir / "refinement"
        
        # NOTE: Using the arguments as defined in the user's original script version
        cmd_refine = [
            "python", "run_m3_auto_refinement.py", 
            "--manifest-path", str(manifest_path.resolve()),
            "--theory-ids", ",".join(promoted_ids), 
            "--output-dir", str(refinement_dir.resolve()),
            "--top-n", str(args.top_n), 
            "--max-iters", str(args.max_iters), 
            "--min-improve", str(args.min_improve),
            "--judge_model_source", args.evaluation_model_source, 
            "--judge_model_name", args.evaluation_model_name,
            "--dialog_model_source", args.dialog_model_source, 
            "--dialog_model_name", args.dialog_model_name,
        ]
        run_cmd(cmd_refine)
        manifest = manifest_tools.load_manifest(manifest_path) # Reload manifest to see new variants

        # Phase B: Evaluate all candidates for this generation.
        # This now uses a standard evaluation script. Let's assume demo_1.py is the one.
        # We gather ALL theories currently marked 'untested' in the manifest.
        ids_to_evaluate = [tid for tid, data in manifest['theories'].items() if data.get('status') == 'untested']
        
        if not ids_to_evaluate:
            print(f"\n[INFO] No new theory variants were created in Gen {gen}. Nothing to evaluate. Ending run.")
            break
            
        print(f"\n--- Evaluating a pool of {len(ids_to_evaluate)} new theories ---")

        # Instead of creating a temp dir, we make the eval script manifest-aware (hypothetically)
        # For now, let's stick to the existing pattern of demo_1.py if it can take a list of files
        # A robust way is to create a temporary directory for this generation's evaluation.
        eval_input_dir = gen_dir / "evaluation_input"
        eval_input_dir.mkdir(exist_ok=True)
        for theory_id in ids_to_evaluate:
            source_path = Path(manifest["theories"][theory_id]["path"])
            shutil.copy(source_path, eval_input_dir / source_path.name)
        
        eval_output_dir = gen_dir / "evaluation_output"
        cmd_eval = [
            "python", "demo/demo_1.py", 
            "--theory_path", str(eval_input_dir), 
            "--experiment_dir", "demo/experiments/",
            "--output_dir", str(eval_output_dir), 
            "--model_source", args.evaluation_model_source,
            "--model_name", args.evaluation_model_name, 
            "--run_role_evaluation", 
            "--role_model_source", args.role_model_source,
            "--role_model_name", args.role_model_name,
        ]
        if args.use_instrument_correction: cmd_eval.append("--use_instrument_correction")
        run_cmd(cmd_eval)
        
        # Phase C: Update manifest with new evaluation scores
        # THIS IS THE CORE FIX: RE-USE THE ROBUST LOGIC FROM GEN 0
        try:
            latest_run_dir = max(
                (eval_output_dir / d for d in os.listdir(eval_output_dir) if d.startswith("run_")),
                key=os.path.getmtime
            )
            summary_path = latest_run_dir / "role_evaluations" / "combined_rankings.json"
            manifest_tools.update_manifest_with_evaluation(manifest, str(summary_path))
        except (FileNotFoundError, ValueError) as e:
            print(f"[FATAL] Could not find or parse output from Gen {gen} evaluation. Halting: {e}"); break
            
        # Phase D: Select best for the *next* generation
        print(f"\n[MANIFEST] Selecting best theories from Generation {gen}...")
        promoted_ids = manifest_tools.select_best_theories_for_next_gen(
            manifest, 
            current_gen=gen,
            top_n=args.top_n,
            min_score=args.promotion_min_score
        )

        if not promoted_ids:
            print(f"[STOP] This generation produced no survivors that passed the threshold of {args.promotion_min_score:.2f}. Halting."); break
        
        manifest_tools.save_manifest(manifest, manifest_path); time.sleep(1)

    # --- 4. Finalization ---
    manifest_tools.save_manifest(manifest, manifest_path)
    print(f"\n🎉 Run complete. Final manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main() 