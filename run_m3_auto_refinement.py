#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_m3_auto_refinement.py (Manifest-Aware Adapter)
===================================================
This script acts as a manifest-aware adapter for the legacy
`run_refinement_loop.py` script.

It does the following:
1. Reads the central manifest and a list of theory IDs.
2. Creates temporary input files (`summary.json`, theories dir) in the
   format expected by the legacy script.
3. Calls `run_refinement_loop.py`.
4. After completion, parses the output of the legacy script.
5. Registers any new, improved theory variants back into the central manifest.
"""

from __future__ import annotations
import sys
import argparse
import json
import shutil
from pathlib import Path
import subprocess

# NEW: Import manifest tools
import manifest_tools

def run_cmd(cmd: list[str]):
    """Wraps subprocess.call and prints the command."""
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(description="M3 Manifest-Aware Refinement Adapter")
    # NEW: Manifest-aware arguments
    parser.add_argument("--manifest-path", required=True, type=Path, help="Path to the master run_manifest.json")
    parser.add_argument("--theory-ids", required=True, help="Comma-separated list of theory IDs to refine")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to store all refinement outputs")
    
    # Arguments to be passed through to the legacy script
    parser.add_argument("--top-n", type=int, default=5, help="筛选前 N")
    parser.add_argument("--eval_mode", choices=["quick", "real"], default="real", help="评估模式")
    parser.add_argument("--max-iters", type=int, default=3, help="深度优化最大迭代次数")
    parser.add_argument("--min-improve", type=float, default=0.05, help="视为有效提升的最小 Δ 分")
    parser.add_argument("--judge_model_source", default="deepseek")
    parser.add_argument("--judge_model_name", default="deepseek-reasoner")
    parser.add_argument("--dialog_model_source", default="deepseek")
    parser.add_argument("--dialog_model_name", default="deepseek-reasoner")
    args = parser.parse_args()

    # --- 1. Load Manifest and Prepare Inputs for Legacy Script ---
    print("[INFO] Loading manifest and preparing inputs for legacy refinement loop...")
    manifest = manifest_tools.load_manifest(args.manifest_path)
    theory_ids = args.theory_ids.split(',')
    
    run_root = args.output_dir
    run_root.mkdir(parents=True, exist_ok=True)
    
    temp_theories_root = run_root / "refinement_input_theories"
    temp_theories_root.mkdir()
    
    temp_summary_data = []
    name_to_id_map = {}
    for tid in theory_ids:
        details = manifest["theories"][tid]
        src_path = Path(details["file_path"])
        dest_path = temp_theories_root / f"{tid}_{src_path.name}"
        shutil.copy(src_path, dest_path)
        
        entry = {
            "theory_name": details["theory_name"],
            "file_path": str(dest_path.resolve()),
            **details.get("scores", {}).get("experimental", {})
        }
        temp_summary_data.append(entry)
        name_to_id_map[details["theory_name"]] = tid

    temp_summary_path = run_root / "temp_summary_for_loop.json"
    with open(temp_summary_path, 'w', encoding='utf-8') as f:
        json.dump(temp_summary_data, f, indent=2)

    # --- 2. Call the Legacy Refinement Loop ---
    depth_output_dir = run_root / "depth_output"
    cmd_refine_loop = [
        "python", "run_refinement_loop.py",
        "--summary_file", str(temp_summary_path),
        "--theories_root", str(temp_theories_root),
        "--output_root", str(depth_output_dir),
        "--top_n", str(args.top_n),
        "--min_improve", str(args.min_improve),
        "--eval_mode", args.eval_mode,
        "--max_iters", str(args.max_iters),
        "--judge_model_source", args.judge_model_source,
        "--judge_model_name", args.judge_model_name,
        "--dialog_model_source", args.dialog_model_source,
        "--dialog_model_name", args.dialog_model_name,
    ]
    if run_cmd(cmd_refine_loop) != 0:
        print("[FATAL] Legacy refinement loop script failed.")
        sys.exit(1)
        
    # --- 3. Absorb Results Back into Manifest ---
    print("\n[INFO] Refinement loop finished. Absorbing results into manifest...")
    summaries_dir = depth_output_dir / "depth_runs"
    if not summaries_dir.exists():
        print("[WARN] No 'depth_runs' output directory found. Skipping absorption.")
        return

    for summary_file in summaries_dir.rglob("summary.json"):
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        parent_theory_name = data.get("theory_name")
        parent_id = name_to_id_map.get(parent_theory_name)
        if not parent_id:
            print(f"[WARN] Could not find parent ID for theory '{parent_theory_name}' in manifest.")
            continue
            
        # If the refinement was successful, find the new candidate file
        # Heuristic: final score > baseline score
        if data.get("final_score", 0) > (data.get("scores", [0])[0]):
            theory_output_dir = summary_file.parent
            try:
                candidate_files = sorted(list(theory_output_dir.glob("candidate_*.json")))
                if not candidate_files:
                    continue
                best_candidate_path = candidate_files[-1]

                with open(best_candidate_path, 'r', encoding='utf-8') as f:
                    refined_data = json.load(f)

                # Register the new variant in the manifest
                variant_id = manifest_tools.register_refined_variant(
                    manifest=manifest,
                    parent_theory_id=parent_id,
                    refined_theory_data=refined_data,
                    refined_file_path=best_candidate_path,
                    refinement_run_dir=theory_output_dir
                )
                # Optionally, add the refinement score to the new variant
                manifest["theories"][variant_id]["scores"]["refinement_role_score"] = data.get("final_score")

            except (FileNotFoundError, IndexError) as e:
                print(f"[WARN] Could not find a candidate file for '{parent_theory_name}' despite score improvement. Error: {e}")

    # --- 4. Save the Updated Manifest ---
    manifest_tools.save_manifest(manifest, args.manifest_path)
    print("🏆 Refinement complete. Manifest has been updated.")

if __name__ == "__main__":
    main() 