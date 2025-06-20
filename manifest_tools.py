#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manifest_tools.py
=================
This module provides a set of functions to manage the 'run manifest',
a central dictionary that tracks all data and metadata for a single
run of the theory generation and evaluation cycle.
"""
from __future__ import annotations
import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List
import glob
import os


def _normalize_name_for_matching(name: str) -> str:
    """Creates a canonical, simplified version of a theory name for robust matching."""
    if not isinstance(name, str):
        return ""
    
    s = name.lower()
    # Remove common variant indicators and acronyms in parentheses
    s = s.replace('(variant 1)', '').replace('(rpt)', '').replace('(spd)', '').replace('(ic)', '')
    # Remove all non-alphanumeric characters
    s = ''.join(c for c in s if c.isalnum())
    return s


def initialize_manifest(run_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Creates the initial manifest for the run."""
    print("[MANIFEST] Initializing new manifest for run_id:", run_id)
    return {
        "run_id": run_id,
        "config": vars(args),
        "theories": {},
        "lineage": {}  # Maps child_id -> parent_id
    }


def save_manifest(manifest: Dict[str, Any], manifest_path: Path):
    """Saves the manifest dictionary to a JSON file."""
    print(f"[MANIFEST] Saving manifest to: {manifest_path}")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Loads a manifest from a JSON file."""
    print(f"[MANIFEST] Loading manifest from: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def register_theory_in_manifest(manifest: Dict[str, Any], theory_data: Dict[str, Any], file_path: Path, generation: int) -> str:
    """Registers a new theory in the manifest, returning its unique ID."""
    theory_id = f"theory_{uuid.uuid4().hex[:8]}"
    theory_name = theory_data.get("name", "Unnamed Theory")
    print(f"[MANIFEST] Registering new theory: '{theory_name}' as ID: {theory_id}")
    
    manifest["theories"][theory_id] = {
        "theory_name": theory_name,
        "file_path": str(file_path.resolve()),
        "generation": generation,
        "status": "created",
        "scores": {}
    }
    return theory_id

def update_manifest_with_evaluation(manifest: dict, summary_path: str | Path):
    """
    Updates the manifest with scores from an evaluation summary file.
    This can handle both a direct list of theories or a dictionary
    containing a 'rankings' key.
    """
    print(f"[MANIFEST] Updating manifest with evaluation results from: {summary_path}")
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[WARN] Could not parse or find evaluation summary file {summary_path}: {e}")
        return

    # [核心修复] 检查加载的数据是字典还是列表
    theories_list = []
    if isinstance(eval_data, dict) and 'rankings' in eval_data:
        # 如果是字典，并且包含 'rankings'键，则提取该列表
        theories_list = eval_data['rankings']
        print(f"[INFO] Successfully extracted {len(theories_list)} theories from 'rankings' key.")
    elif isinstance(eval_data, list):
        # 如果本身就是列表，直接使用
        theories_list = eval_data
    else:
        print(f"[WARN] Evaluation summary file {summary_path} has an unexpected format. Skipping.")
        return

    # 后续逻辑现在处理的是一个保证存在的列表 `theories_list`
    name_to_score_map = {
        item['theory_name']: item.get('combined_score', item.get('composite_score', item.get('success_rate')))
        for item in theories_list
        if 'theory_name' in item
    }

    if not name_to_score_map:
        print(f"[WARN] No theory scores found in {summary_path}.")
        return

    # Create a lookup map of normalized names to theory IDs for all unevaluated theories
    name_to_id_map = {
        _normalize_name_for_matching(t_data.get('theory_name')): t_id
        for t_id, t_data in manifest['theories'].items()
        if t_data.get('status') in ['created', 'refined']  # 只匹配未评估的理论
    }

    updated_count = 0
    for summary_name, score in name_to_score_map.items():
        normalized_summary_name = _normalize_name_for_matching(summary_name)
        
        if normalized_summary_name in name_to_id_map and score is not None:
            theory_id = name_to_id_map[normalized_summary_name]
            manifest['theories'][theory_id]['score'] = float(score)
            manifest['theories'][theory_id]['status'] = 'evaluated'
            manifest['theories'][theory_id]['eval_summary_path'] = str(summary_path)
            updated_count += 1
            # Remove from map to prevent matching the same theory twice
            del name_to_id_map[normalized_summary_name]
        elif score is None:
            print(f"[WARN] Theory '{summary_name}' has no valid score (None), skipping.")

    print(f"[MANIFEST] Updated {updated_count} theories with evaluation scores.")


def register_refined_variant(manifest: Dict[str, Any], parent_theory_id: str, refined_theory_data: Dict[str, Any], refined_file_path: Path, refinement_run_dir: Path) -> str:
    """Registers a refined theory variant in the manifest."""
    parent_details = manifest["theories"][parent_theory_id]
    parent_gen = parent_details["generation"]
    
    refined_id = f"theory_{uuid.uuid4().hex[:8]}"
    refined_name = refined_theory_data.get("name", "Unnamed Refined Theory")
    print(f"[MANIFEST] Registering refined variant for '{parent_details['theory_name']}' as ID: {refined_id}")

    # 确定精炼变体应属于的代际
    # 如果父理论已经是晋级状态，则精炼变体应属于下一代
    if parent_details.get('status') == 'promoted' or 'promoted_to_gen_' in parent_details.get('status', ''):
        target_generation = parent_gen + 1
    else:
        target_generation = parent_gen

    manifest["theories"][refined_id] = {
        "theory_name": refined_name,
        "file_path": str(refined_file_path.resolve()),
        "generation": target_generation,
        "status": "created",  # New variants need to be evaluated
        "scores": {},
        "refinement_run_dir": str(refinement_run_dir),
        "refinement_parent": parent_theory_id  # Track which theory this is refined from
    }
    manifest["lineage"][refined_id] = parent_theory_id
    return refined_id

def select_best_theories_for_next_gen(manifest: Dict[str, Any], current_gen: int, top_n: int, min_score: float) -> List[str]:
    """
    Analyzes the manifest to select the best version of each theory line for the next generation.
    Returns a list of theory_ids that should be promoted.
    """
    print(f"\n[MANIFEST] Selecting best theories from Generation {current_gen}...")
    
    # Group all theories by their original ancestor to form family trees
    family_trees: Dict[str, List[str]] = {}
    for theory_id in manifest["theories"]:
        ancestor_id = theory_id
        while ancestor_id in manifest["lineage"]:
            ancestor_id = manifest["lineage"][ancestor_id]
        
        if ancestor_id not in family_trees:
            family_trees[ancestor_id] = []
        family_trees[ancestor_id].append(theory_id)

    promoted_ids = []
    
    for family_name, members in family_trees.items():
        print(f"  - Analyzing family of '{family_name}':")
        best_member_id = None
        best_score = -1.0

        for member_id in members:
            details = manifest["theories"][member_id]
            current_score = details.get('score')
            score_str = f"{current_score:.3f}" if current_score is not None else "N/A"
            print(f"    - Member {member_id[-8:]} ({details['status']}) has score: {score_str}")
            if current_score is not None and details.get('status') in ['evaluated', 'promoted']:
                if current_score > best_score:
                    best_score = current_score
                    best_member_id = member_id
        
        # 只有分数达到阈值的理论才能晋级
        if best_member_id and best_score >= min_score:
            promoted_ids.append(best_member_id)
            # 只有当理论还没有被标记为promoted时才更新状态
            if manifest["theories"][best_member_id]["status"] != "promoted":
                manifest["theories"][best_member_id]["status"] = f"promoted_to_gen_{current_gen + 1}"
            print(f"    -> Promoting {best_member_id[-8:]} with score {best_score:.3f}")
        elif best_member_id:
            print(f"    -> Best member {best_member_id[-8:]} with score {best_score:.3f} did not meet threshold {min_score:.3f}")

    print(f"[MANIFEST] Selected {len(promoted_ids)} theories for promotion (threshold: {min_score:.3f})")
    return promoted_ids 