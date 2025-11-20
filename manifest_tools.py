import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

def initialize_manifest(run_id: str, args: Any) -> Dict[str, Any]:
    """Initialize the run manifest dictionary."""
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config": vars(args) if hasattr(args, '__dict__') else args,
        "theories": {},
        "generations": {}
    }
    return manifest

def save_manifest(manifest: Dict[str, Any], path: str):
    """Save the manifest to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def load_manifest(path: str) -> Dict[str, Any]:
    """Load the manifest from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def register_theory_in_manifest(manifest: Dict[str, Any], theory_data: Dict[str, Any], file_path: str, generation: int) -> str:
    """Register a new theory in the manifest and return its ID."""
    theory_name = theory_data.get('name', 'Unknown')
    # Generate a simple ID if not present
    theory_id = theory_data.get('id')
    if not theory_id:
        # Create a slug from name
        import re
        slug = re.sub(r'[^a-zA-Z0-9]', '_', theory_name).lower()
        theory_id = f"gen{generation}_{slug}_{datetime.now().strftime('%H%M%S')}"
    
    if 'theories' not in manifest:
        manifest['theories'] = {}
        
    manifest['theories'][theory_id] = {
        "id": theory_id,
        "theory_name": theory_name,
        "file_path": str(file_path),
        "generation": generation,
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "scores": {}
    }
    return theory_id

def update_manifest_with_evaluation(manifest: Dict[str, Any], ranking_file: str):
    """Update manifest with evaluation results from a ranking file."""
    try:
        with open(ranking_file, 'r', encoding='utf-8') as f:
            rankings = json.load(f)
            
        # Rankings is a list of dicts
        for rank in rankings:
            # Try to match by theory name or file path
            theory_name = rank.get('theory_name')
            file_path = rank.get('file_path')
            
            matched_id = None
            for tid, tdata in manifest['theories'].items():
                if tdata.get('theory_name') == theory_name:
                    matched_id = tid
                    break
            
            if matched_id:
                manifest['theories'][matched_id]['scores'] = {
                    "success_rate": rank.get('success_rate'),
                    "average_chi2": rank.get('average_chi2'),
                    "experiments_count": rank.get('experiments_count')
                }
                manifest['theories'][matched_id]['score'] = rank.get('success_rate', 0) # Use success rate as main score
                manifest['theories'][matched_id]['status'] = 'evaluated'
                
    except Exception as e:
        print(f"[ERROR] Failed to update manifest with evaluation: {e}")

def select_best_theories_for_next_gen(manifest: Dict[str, Any], current_gen: int, top_n: int, min_score: float) -> List[str]:
    """Select the best theories from the current generation to be parents for the next."""
    candidates = []
    for tid, tdata in manifest['theories'].items():
        if tdata.get('generation') == current_gen:
            score = tdata.get('score', 0)
            if score >= min_score:
                candidates.append((tid, score))
    
    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N
    selected = [c[0] for c in candidates[:top_n]]
    return selected
