import os
import shutil
import glob
from pathlib import Path

# Define the root directory (current directory)
PROJECT_ROOT = Path(os.getcwd())

# Define the whitelist of files and directories to KEEP
# Paths can be absolute or relative to PROJECT_ROOT
KEEP_LIST = [
    # Core Workflow Scripts
    "run_full_cycle.py",
    "run_evolution_cycle.py",
    "run_direct_synthesis.py",
    "run_m3_auto_refinement.py",
    
    # Core Packages (Directories)
    "theory_generation",
    "utils",
    "schemas",
    
    # Demo & Evaluation
    "demo/demo_1.py",
    "demo/auto_role_evaluation.py",
    "demo/instrument_correction.py",
    "demo/experiments",
    
    # Data
    "data/theories_v2.1",
    "cards",
    
    # Results & Diagrams
    "paper/tables/global_ranking_with_models.tex",
    "paper/tables/global_ranking_with_models.csv",
    "global_theory_registry_multi",
    "project_workflow_detailed.svg",
    "generate_workflow_svg.py",
    
    # Documentation & Config
    "requirements.txt",
    "README.md",
    "workflow.dot",
    
    # Self
    "cleanup_project.py",
    ".git",
    ".gitignore"
]

def normalize_path(p):
    return os.path.abspath(os.path.join(PROJECT_ROOT, p))

def is_kept(path):
    abs_path = os.path.abspath(path)
    
    for keep_item in KEEP_LIST:
        keep_abs = normalize_path(keep_item)
        
        # Exact match
        if abs_path == keep_abs:
            return True
            
        # Parent directory match (if keep_item is a directory, keep all children)
        # Check if abs_path starts with keep_abs
        if abs_path.startswith(keep_abs + os.sep):
            return True
            
        # Child match (if we are checking a parent of a kept item, don't delete the parent yet)
        # e.g. if checking "demo", and we keep "demo/demo_1.py", we must keep "demo"
        if keep_abs.startswith(abs_path + os.sep):
            return True
            
    return False

def cleanup():
    print(f"Starting cleanup in: {PROJECT_ROOT}")
    print("WARNING: This will delete all files NOT in the keep list.")
    
    # Collect all files and directories
    all_items = []
    for root, dirs, files in os.walk(PROJECT_ROOT, topdown=False):
        for name in files:
            all_items.append(os.path.join(root, name))
        for name in dirs:
            all_items.append(os.path.join(root, name))
            
    deleted_count = 0
    kept_count = 0
    
    for item in all_items:
        # Skip if it's the script itself or .git
        if os.path.basename(item) == "cleanup_project.py" or ".git" in item.split(os.sep):
            continue
            
        if not is_kept(item):
            try:
                if os.path.isfile(item) or os.path.islink(item):
                    os.remove(item)
                    print(f"[DELETE] File: {os.path.relpath(item, PROJECT_ROOT)}")
                    deleted_count += 1
                elif os.path.isdir(item):
                    # Only delete directory if it's empty (os.rmdir) or force if we know it's not kept
                    # Since we walk bottom-up, files inside should be gone if they weren't kept
                    try:
                        os.rmdir(item)
                        print(f"[DELETE] Dir:  {os.path.relpath(item, PROJECT_ROOT)}")
                        deleted_count += 1
                    except OSError:
                        # Directory not empty, meaning it contains kept files
                        print(f"[KEEP]   Dir:  {os.path.relpath(item, PROJECT_ROOT)} (not empty)")
                        kept_count += 1
            except Exception as e:
                print(f"[ERROR] Could not delete {item}: {e}")
        else:
            # print(f"[KEEP]   {os.path.relpath(item, PROJECT_ROOT)}")
            kept_count += 1
            
    print("-" * 40)
    print(f"Cleanup complete.")
    print(f"Deleted: {deleted_count} items")
    print(f"Kept:    {kept_count} items (approx)")

if __name__ == "__main__":
    cleanup()
