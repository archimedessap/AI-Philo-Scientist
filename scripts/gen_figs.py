#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from typing import List, Dict, Any


def find_run_manifests(root: Path) -> List[Path]:
    candidates = []
    for pattern in [
        "output_*/*/run_manifest.json",
        "output_*/*/*/run_manifest.json",
        "**/run_*/*/run_manifest.json",
    ]:
        candidates.extend(root.glob(pattern))
    seen = set()
    unique = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def extract_score(manifest: Dict[str, Any]) -> float:
    if not isinstance(manifest, dict):
        return float("nan")
    if "score" in manifest and isinstance(manifest["score"], (int, float)):
        return float(manifest["score"])
    if "combined_score" in manifest and isinstance(manifest["combined_score"], (int, float)):
        return float(manifest["combined_score"]) 
    for k, v in manifest.items():
        if isinstance(v, dict) and "combined_score" in v and isinstance(v["combined_score"], (int, float)):
            return float(v["combined_score"])
    return float("nan")


def main():
    if len(sys.argv) < 2:
        print("Usage: gen_figs.py OUT_DIR", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    paths = find_run_manifests(root)
    rows = []
    for p in paths:
        try:
            obj = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        score = extract_score(obj)
        run_id = "/".join(p.parts[-3:]) if p.name == "run_manifest.json" else "/".join(p.parts[-2:])
        if score == score:
            rows.append((run_id, score))

    # Sort desc and take top-N
    rows.sort(key=lambda t: t[1], reverse=True)
    top = rows[:10]

    # Try to render with matplotlib; if unavailable, emit a placeholder text figure using matplotlib fallback
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not top:
            # Empty placeholder figure
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'No runs found', ha='center', va='center')
            ax.axis('off')
        else:
            labels = [t[0] for t in top]
            scores = [t[1] for t in top]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(scores)), scores, color='#4C78A8')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Composite Score')
            ax.set_title('Top-N Composite Scores')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([l.replace('_', '\\_') for l in labels], rotation=30, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.2)
            fig.tight_layout()

        out_pdf = out_dir / 'top_scores.pdf'
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"Wrote {out_pdf}")
    except Exception as e:
        # Fallback: write a minimal text artifact noting the failure
        note = out_dir / 'top_scores.txt'
        note.write_text(f"Could not generate plot: {e}\nFound {len(top)} runs.\n", encoding='utf-8')
        print(f"Matplotlib not available; wrote {note}")


if __name__ == '__main__':
    main()

