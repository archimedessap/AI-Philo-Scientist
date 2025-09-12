#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNS-Lite Orchestrator

Thin wrapper to run the network synthesis method via the generation hub with
minimal flags and produce all intermediate artifacts under a timestamped dir.

Usage:
  python3 scripts/agent_orchestrator.py \
    --theories_dir demo/theories \
    --output_root output_unified_$(date +%Y%m%d_%H%M%S) \
    --model_source google --model_name gemini-2.5-flash \
    [--max_pairs 10] [--relaxation_budget 6] [--dry_run]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]))
from theory_generation.generation_hub import get_generation_hub


def main():
    ap = argparse.ArgumentParser(description="Run CNS-Lite network synthesis")
    ap.add_argument("--theories_dir", type=str, default="demo/theories")
    ap.add_argument("--output_root", type=str, default="output_unified_cns")
    ap.add_argument("--max_pairs", type=int, default=10)
    ap.add_argument("--relaxation_budget", type=int, default=6)
    ap.add_argument("--model_source", type=str, default="google")
    ap.add_argument("--model_name", type=str, default="gemini-2.5-flash")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    hub = get_generation_hub()
    result = hub.generate_theories(
        method="network",
        theories_dir=str(Path(args.theories_dir).resolve()),
        output_dir=str(out_root.resolve()),
        max_pairs=args.max_pairs,
        relaxation_budget=args.relaxation_budget,
        model_source=args.model_source,
        model_name=args.model_name,
        dry_run=args.dry_run,
    )

    print("=== CNS-Lite run completed ===")
    print(result)


if __name__ == "__main__":
    main()
