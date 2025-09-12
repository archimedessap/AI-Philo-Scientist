#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contradiction Graph Builder (CNS-Lite)

Builds a lightweight contradiction/network view from pairwise analyses.
- Inputs: list of pairwise contradiction analyses (as produced by ContradictionAnalyzer)
- Outputs: a graph dict with nodes (theories), edges (contradictions), and axes metadata

LLM-first design: If an LLM interface is provided and available, it can be
asked to merge/normalize axes and cluster synonyms. Otherwise, a deterministic
fallback performs simple normalization by lowercasing and trimming.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Set, Tuple
import re


def _norm_axis(name: str) -> str:
    if not name:
        return "unknown"
    s = name.strip().lower()
    # quick aliasing
    aliases = {
        "wave function reality": "wave_function_reality",
        "wavefunction reality": "wave_function_reality",
        "measurement process": "measurement_process",
        "observer role": "observer_role",
        "determinism": "determinism",
        "non locality": "non_locality",
        "non-locality": "non_locality",
        "mathematical formalism": "mathematical_formalism",
        "ontological status": "ontological_status",
        "quantum classical boundary": "quantum_classical_boundary",
    }
    s = aliases.get(s, s)
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    return s or "unknown"


class ContradictionGraphBuilder:
    def __init__(self, llm_interface=None):
        self.llm = llm_interface

    def build_graph(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        axes: Dict[str, Dict[str, Any]] = {}
        theories: Set[str] = set()
        edges: List[Dict[str, Any]] = []

        # Collect raw axes and edges
        for a in analyses:
            t1 = a.get("theory1") or a.get("theory_i")
            t2 = a.get("theory2") or a.get("theory_j")
            if not t1 or not t2:
                continue
            theories.update([t1, t2])
            for c in a.get("contradictions", []):
                axis = _norm_axis(c.get("dimension", ""))
                imp = c.get("importance_score", 5)
                pos1 = str(c.get("theory1_position", "")).strip()
                pos2 = str(c.get("theory2_position", "")).strip()
                tension = str(c.get("core_tension", "")).strip()
                edges.append({
                    "theory_i": t1,
                    "theory_j": t2,
                    "axis": axis,
                    "theory_i_position": pos1,
                    "theory_j_position": pos2,
                    "tension": tension,
                    "importance": imp,
                })
                if axis not in axes:
                    axes[axis] = {
                        "name": axis,
                        "type": "categorical",
                        "domain": [],
                        "aliases": [],
                    }

        # Simple aggregation of positions to suggest domains
        axis_values: Dict[str, Set[str]] = {ax: set() for ax in axes}
        for e in edges:
            if e["theory_i_position"]:
                axis_values[e["axis"]].add(e["theory_i_position"])
            if e["theory_j_position"]:
                axis_values[e["axis"]].add(e["theory_j_position"])
        for ax, vals in axis_values.items():
            # Deduplicate and trim long entries
            domain = []
            for v in vals:
                v2 = v.strip()
                if not v2:
                    continue
                # keep succinct values only; otherwise store as notes
                domain.append(v2 if len(v2) <= 128 else (v2[:120] + "…"))
            axes[ax]["domain"] = sorted(domain)

        # Build nodes with rough axis position notes (non-numeric)
        node_map: Dict[str, Dict[str, Any]] = {}
        for t in sorted(theories):
            node_map[t] = {"name": t, "axis_positions": {}, "notes": []}
        for e in edges:
            t1, t2, ax = e["theory_i"], e["theory_j"], e["axis"]
            if e["theory_i_position"] and ax in axes:
                node_map[t1]["axis_positions"].setdefault(ax, set()).add(e["theory_i_position"])
            if e["theory_j_position"] and ax in axes:
                node_map[t2]["axis_positions"].setdefault(ax, set()).add(e["theory_j_position"])
        # Convert sets to lists
        for n in node_map.values():
            for ax, s in list(n["axis_positions"].items()):
                n["axis_positions"][ax] = sorted(list(s))

        graph = {
            "nodes": {"theories": list(node_map.values())},
            "edges": {"contradictions": edges},
            "axes": list(axes.values()),
            "meta": {"builder": "cns-lite", "version": "0.1"},
        }
        return graph

