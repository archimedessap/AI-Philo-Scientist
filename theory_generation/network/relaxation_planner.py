#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relaxation Planner (CNS-Lite)

Given a contradiction graph, propose a minimal set of concept relaxations and
a parameterized super-space to enlarge the theory space while targeting high-
priority tensions. Designed to be LLM-first with a deterministic fallback.
"""

from __future__ import annotations

from typing import Dict, Any, List


GENERIC_RELAXATIONS = {
    "wave_function_reality": [
        ("ontic↔epistemic", "dual-aspect or layered (ontic potentiality + epistemic map)"),
        ("absolute↔relational", "relational-perspectival realism"),
    ],
    "measurement_process": [
        ("collapse↔unitary", "thresholded decoherence / transactional completion / coherence crystallization"),
    ],
    "observer_role": [
        ("human↔system", "agent as physical system: memory + self-reference + irreversible state"),
    ],
    "non_locality": [
        ("causal↔holistic", "holistic relational potentials with no-signalling"),
    ],
    "mathematical_formalism": [
        ("standard↔modified", "unitary core + context-triggered weak nonlinearity/stochasticity in extremes"),
    ],
}


class RelaxationPlanner:
    def __init__(self, llm_interface=None):
        self.llm = llm_interface

    def plan(self, graph: Dict[str, Any], budget: int = 6) -> Dict[str, Any]:
        # Rank axes by accumulated importance across edges
        axis_importance: Dict[str, float] = {}
        for e in graph.get("edges", {}).get("contradictions", []):
            ax = e.get("axis")
            imp = float(e.get("importance", 5))
            axis_importance[ax] = axis_importance.get(ax, 0.0) + imp
        ranked_axes = sorted(axis_importance.items(), key=lambda x: x[1], reverse=True)

        relaxations: List[Dict[str, Any]] = []
        used = 0
        for ax, _score in ranked_axes:
            if used >= budget:
                break
            candidates = GENERIC_RELAXATIONS.get(ax, [])
            if not candidates:
                continue
            # pick the first as minimal relaxation option
            kind, to_desc = candidates[0]
            relaxations.append({
                "axis": ax,
                "from_to": kind,
                "generalization": to_desc,
                "rationale": "Minimize conflict by broadening concept semantics while preserving testability.",
                "cost": 2,
            })
            used += 1

        super_space = []
        for ax in [a for a, _ in ranked_axes[:max(1, budget)]]:
            dom = []
            for a in graph.get("axes", []):
                if a.get("name") == ax:
                    dom = a.get("domain", [])
                    break
            super_space.append({
                "axis": ax,
                "paramization": "categorical",
                "domain": dom + ["generalized"],
                "constraints": [],
            })

        # target top-N most important edges
        edges = list(graph.get("edges", {}).get("contradictions", []))
        edges_sorted = sorted(edges, key=lambda e: float(e.get("importance", 5)), reverse=True)
        satisfaction_targets = []
        for e in edges_sorted[:min(10, len(edges_sorted))]:
            satisfaction_targets.append({
                "theory_i": e.get("theory_i"),
                "theory_j": e.get("theory_j"),
                "axis": e.get("axis"),
                "importance": e.get("importance", 5),
            })

        return {
            "relaxations": relaxations,
            "super_space": super_space,
            "satisfaction_targets": satisfaction_targets,
            "meta": {"planner": "cns-lite", "version": "0.1"},
        }

