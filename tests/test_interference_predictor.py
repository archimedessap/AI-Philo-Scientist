import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_experiment.predictors import interference
from theory_experiment.experiment_evaluator import ExperimentEvaluator
from theory_experiment.utils.theory_adapter import TheoryAdapter


def _load_experiment(exp_id: str):
    with open("theory_experiment/data/experiments.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item["id"] == exp_id:
            return deepcopy(item)
    raise ValueError(f"Experiment {exp_id} not found")


def test_c60_fallback_same_as_qm():
    exp = _load_experiment("c60_molecule_interf")
    result = interference.predict_c60({"name": "Baseline"}, exp)
    assert result == {"value": exp["std_prediction"]["value"]}


def test_c60_missing_params_for_math_change():
    exp = _load_experiment("c60_molecule_interf")
    theory = {
        "name": "CSL without params",
        "math_relation_to_SQM": {
            "type": "modification",
            "math_change": True
        },
        "parameters": {
            "gamma": {},
            "rc": {}
        }
    }
    result = interference.predict_c60(theory, exp)
    assert result.get("status") == "missing_parameters"


def test_math_change_without_params_structure():
    exp = _load_experiment("c60_molecule_interf")
    theory = {
        "name": "Math change but no params",
        "math_relation_to_SQM": {
            "type": "modification",
            "math_change": True
        }
    }
    result = interference.predict_c60(theory, exp)
    assert result.get("status") == "missing_parameters"


def test_c60_with_gamma_suppression():
    exp = _load_experiment("c60_molecule_interf")
    theory = {
        "name": "CSL variant",
        "parameters": {
            "gamma": {"value": 1e-8},
            "rc": {"value": 1e-7},
        },
    }
    result = interference.predict_c60(theory, exp)
    std_vis = exp["std_prediction"]["value"]
    assert "value" in result
    assert 0.0 <= result["value"] <= std_vis
    assert result["value"] < std_vis


def test_experiment_evaluator_residuals():
    evaluator = ExperimentEvaluator("theory_experiment/data/experiments.json")
    exp = _load_experiment("double_slit_electron")
    evaluator.experiments = [exp]

    class AlwaysQM:
        def predict(self, experiment):
            return {"label": "same_as_QM"}

    theory = {"name": "QM baseline"}
    result = asyncio.run(evaluator.evaluate_theory(theory, AlwaysQM()))

    assert result["failed_experiments"] == []
    assert result["outlier_experiments"] == []
    entry = result["prediction_results"][0]
    std_value = exp["std_prediction"]["value"]
    measured_value = exp["measured"]["value"]
    sigma = exp["measured"]["sigma"]
    assert entry["effective_prediction"] == pytest.approx(std_value)
    assert entry["residual"] == pytest.approx(std_value - measured_value)
    assert entry["z_score"] == pytest.approx((std_value - measured_value) / sigma)
def test_parameter_support_with_range():
    theory = {
        "name": "Range theory",
        "parameters": {
            "gamma": {
                "range": [1e-8, 1e-6]
            }
        }
    }
    assert TheoryAdapter.has_parameter_support(theory) is True
