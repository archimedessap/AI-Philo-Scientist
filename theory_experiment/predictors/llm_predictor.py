# -*- coding: utf-8 -*-
"""
LLM-driven predictor: build prompt → call LLMInterface → parse JSON
"""
import json, re, hashlib, pickle, textwrap
from pathlib import Path
from theory_generation.llm_interface import LLMInterface  # 你已有

CACHE_DIR = Path(__file__).parent.parent / "formula_cache"
CACHE_DIR.mkdir(exist_ok=True)

JSON_SPEC = (
    "Strictly output ONE line JSON ONLY, like:\n"
    '{"value": 0.123}\n'
    'or {"label": "same_as_QM"}\n'
    'or {"status": "unpredictable"}'
)

def _cache_key(tid, eid):
    return CACHE_DIR / (hashlib.sha1(f"{tid}:{eid}".encode()).hexdigest() + ".pkl")

class LLMPredictor:
    def __init__(self, theory: dict, model_name: str = "deepseek-chat", model_source: str = "openai"):
        self.theory = theory
        self.theory_id = theory.get("id") or hashlib.md5(
            theory["name"].encode()).hexdigest()[:8]
        
        # 如果没有明确指定model_source，则根据model_name推断
        if model_source == "openai" and "deepseek" in model_name.lower():
            model_source = "deepseek"
            
        self.llm = LLMInterface(model_source=model_source, model_name=model_name)

    # -------------- public --------------
    def predict(self, experiment: dict):
        ck = _cache_key(self.theory_id, experiment["id"])
        if ck.exists():
            return pickle.loads(ck.read_bytes())

        prompt = self._build_prompt(experiment)
        try:
            resp = self.llm.query([{"role": "user", "content": prompt}], temperature=0.1)
            pred = self._extract_json(resp)
        except Exception as e:
            print(f"[ERROR] LLM调用失败: {e}")
            return {"status": "error", "message": str(e)}

        pickle.dump(pred, ck.open("wb"))
        return pred

    # -------------- helpers -------------
    def _build_prompt(self, exp: dict) -> str:
        eqs = "\n".join(
            f"- {e.get('name','eq')}: {e['equation']}"
            for e in self.theory
            .get("mathematical_formalism", {})
            .get("key_equations", [])
        ) or "(no explicit equations provided)"

        return textwrap.dedent(f"""
        You are a quantum physicist. Predict the observable for the given experiment.

        # Theory
        {self.theory['name']}
        Equations:
        {eqs}
        Parameters:
        {json.dumps(self.theory.get("parameters", {}), ensure_ascii=False)}

        # Experiment JSON
        {json.dumps(exp, ensure_ascii=False)}

        {JSON_SPEC}
        """).strip()

    def _extract_json(self, resp: str):
        m = re.search(r"\{.*\}", resp)
        if not m:
            return {"status": "unpredictable"}
        try:
            obj = json.loads(m.group())
            if "value" in obj:
                obj["value"] = float(obj["value"])
            return obj
        except Exception:
            return {"status": "unpredictable"}
