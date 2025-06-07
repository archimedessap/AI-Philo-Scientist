import json, re, os
from theory_generation.llm_interface import LLMInterface

PROMPT = """
You are a quantum-mechanics assistant.

## Theory
{theory}

## Experiment
{experiment}

## Task
(1) Derive the expected observable **{obs}** step by step in LaTeX:
    • Start from the theory's Hamiltonian / evolution law.
    • Insert the experimental setup parameters.
(2) Give the final numeric prediction.

## Output (TWO JSON lines only)
{"derivation":"<LaTeX ... substitutions ...>"}
{"value": <float>}
"""

def extract_json(text:str):
    """抓第二行 JSON (value 行)"""
    for line in text.splitlines():
        line=line.strip()
        if line.startswith("{") and line.endswith("}"):
            try: return json.loads(line)
            except: pass
    raise ValueError("No JSON found")

class LLMPredictor:
    def __init__(self, model="deepseek-chat"):
        self.llm = LLMInterface(model_name=model, temperature=0.2)

    async def predict(self, theory:dict, experiment:dict):
        prompt = PROMPT.format(
            theory=json.dumps(theory, ensure_ascii=False, indent=2),
            experiment=json.dumps(experiment["setup"], ensure_ascii=False, indent=2),
            obs=experiment["observable"]
        )
        resp = await self.llm.query_async([{"role":"user","content":prompt}])
        result = extract_json(resp)
        # 也可把 derivation 行写日志
        return result
