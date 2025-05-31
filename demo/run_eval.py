#!/usr/bin/env python3
# coding: utf-8
"""
demo/run_eval.py
评估：单条 C60 双缝实验 × 任意数量理论 × 任意 LLM 后端（全部走 LLM 自动桥接）
"""
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, math, asyncio
from pathlib import Path

# === 1. 读取 CLI 参数 =======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--theory_dir",    default="demo/theories",
                    help="目录下放若干 *.json 理论文件")
parser.add_argument("--experiments",   default="demo/experiments/c60_double_slit.json")
parser.add_argument("--model_name",    default="deepseek-chat")
parser.add_argument("--model_source", default="deepseek",   # 改为model_source
                    help="openai / deepseek / any provider flag that your LLMInterface supports")
parser.add_argument("--temperature",   type=float, default=0.2)
parser.add_argument("--output_dir", default="demo/outputs", 
                    help="保存LLM完整响应的目录")
args = parser.parse_args()

# === 2. 引入你自己的 LLMInterface ==========================================
from theory_generation.llm_interface import LLMInterface   # ← 只用这一份即可


# === 3. 通用 Prompt 模板 + 解析函数 =========================================
PROMPT = """
You are a quantum-mechanics assistant.

## Theory
{theory}

## Experiment setup
{setup}

## Task
1. Derive, in LaTeX, the expected observable **{obs}** step by step.
2. Insert numbers and compute the final value.

## Output (TWO JSON lines only)
{"derivation":"<LaTeX ... substitutions ...>"}
{"value": <float>}
"""

def extract_json(text: str):
    """抓取第二行 JSON（包含 value）"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON found in LLM reply")


# === 4. LLM 预测器封装 ======================================================
class LLMPredictor:
    def __init__(self, model_name, provider, temperature):
        # 移除temperature参数
        self.llm = LLMInterface(
            model_name=model_name,
            model_source=provider  # 已经从provider改为model_source
        )
        # 保存temperature以便在query时使用
        self.temperature = temperature

    async def predict(self, theory: dict, experiment: dict):
        prompt = PROMPT.format(
            theory=json.dumps(theory["formalism"], ensure_ascii=False, indent=2),
            setup=json.dumps(experiment["setup"], ensure_ascii=False, indent=2),
            obs=experiment["observable"]
        )
        # 在这里传入temperature参数
        reply = await self.llm.query_async(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature  # 在调用时使用temperature
        )
        
        # 保存完整响应到文件
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        theory_name = theory.get("name", "unnamed").replace(" ", "_").lower()
        output_file = os.path.join(args.output_dir, f"{theory_name}_{timestamp}.json")
        
        # 创建包含完整信息的输出对象
        output = {
            "theory_name": theory.get("name", "unnamed"),
            "experiment_id": experiment.get("id", "unknown"),
            "prompt": prompt,
            "llm_response": reply,
            "timestamp": timestamp
        }
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 已保存完整响应到: {output_file}")
        
        # 仍然尝试提取value用于评估
        try:
            for line in reply.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    obj = json.loads(line)
                    if "value" in obj:
                        # 在predict方法中，在调用LLM后立即添加这段代码
                        os.makedirs("demo/raw_responses", exist_ok=True)
                        with open(f"demo/raw_responses/{theory_name}_raw.txt", "w") as f:
                            f.write(reply)
                        print(f"已保存原始响应到: demo/raw_responses/{theory_name}_raw.txt")
                        return obj
            # 如果找不到value，创建一个临时对象返回错误信息
            return {"value": float("nan"), "error": "No value found in response"}
        except Exception as e:
            return {"value": float("nan"), "error": str(e)}


# === 5. 评估单个理论 ========================================================
def chi2(pred_val: float, meas: dict):
    return (pred_val - meas["value"])**2 / meas["sigma"]**2

async def evaluate_one(theory_path: Path, experiment: dict, predictor: LLMPredictor):
    theory = json.load(open(theory_path))
    try:
        result = await predictor.predict(theory, experiment)
        if "error" in result:
            return (theory["name"], False, float("inf"), f"Error: {result['error']}")
        val = result["value"]
        c2 = chi2(val, experiment["measured"])
        ok = c2 < 4
        return (theory["name"], ok, c2, val)
    except Exception as e:
        return (theory["name"], False, float("inf"), f"Error: {str(e)}")


# === 6. 主流程 ==============================================================
async def main():
    experiment = json.load(open(args.experiments))
    predictor  = LLMPredictor(args.model_name, args.model_source, args.temperature)

    tasks = [evaluate_one(p, experiment, predictor)
             for p in Path(args.theory_dir).glob("*.json")]

    for name, ok, c2, val in await asyncio.gather(*tasks):
        mark = "✔" if ok else "✖"
        print(f"{mark} {name[:38]:38s} χ²={c2:6.2g}  pred={val}")

if __name__ == "__main__":
    asyncio.run(main())
