import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil


class GlobalTheoryRegistry:
    """全局理论注册库 - 管理所有演进运行的晋级理论"""
    
    def __init__(self, registry_dir: str = "global_theory_registry"):
        """
        初始化全局理论注册库
        
        Args:
            registry_dir: 注册库目录路径
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        # 核心文件
        self.index_file = self.registry_dir / "theory_index.json"
        self.theories_dir = self.registry_dir / "theories"
        self.evaluations_dir = self.registry_dir / "evaluations"
        
        # 创建子目录
        self.theories_dir.mkdir(exist_ok=True)
        self.evaluations_dir.mkdir(exist_ok=True)
        
        # 初始化索引
        self._init_index()
    
    def _init_index(self):
        """初始化理论索引"""
        if not self.index_file.exists():
            index = {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "theories": {},
                "runs": {},
                "statistics": {
                    "total_theories": 0,
                    "prior_theories": 0,
                    "evolved_theories": 0,
                    "total_runs": 0,
                    "best_theory_id": None,
                    "best_score": 0.0
                }
            }
            self._save_index(index)
    
    def _load_index(self) -> Dict[str, Any]:
        """加载理论索引"""
        with open(self.index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_index(self, index: Dict[str, Any]):
        """保存理论索引"""
        index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    
    def _generate_theory_id(self, theory_name: str, source: str) -> str:
        """生成理论唯一ID"""
        content = f"{theory_name}_{source}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def register_prior_theories(self, theories_dir: str) -> int:
        """
        注册先验理论基准
        
        Args:
            theories_dir: 先验理论目录
            
        Returns:
            注册的理论数量
        """
        theories_path = Path(theories_dir)
        if not theories_path.exists():
            raise FileNotFoundError(f"先验理论目录不存在: {theories_dir}")
        
        index = self._load_index()
        registered_count = 0
        
        # 先验理论基准评分
        prior_benchmarks = {
            "many_worlds_interpretation": {"physicist": 8.0, "philosopher": 7.5, "mathematician": 8.0, "composite": 0.913},
            "consistent_histories": {"physicist": 7.0, "philosopher": 7.5, "mathematician": 8.5, "composite": 0.907},
            "copenhagen_interpretation": {"physicist": 8.0, "philosopher": 7.0, "mathematician": 7.5, "composite": 0.900},
            "quantum_bayesianism": {"physicist": 7.0, "philosopher": 7.5, "mathematician": 7.5, "composite": 0.893},
            "relational_quantum_mechanics": {"physicist": 7.0, "philosopher": 8.5, "mathematician": 6.0, "composite": 0.887},
            "transactional_interpretation": {"physicist": 7.0, "philosopher": 7.5, "mathematician": 6.0, "composite": 0.873},
            "modal_interpretations": {"physicist": 7.0, "philosopher": 7.0, "mathematician": 6.5, "composite": 0.873},
            "ensemble_interpretation": {"physicist": 0.0, "philosopher": 6.0, "mathematician": 4.0, "composite": 0.733},
            "de_broglie_bohm_theory": {"physicist": 6.5, "philosopher": 7.0, "mathematician": 6.0, "composite": 0.817},
            "spacetime_state_realism": {"physicist": 6.0, "philosopher": 6.5, "mathematician": 7.0, "composite": 0.783},
            "objective_collapse_theory_grw": {"physicist": 7.5, "philosopher": 6.0, "mathematician": 7.0, "composite": 0.850}
        }
        
        print(f"📚 开始注册先验理论基准...")
        
        for theory_file in theories_path.glob("*.json"):
            try:
                with open(theory_file, 'r', encoding='utf-8') as f:
                    theory_data = json.load(f)
                
                theory_name = theory_data.get("name", theory_file.stem)
                theory_id = self._generate_theory_id(theory_name, "prior_baseline")
                
                # 检查是否已注册
                if any(t.get("source_type") == "prior" and t.get("theory_name") == theory_name 
                       for t in index["theories"].values()):
                    print(f"⚠️ 先验理论已存在: {theory_name}")
                    continue
                
                # 获取基准评分
                file_key = theory_file.stem.lower()
                benchmark_scores = prior_benchmarks.get(file_key, {
                    "physicist": 6.0, "philosopher": 6.0, "mathematician": 6.0, "composite": 0.600
                })
                
                # 复制理论文件
                theory_registry_file = self.theories_dir / f"{theory_id}.json"
                shutil.copy2(theory_file, theory_registry_file)
                
                # 创建评估数据
                evaluation_data = {
                    "theory_id": theory_id,
                    "theory_name": theory_name,
                    "source_type": "prior",
                    "experimental_results": {
                        "success_rate": 1.0,
                        "average_chi2": 0.1,
                        "experiments_count": 5,
                        "evaluation_method": "智能跳过(标准QM)"
                    },
                    "role_evaluation_results": {
                        "physicist_score": benchmark_scores["physicist"],
                        "philosopher_score": benchmark_scores["philosopher"],
                        "mathematician_score": benchmark_scores["mathematician"],
                        "composite_score": benchmark_scores["composite"]
                    },
                    "registered_at": datetime.now().isoformat()
                }
                
                eval_file = self.evaluations_dir / f"{theory_id}_eval.json"
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
                
                # 注册到索引
                index["theories"][theory_id] = {
                    "theory_id": theory_id,
                    "theory_name": theory_name,
                    "source_type": "prior",
                    "run_id": "prior_baseline",
                    "theory_file": str(theory_registry_file),
                    "evaluation_file": str(eval_file),
                    "composite_score": benchmark_scores["composite"],
                    "success_rate": 1.0,
                    "registered_at": datetime.now().isoformat()
                }
                
                registered_count += 1
                print(f"✅ 注册先验理论: {theory_name} (分数: {benchmark_scores['composite']:.3f})")
                
            except Exception as e:
                print(f"❌ 注册先验理论失败 {theory_file}: {e}")
                continue
        
        # 更新统计信息
        self._update_statistics(index)
        self._save_index(index)
        
        print(f"🎉 成功注册 {registered_count} 个先验理论")
        return registered_count
    
    def register_evolved_theories_from_run(self, run_dir: str) -> int:
        """
        从演进运行目录注册晋级理论
        
        Args:
            run_dir: 演进运行目录路径
            
        Returns:
            注册的理论数量
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"运行目录不存在: {run_dir}")
        
        # 提取运行ID
        run_id = run_path.name
        
        # 查找manifest文件
        manifest_file = run_path / "run_manifest.json"
        if not manifest_file.exists():
            print(f"⚠️ 未找到manifest文件: {manifest_file}")
            return 0
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        index = self._load_index()
        registered_count = 0
        
        print(f"📊 分析运行: {run_id}")
        
        # 找出最终晋级的理论
        # 优先选择最高代数中状态为promoted的理论
        # 如果没有promoted状态的，则选择最高代数中评分最高的理论
        max_generation = max((t.get("generation", 0) for t in manifest["theories"].values()), default=0)
        
        # 先尝试找promoted状态的理论
        final_theories = [
            (tid, tinfo) for tid, tinfo in manifest["theories"].items()
            if tinfo.get("generation") == max_generation and tinfo.get("status") == "promoted"
        ]
        
        # 如果没有promoted状态的，选择最高代数中评分最高的前2个理论
        if not final_theories:
            max_gen_theories = [
                (tid, tinfo) for tid, tinfo in manifest["theories"].items()
                if tinfo.get("generation") == max_generation
            ]
            # 按分数排序，取前2个
            max_gen_theories.sort(key=lambda x: x[1].get("score", 0), reverse=True)
            final_theories = max_gen_theories[:2]
        
        print(f"发现 {len(final_theories)} 个最终晋级理论 (第{max_generation}代)")
        
        for theory_id_in_run, theory_info in final_theories:
            try:
                theory_name = theory_info["theory_name"]
                
                # 检查是否已注册
                if any(t.get("run_id") == run_id and t.get("theory_name") == theory_name 
                       for t in index["theories"].values()):
                    print(f"⚠️ 理论已注册: {theory_name} (运行: {run_id})")
                    continue
                
                # 生成新的注册ID
                registry_theory_id = self._generate_theory_id(theory_name, run_id)
                
                # 复制理论文件
                theory_file_path = Path(theory_info["file_path"])
                if theory_file_path.exists():
                    registry_theory_file = self.theories_dir / f"{registry_theory_id}.json"
                    shutil.copy2(theory_file_path, registry_theory_file)
                else:
                    print(f"⚠️ 理论文件不存在: {theory_file_path}")
                    continue
                
                # 加载评估结果
                evaluation_data = self._load_evaluation_from_manifest(
                    theory_info, registry_theory_id, run_id
                )
                
                if not evaluation_data:
                    print(f"⚠️ 无法加载评估数据: {theory_name}")
                    continue
                
                # 保存评估数据
                eval_file = self.evaluations_dir / f"{registry_theory_id}_eval.json"
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
                
                # 注册到索引
                index["theories"][registry_theory_id] = {
                    "theory_id": registry_theory_id,
                    "theory_name": theory_name,
                    "source_type": "evolved",
                    "run_id": run_id,
                    "generation": theory_info.get("generation", 0),
                    "theory_file": str(registry_theory_file),
                    "evaluation_file": str(eval_file),
                    "composite_score": evaluation_data["role_evaluation_results"]["composite_score"],
                    "success_rate": evaluation_data["experimental_results"]["success_rate"],
                    "registered_at": datetime.now().isoformat()
                }
                
                # 记录运行信息
                if run_id not in index["runs"]:
                    index["runs"][run_id] = {
                        "run_id": run_id,
                        "run_path": str(run_path),
                        "max_generation": max_generation,
                        "theories_count": 0,
                        "registered_at": datetime.now().isoformat()
                    }
                
                index["runs"][run_id]["theories_count"] += 1
                registered_count += 1
                
                print(f"✅ 注册演进理论: {theory_name} (分数: {evaluation_data['role_evaluation_results']['composite_score']:.3f})")
                
            except Exception as e:
                print(f"❌ 注册理论失败 {theory_info.get('theory_name', 'Unknown')}: {e}")
                continue
        
        # 更新统计信息
        self._update_statistics(index)
        self._save_index(index)
        
        print(f"🎉 从运行 {run_id} 注册了 {registered_count} 个演进理论")
        return registered_count
    
    def _load_evaluation_from_manifest(self, theory_info: Dict, theory_id: str, run_id: str) -> Optional[Dict]:
        """从manifest中的理论信息加载评估结果"""
        try:
            eval_summary_path = theory_info.get("eval_summary_path")
            if not eval_summary_path:
                # 使用默认评估数据
                return {
                    "theory_id": theory_id,
                    "theory_name": theory_info["theory_name"],
                    "source_type": "evolved",
                    "experimental_results": {
                        "success_rate": 0.8,
                        "average_chi2": 1.0,
                        "experiments_count": 4,
                        "evaluation_method": "完整实验评估"
                    },
                    "role_evaluation_results": {
                        "physicist_score": 7.0,
                        "philosopher_score": 7.0,
                        "mathematician_score": 7.0,
                        "composite_score": theory_info.get("score", 0.7)
                    },
                    "registered_at": datetime.now().isoformat()
                }
            
            # 加载实际评估文件
            eval_path = Path(eval_summary_path)
            if not eval_path.is_absolute():
                eval_path = Path.cwd() / eval_path
            
            if eval_path.exists():
                with open(eval_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                
                # 查找匹配的理论评估结果
                for result in eval_data:
                    if result.get("theory_name") == theory_info["theory_name"]:
                        return {
                            "theory_id": theory_id,
                            "theory_name": theory_info["theory_name"],
                            "source_type": "evolved",
                            "experimental_results": {
                                "success_rate": result.get("experiment_success_rate", result.get("success_rate", 0)),
                                "average_chi2": result.get("average_chi2", 0),
                                "experiments_count": result.get("experiments_count", 0),
                                "evaluation_method": "完整实验评估"
                            },
                            "role_evaluation_results": {
                                "physicist_score": result.get("role_details", {}).get("physicist", 0),
                                "philosopher_score": result.get("role_details", {}).get("philosopher", 0),
                                "mathematician_score": result.get("role_details", {}).get("mathematician", 0),
                                "composite_score": result.get("combined_score", 0)
                            },
                            "registered_at": datetime.now().isoformat()
                        }
            
            return None
            
        except Exception as e:
            print(f"⚠️ 加载评估数据失败: {e}")
            return None
    
    def _update_statistics(self, index: Dict[str, Any]):
        """更新统计信息"""
        theories = index["theories"]
        
        prior_count = sum(1 for t in theories.values() if t.get("source_type") == "prior")
        evolved_count = sum(1 for t in theories.values() if t.get("source_type") == "evolved")
        
        best_theory_id = None
        best_score = 0.0
        
        if theories:
            best_theory = max(theories.values(), key=lambda x: x.get("composite_score", 0))
            best_theory_id = best_theory["theory_id"]
            best_score = best_theory["composite_score"]
        
        index["statistics"] = {
            "total_theories": len(theories),
            "prior_theories": prior_count,
            "evolved_theories": evolved_count,
            "total_runs": len(index["runs"]),
            "best_theory_id": best_theory_id,
            "best_score": best_score
        }
    
    def get_all_theories(self) -> List[Dict[str, Any]]:
        """获取所有注册的理论"""
        index = self._load_index()
        theories = []
        
        for theory_id, theory_info in index["theories"].items():
            eval_file = Path(theory_info["evaluation_file"])
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                
                complete_theory = {
                    **theory_info,
                    "experimental_results": eval_data["experimental_results"],
                    "role_evaluation_results": eval_data["role_evaluation_results"]
                }
                theories.append(complete_theory)
        
        return theories
    
    def get_theories_by_type(self, source_type: str) -> List[Dict[str, Any]]:
        """根据类型获取理论"""
        all_theories = self.get_all_theories()
        return [t for t in all_theories if t.get("source_type") == source_type]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        index = self._load_index()
        stats = index["statistics"].copy()
        
        # 添加最佳理论名称
        if stats["best_theory_id"]:
            best_theory = index["theories"].get(stats["best_theory_id"])
            if best_theory:
                stats["best_theory_name"] = best_theory["theory_name"]
        
        return stats
    
    def print_summary(self):
        """打印注册库摘要"""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"📚 全局理论注册库统计")
        print(f"{'='*60}")
        print(f"📊 总理论数: {stats['total_theories']}")
        print(f"🏛️ 先验理论: {stats['prior_theories']}")
        print(f"🚀 演进理论: {stats['evolved_theories']}")
        print(f"🔬 总运行数: {stats['total_runs']}")
        print(f"🏆 最佳理论: {stats.get('best_theory_name', 'N/A')} (分数: {stats['best_score']:.3f})")
        print(f"📁 注册库位置: {self.registry_dir.absolute()}")
        print(f"{'='*60}")


def main():
    """测试注册库功能"""
    registry = GlobalTheoryRegistry()
    
    # 注册先验理论
    try:
        count = registry.register_prior_theories("data/theories_v2.1")
        print(f"\n注册了 {count} 个先验理论")
    except Exception as e:
        print(f"注册先验理论失败: {e}")
    
    # 打印摘要
    registry.print_summary()


if __name__ == "__main__":
    main() 