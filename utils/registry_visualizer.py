import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys

# 确保可以导入TheoryRegistry
sys.path.append(str(Path(__file__).parent.parent))
from utils.theory_registry import TheoryRegistry

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


class RegistryVisualizer:
    """
    从统一理论库生成对比可视化报告
    """
    def __init__(self, registry_path: str = "theory_registry"):
        """
        初始化可视化工具
        
        Args:
            registry_path: 理论注册库路径
        """
        self.registry = TheoryRegistry(registry_path)
        self.output_dir = Path("registry_visuals")
        self.output_dir.mkdir(exist_ok=True)
        
        self.color_map = {
            "prior": "#4682B4",       # 钢蓝色
            "evolved": "#FF6347",     # 番茄红
            "physicist": "#3CB371",   # 中海绿色
            "philosopher": "#FFD700", # 金色
            "mathematician": "#9370DB" # 中紫色
        }
    
    def generate_full_report(self):
        """生成完整的可视化报告"""
        print(f"\n{'='*60}")
        print(f"📊 开始生成可视化报告...")
        print(f"{'='*60}")
        
        # 加载数据
        prior_theories = self.registry.get_theories_by_type("prior")
        evolved_theories = self.registry.get_theories_by_type("evolved")
        
        if not prior_theories and not evolved_theories:
            print("❌ 注册库中没有任何理论，无法生成报告。")
            return
        
        # 1. 综合分数对比图
        self.create_composite_score_comparison(prior_theories, evolved_theories)
        
        # 2. 角色评分对比图
        self.create_role_score_comparison(prior_theories, evolved_theories)
        
        # 3. 实验成功率对比图
        self.create_success_rate_comparison(prior_theories, evolved_theories)
        
        # 4. 生成总结报告
        self.generate_summary_text_report(prior_theories, evolved_theories)
        
        print(f"\n🎉 报告已生成到: {self.output_dir.absolute()}")
        
    def create_composite_score_comparison(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """创建综合分数对比条形图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 准备数据并排序
        all_theories = sorted(prior_theories + evolved_theories, 
                              key=lambda x: x["composite_score"], reverse=True)
        
        names = [t['theory_name'][:30] + '...' if len(t['theory_name']) > 30 else t['theory_name'] 
                 for t in all_theories]
        scores = [t['composite_score'] for t in all_theories]
        colors = [self.color_map[t['source_type']] for t in all_theories]
        
        bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # 最高分在顶部
        ax.set_xlabel('综合分数 (越高越好)', fontsize=12)
        ax.set_title('所有理论综合分数排名', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
            
        # 添加图例
        legend_elements = [
            mpatches.Patch(color=self.color_map['prior'], label='先验理论'),
            mpatches.Patch(color=self.color_map['evolved'], label='演进理论')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        output_file = self.output_dir / "composite_score_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 综合分数对比图已保存: {output_file}")
    
    def create_role_score_comparison(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """创建角色评分雷达图"""
        if not evolved_theories:
            print("⚠️ 没有演进理论，跳过角色评分对比图。")
            return
            
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 准备数据
        labels = ['物理学家', '哲学家', '数学家']
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # 闭合雷达图
        
        # 先验理论平均分
        prior_avg_scores = [0, 0, 0]
        if prior_theories:
            for t in prior_theories:
                prior_avg_scores[0] += t['role_evaluation_results']['physicist_score']
                prior_avg_scores[1] += t['role_evaluation_results']['philosopher_score']
                prior_avg_scores[2] += t['role_evaluation_results']['mathematician_score']
            prior_avg_scores = [s / len(prior_theories) for s in prior_avg_scores]
        
        prior_avg_scores += prior_avg_scores[:1]
        
        # 演进理论平均分
        evolved_avg_scores = [0, 0, 0]
        for t in evolved_theories:
            evolved_avg_scores[0] += t['role_evaluation_results']['physicist_score']
            evolved_avg_scores[1] += t['role_evaluation_results']['philosopher_score']
            evolved_avg_scores[2] += t['role_evaluation_results']['mathematician_score']
        evolved_avg_scores = [s / len(evolved_theories) for s in evolved_avg_scores]
        evolved_avg_scores += evolved_avg_scores[:1]
        
        # 绘制雷达图
        ax.plot(angles, prior_avg_scores, color=self.color_map['prior'], 
                linewidth=2, linestyle='solid', label='先验理论平均分')
        ax.fill(angles, prior_avg_scores, color=self.color_map['prior'], alpha=0.25)
        
        ax.plot(angles, evolved_avg_scores, color=self.color_map['evolved'], 
                linewidth=2, linestyle='solid', label='演进理论平均分')
        ax.fill(angles, evolved_avg_scores, color=self.color_map['evolved'], alpha=0.25)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title('角色平均评分对比 (0-10分)', size=20, color='black', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        output_file = self.output_dir / "role_score_radar_chart.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 角色评分雷达图已保存: {output_file}")
    
    def create_success_rate_comparison(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """创建实验成功率分布图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        prior_rates = [t['success_rate'] * 100 for t in prior_theories]
        evolved_rates = [t['success_rate'] * 100 for t in evolved_theories]
        
        if prior_rates:
            ax.hist(prior_rates, bins=10, range=(0,100), color=self.color_map['prior'], 
                    alpha=0.7, label='先验理论')
        if evolved_rates:
            ax.hist(evolved_rates, bins=10, range=(0,100), color=self.color_map['evolved'], 
                    alpha=0.7, label='演进理论')
        
        ax.set_xlabel('实验成功率 (%)')
        ax.set_ylabel('理论数量')
        ax.set_title('理论实验成功率分布', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "success_rate_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 实验成功率分布图已保存: {output_file}")
        
    def generate_summary_text_report(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """生成文本总结报告"""
        report_lines = []
        
        def add_line(text, level=0):
            report_lines.append("  " * level + text)
        
        add_line("=" * 60)
        add_line("🏆 理论库综合对比报告 🏆")
        add_line("=" * 60)
        
        # 总体统计
        stats = self.registry.get_registry_statistics()
        add_line("\n📊 注册库概览:")
        for key, value in stats.items():
            if isinstance(value, float):
                add_line(f"  - {key}: {value:.3f}", level=1)
            else:
                add_line(f"  - {key}: {value}", level=1)
        
        # 先验理论分析
        if prior_theories:
            add_line("\n🏛️ 先验理论分析:")
            avg_prior_score = sum(t['composite_score'] for t in prior_theories) / len(prior_theories)
            add_line(f"  - 平均综合分: {avg_prior_score:.3f}", level=1)
            
        # 演进理论分析
        if evolved_theories:
            add_line("\n🚀 演进理论分析:")
            avg_evolved_score = sum(t['composite_score'] for t in evolved_theories) / len(evolved_theories)
            best_evolved = max(evolved_theories, key=lambda x: x['composite_score'])
            add_line(f"  - 平均综合分: {avg_evolved_score:.3f}", level=1)
            add_line(f"  - 最佳演进理论: {best_evolved['theory_name']}", level=1)
            add_line(f"    - 来源运行: {best_evolved['run_id']}", level=2)
            add_line(f"    - 综合分数: {best_evolved['composite_score']:.3f}", level=2)
            add_line(f"    - 实验成功率: {best_evolved['success_rate']:.1%}", level=2)

        # 对比结论
        add_line("\n🔍 对比结论:")
        best_overall = stats.get("best_theory", "N/A")
        add_line(f"  - 当前最佳理论: {best_overall} (分数: {stats.get('best_score', 0):.3f})", level=1)
        
        best_prior = max(prior_theories, key=lambda x: x['composite_score'])
        if evolved_theories:
            best_evolved = max(evolved_theories, key=lambda x: x['composite_score'])
            if best_evolved['composite_score'] > best_prior['composite_score']:
                add_line(f"  - 演进系统已发现超越先验基准的理论!", level=1)
                add_line(f"    - {best_evolved['theory_name']} ({best_evolved['composite_score']:.3f}) > "
                         f"{best_prior['theory_name']} ({best_prior['composite_score']:.3f})", level=2)
            else:
                add_line(f"  - 演进理论正在接近但尚未超越最佳先验理论。", level=1)
        
        report_content = "\n".join(report_lines)
        output_file = self.output_dir / "comparison_summary.txt"
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 文本总结报告已保存: {output_file}")
        print("\n" + report_content)


def main():
    """主函数，生成报告"""
    visualizer = RegistryVisualizer()
    visualizer.generate_full_report()

if __name__ == "__main__":
    main() 