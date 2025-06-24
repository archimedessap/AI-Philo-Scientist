import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# 导入全局理论注册库
try:
    from .global_theory_registry import GlobalTheoryRegistry
except ImportError:
    from global_theory_registry import GlobalTheoryRegistry

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GlobalTheoryVisualizer:
    """
    基于全局理论注册库的可视化对比程序
    """
    
    def __init__(self, registry_dir: str = "global_theory_registry", output_dir: str = "theory_visuals"):
        """
        初始化可视化工具
        
        Args:
            registry_dir: 全局理论注册库目录
            output_dir: 可视化输出目录
        """
        self.registry = GlobalTheoryRegistry(registry_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 颜色配置
        self.colors = {
            "prior": "#4682B4",      # 钢蓝色 - 先验理论
            "evolved": "#FF6347",    # 番茄红 - 演进理论
            "physicist": "#32CD32",  # 酸橙绿 - 物理学家
            "philosopher": "#FFD700", # 金色 - 哲学家
            "mathematician": "#9370DB" # 中紫色 - 数学家
        }
    
    def generate_comprehensive_report(self):
        """生成完整的可视化对比报告"""
        print(f"\n{'='*60}")
        print(f"📊 开始生成全局理论对比报告...")
        print(f"{'='*60}")
        
        # 获取数据
        prior_theories = self.registry.get_theories_by_type("prior")
        evolved_theories = self.registry.get_theories_by_type("evolved")
        
        if not prior_theories and not evolved_theories:
            print("❌ 注册库中没有任何理论，无法生成报告")
            return
        
        print(f"📚 加载数据: {len(prior_theories)} 个先验理论, {len(evolved_theories)} 个演进理论")
        
        # 生成各种图表
        self._create_overall_ranking_chart(prior_theories, evolved_theories)
        self._create_score_distribution_chart(prior_theories, evolved_theories)
        self._create_role_evaluation_radar_chart(prior_theories, evolved_theories)
        self._create_evolution_timeline_chart(evolved_theories)
        self._create_success_rate_comparison(prior_theories, evolved_theories)
        
        # 生成文本报告
        self._generate_text_report(prior_theories, evolved_theories)
        
        print(f"\n🎉 完整报告已生成到: {self.output_dir.absolute()}")
        return self.output_dir
    
    def _create_overall_ranking_chart(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """创建综合排名图表"""
        all_theories = prior_theories + evolved_theories
        if not all_theories:
            return
        
        # 按综合分数排序
        all_theories.sort(key=lambda x: x["composite_score"], reverse=True)
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(all_theories) * 0.4)))
        
        names = []
        scores = []
        colors = []
        
        for theory in all_theories:
            # 处理理论名称长度
            name = theory["theory_name"]
            if len(name) > 35:
                name = name[:32] + "..."
            names.append(name)
            scores.append(theory["composite_score"])
            colors.append(self.colors[theory["source_type"]])
        
        bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 设置坐标轴
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('综合评分 (0-1)', fontsize=12, fontweight='bold')
        ax.set_title('全局理论综合评分排行榜', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1.0)
        
        # 添加分数标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
            
            # 添加排名标签
            ax.text(0.005, bar.get_y() + bar.get_height()/2,
                    f'#{i+1}', ha='left', va='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color=self.colors['prior'], label=f'先验理论 ({len(prior_theories)}个)'),
            mpatches.Patch(color=self.colors['evolved'], label=f'演进理论 ({len(evolved_theories)}个)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        output_file = self.output_dir / "overall_ranking.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 综合排名图表已保存: {output_file}")
    
    def _create_score_distribution_chart(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """创建分数分布对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
        prior_scores = [t["composite_score"] for t in prior_theories] if prior_theories else []
        evolved_scores = [t["composite_score"] for t in evolved_theories] if evolved_theories else []
        
        # 左图：分数分布直方图
        if prior_scores:
            ax1.hist(prior_scores, bins=10, range=(0, 1), alpha=0.7, 
                    color=self.colors['prior'], label=f'先验理论 (n={len(prior_scores)})', edgecolor='black')
        if evolved_scores:
            ax1.hist(evolved_scores, bins=10, range=(0, 1), alpha=0.7, 
                    color=self.colors['evolved'], label=f'演进理论 (n={len(evolved_scores)})', edgecolor='black')
        
        ax1.set_xlabel('综合评分', fontsize=12)
        ax1.set_ylabel('理论数量', fontsize=12)
        ax1.set_title('理论评分分布对比', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 右图：箱线图对比
        data_to_plot = []
        labels = []
        if prior_scores:
            data_to_plot.append(prior_scores)
            labels.append('先验理论')
        if evolved_scores:
            data_to_plot.append(evolved_scores)
            labels.append('演进理论')
        
        if data_to_plot:
            bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = [self.colors['prior'], self.colors['evolved']][:len(data_to_plot)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_ylabel('综合评分', fontsize=12)
        ax2.set_title('理论评分统计对比', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "score_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 分数分布图表已保存: {output_file}")
    
    def _create_role_evaluation_radar_chart(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """创建角色评估雷达图"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 角色标签
        categories = ['物理学家', '哲学家', '数学家']
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合雷达图
        
        # 计算平均分
        def calc_avg_scores(theories):
            if not theories:
                return [0, 0, 0]
            
            physicist_avg = np.mean([t["role_evaluation_results"]["physicist_score"] for t in theories])
            philosopher_avg = np.mean([t["role_evaluation_results"]["philosopher_score"] for t in theories])
            mathematician_avg = np.mean([t["role_evaluation_results"]["mathematician_score"] for t in theories])
            return [physicist_avg, philosopher_avg, mathematician_avg]
        
        prior_avg = calc_avg_scores(prior_theories)
        evolved_avg = calc_avg_scores(evolved_theories)
        
        # 闭合数据
        prior_avg += prior_avg[:1]
        evolved_avg += evolved_avg[:1]
        
        # 绘制雷达图
        if prior_theories:
            ax.plot(angles, prior_avg, 'o-', linewidth=2, label=f'先验理论平均 (n={len(prior_theories)})', 
                   color=self.colors['prior'])
            ax.fill(angles, prior_avg, alpha=0.25, color=self.colors['prior'])
        
        if evolved_theories:
            ax.plot(angles, evolved_avg, 's-', linewidth=2, label=f'演进理论平均 (n={len(evolved_theories)})', 
                   color=self.colors['evolved'])
            ax.fill(angles, evolved_avg, alpha=0.25, color=self.colors['evolved'])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_title('角色评估雷达图对比\n(评分范围: 0-10)', size=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        output_file = self.output_dir / "role_evaluation_radar.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 角色评估雷达图已保存: {output_file}")
    
    def _create_evolution_timeline_chart(self, evolved_theories: List[Dict]):
        """创建演进理论时间线图表"""
        if not evolved_theories:
            print("⚠️ 没有演进理论，跳过时间线图表")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 按运行分组
        runs = {}
        for theory in evolved_theories:
            run_id = theory["run_id"]
            if run_id not in runs:
                runs[run_id] = []
            runs[run_id].append(theory)
        
        # 绘制每个运行的理论
        y_pos = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(runs)))
        
        for i, (run_id, theories) in enumerate(runs.items()):
            theories.sort(key=lambda x: x["composite_score"], reverse=True)
            
            for j, theory in enumerate(theories):
                ax.barh(y_pos, theory["composite_score"], 
                       color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # 添加理论名称
                name = theory["theory_name"]
                if len(name) > 25:
                    name = name[:22] + "..."
                ax.text(0.01, y_pos, f"{name} ({theory['composite_score']:.3f})", 
                       va='center', fontsize=9, fontweight='bold')
                y_pos += 1
            
            # 添加运行分隔线
            if i < len(runs) - 1:
                ax.axhline(y=y_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('综合评分', fontsize=12, fontweight='bold')
        ax.set_ylabel('演进理论', fontsize=12, fontweight='bold')
        ax.set_title('演进理论时间线 (按运行分组)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加运行标签
        y_start = 0
        for i, (run_id, theories) in enumerate(runs.items()):
            y_mid = y_start + len(theories) / 2 - 0.5
            ax.text(-0.05, y_mid, run_id, rotation=90, va='center', ha='right', 
                   fontsize=10, fontweight='bold', color=colors[i])
            y_start += len(theories)
        
        plt.tight_layout()
        output_file = self.output_dir / "evolution_timeline.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 演进时间线图表已保存: {output_file}")
    
    def _create_success_rate_comparison(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """创建实验成功率对比图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        prior_rates = [t["success_rate"] * 100 for t in prior_theories] if prior_theories else []
        evolved_rates = [t["success_rate"] * 100 for t in evolved_theories] if evolved_theories else []
        
        # 创建数据
        data = []
        labels = []
        colors = []
        
        if prior_rates:
            data.append(prior_rates)
            labels.append(f'先验理论\n(n={len(prior_rates)})')
            colors.append(self.colors['prior'])
        
        if evolved_rates:
            data.append(evolved_rates)
            labels.append(f'演进理论\n(n={len(evolved_rates)})')
            colors.append(self.colors['evolved'])
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('实验成功率 (%)', fontsize=12, fontweight='bold')
        ax.set_title('实验成功率对比', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        output_file = self.output_dir / "success_rate_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 实验成功率对比图已保存: {output_file}")
    
    def _generate_text_report(self, prior_theories: List[Dict], evolved_theories: List[Dict]):
        """生成文本报告"""
        report_lines = []
        
        def add_line(text, level=0):
            report_lines.append("  " * level + text)
        
        # 报告标题
        add_line("=" * 70)
        add_line("🏆 全局理论注册库综合对比报告")
        add_line("=" * 70)
        add_line(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line("")
        
        # 统计概览
        stats = self.registry.get_statistics()
        add_line("📊 统计概览:")
        add_line(f"总理论数: {stats['total_theories']}", 1)
        add_line(f"先验理论: {stats['prior_theories']}", 1)
        add_line(f"演进理论: {stats['evolved_theories']}", 1)
        add_line(f"演进运行数: {stats['total_runs']}", 1)
        add_line(f"当前最佳理论: {stats.get('best_theory_name', 'N/A')} (分数: {stats['best_score']:.3f})", 1)
        add_line("")
        
        # 先验理论分析
        if prior_theories:
            add_line("🏛️ 先验理论分析:")
            prior_scores = [t["composite_score"] for t in prior_theories]
            add_line(f"平均综合分: {np.mean(prior_scores):.3f}", 1)
            add_line(f"最高分: {max(prior_scores):.3f}", 1)
            add_line(f"最低分: {min(prior_scores):.3f}", 1)
            add_line(f"标准差: {np.std(prior_scores):.3f}", 1)
            
            # 前三名先验理论
            top_prior = sorted(prior_theories, key=lambda x: x["composite_score"], reverse=True)[:3]
            add_line("前三名先验理论:", 1)
            for i, theory in enumerate(top_prior):
                add_line(f"{i+1}. {theory['theory_name']} (分数: {theory['composite_score']:.3f})", 2)
            add_line("")
        
        # 演进理论分析
        if evolved_theories:
            add_line("🚀 演进理论分析:")
            evolved_scores = [t["composite_score"] for t in evolved_theories]
            add_line(f"平均综合分: {np.mean(evolved_scores):.3f}", 1)
            add_line(f"最高分: {max(evolved_scores):.3f}", 1)
            add_line(f"最低分: {min(evolved_scores):.3f}", 1)
            add_line(f"标准差: {np.std(evolved_scores):.3f}", 1)
            
            # 前三名演进理论
            top_evolved = sorted(evolved_theories, key=lambda x: x["composite_score"], reverse=True)[:3]
            add_line("前三名演进理论:", 1)
            for i, theory in enumerate(top_evolved):
                add_line(f"{i+1}. {theory['theory_name']} (分数: {theory['composite_score']:.3f}, 运行: {theory['run_id']})", 2)
            add_line("")
        
        # 对比结论
        add_line("🔍 对比结论:")
        if prior_theories and evolved_theories:
            prior_avg = np.mean([t["composite_score"] for t in prior_theories])
            evolved_avg = np.mean([t["composite_score"] for t in evolved_theories])
            improvement = evolved_avg - prior_avg
            
            if improvement > 0:
                add_line(f"✅ 演进理论平均分超越先验基准 (+{improvement:.3f})", 1)
            else:
                add_line(f"📈 演进理论平均分低于先验基准 ({improvement:.3f})", 1)
            
            # 检查是否有演进理论超越最佳先验理论
            best_prior = max(prior_theories, key=lambda x: x["composite_score"])
            best_evolved = max(evolved_theories, key=lambda x: x["composite_score"])
            
            if best_evolved["composite_score"] > best_prior["composite_score"]:
                breakthrough = best_evolved["composite_score"] - best_prior["composite_score"]
                add_line(f"🏆 发现突破性理论!", 1)
                add_line(f"{best_evolved['theory_name']} ({best_evolved['composite_score']:.3f}) > {best_prior['theory_name']} ({best_prior['composite_score']:.3f})", 2)
                add_line(f"突破幅度: +{breakthrough:.3f}", 2)
            else:
                gap = best_prior["composite_score"] - best_evolved["composite_score"]
                add_line(f"📊 最佳演进理论尚未超越先验基准 (差距: {gap:.3f})", 1)
        
        add_line("")
        add_line("=" * 70)
        
        # 保存报告
        report_content = "\n".join(report_lines)
        output_file = self.output_dir / "comprehensive_report.txt"
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 文本报告已保存: {output_file}")
        print("\n" + "="*50)
        print("📋 报告摘要:")
        print("="*50)
        print(report_content.split("🔍 对比结论:")[1].split("="*70)[0].strip())


def main():
    """主函数 - 生成全局理论对比报告"""
    visualizer = GlobalTheoryVisualizer()
    visualizer.generate_comprehensive_report()


if __name__ == "__main__":
    main() 