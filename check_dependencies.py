#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖检查脚本

检查项目运行所需的所有依赖是否正确安装
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import importlib.util


class DependencyChecker:
    """依赖检查器"""
    
    def __init__(self):
        self.results = {
            'python_version': {},
            'required_packages': {},
            'optional_packages': {},
            'system_dependencies': {},
            'directories': {},
            'config_files': {},
            'api_keys': {}
        }
        self.errors = []
        self.warnings = []
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("🐍 检查Python版本...")
        
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        version_ok = current_version >= required_version
        
        self.results['python_version'] = {
            'current': f"{current_version[0]}.{current_version[1]}",
            'required': f"{required_version[0]}.{required_version[1]}+",
            'status': '✅' if version_ok else '❌'
        }
        
        if not version_ok:
            self.errors.append(
                f"Python版本过低: 需要 {required_version[0]}.{required_version[1]}+，"
                f"当前是 {current_version[0]}.{current_version[1]}"
            )
        
        print(f"  当前版本: {current_version[0]}.{current_version[1]} {self.results['python_version']['status']}")
        return version_ok
    
    def check_required_packages(self) -> bool:
        """检查必需的Python包"""
        print("\n📦 检查必需包...")
        
        required_packages = {
            'numpy': '1.19.0',
            'asyncio': None,  # 标准库，无版本要求
            'json': None,
            'pathlib': None,
            'dataclasses': None,
            'typing': None,
        }
        
        all_ok = True
        
        for package, min_version in required_packages.items():
            status, installed_version = self._check_package(package, min_version)
            
            self.results['required_packages'][package] = {
                'required': min_version or 'any',
                'installed': installed_version or 'not found',
                'status': '✅' if status else '❌'
            }
            
            if not status:
                all_ok = False
                self.errors.append(f"缺少必需包: {package}")
            
            print(f"  {package}: {self.results['required_packages'][package]['status']}")
        
        return all_ok
    
    def check_optional_packages(self) -> bool:
        """检查可选的Python包"""
        print("\n📦 检查可选包...")
        
        optional_packages = {
            'matplotlib': '3.3.0',
            'seaborn': '0.11.0',
            'pandas': '1.1.0',
            'scikit-learn': '0.23.0',
            'torch': '1.7.0',
            'transformers': '4.0.0',
            'openai': '1.0.0',
            'anthropic': None,
            'google-generativeai': None,
            'requests': '2.25.0',
            'aiohttp': '3.7.0',
            'python-dotenv': None,
        }
        
        for package, min_version in optional_packages.items():
            status, installed_version = self._check_package(package, min_version)
            
            self.results['optional_packages'][package] = {
                'required': min_version or 'any',
                'installed': installed_version or 'not found',
                'status': '✅' if status else '⚠️'
            }
            
            if not status:
                self.warnings.append(f"可选包未安装: {package}")
            
            print(f"  {package}: {self.results['optional_packages'][package]['status']}")
        
        return True  # 可选包不影响整体状态
    
    def _check_package(self, package_name: str, min_version: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """检查单个包是否安装"""
        try:
            # 尝试导入包
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return False, None
            
            # 获取版本信息
            module = importlib.import_module(package_name)
            version = None
            
            # 尝试不同的版本属性
            for attr in ['__version__', 'VERSION', 'version']:
                if hasattr(module, attr):
                    version = str(getattr(module, attr))
                    break
            
            # 如果有最小版本要求，进行比较
            if min_version and version:
                # 简单的版本比较（可能需要更复杂的逻辑）
                return version >= min_version, version
            
            return True, version
            
        except ImportError:
            return False, None
        except Exception as e:
            self.warnings.append(f"检查包 {package_name} 时出错: {e}")
            return False, None
    
    def check_directories(self) -> bool:
        """检查必需的目录"""
        print("\n📁 检查目录结构...")
        
        required_dirs = [
            'data',
            'data/theories_v2.1',
            'theory_generation',
            'core_embedding',
            'evaluation',
            'utils',
            'cache',
            'logs',
            'output_clean_evolution'
        ]
        
        all_ok = True
        
        for dir_path in required_dirs:
            exists = Path(dir_path).exists()
            
            self.results['directories'][dir_path] = {
                'exists': exists,
                'status': '✅' if exists else '❌'
            }
            
            if not exists:
                # 尝试创建目录
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    self.results['directories'][dir_path]['status'] = '✅ (已创建)'
                    print(f"  {dir_path}: ✅ (已创建)")
                except Exception as e:
                    all_ok = False
                    self.errors.append(f"无法创建目录 {dir_path}: {e}")
                    print(f"  {dir_path}: ❌")
            else:
                print(f"  {dir_path}: ✅")
        
        return all_ok
    
    def check_config_files(self) -> bool:
        """检查配置文件"""
        print("\n⚙️ 检查配置文件...")
        
        config_files = {
            '.env': 'optional',
            'config.json': 'optional',
            'requirements.txt': 'required',
        }
        
        all_ok = True
        
        for file_path, requirement in config_files.items():
            exists = Path(file_path).exists()
            
            self.results['config_files'][file_path] = {
                'exists': exists,
                'required': requirement,
                'status': '✅' if exists else ('❌' if requirement == 'required' else '⚠️')
            }
            
            if not exists and requirement == 'required':
                all_ok = False
                self.errors.append(f"缺少必需的配置文件: {file_path}")
            elif not exists:
                self.warnings.append(f"缺少可选的配置文件: {file_path}")
            
            print(f"  {file_path}: {self.results['config_files'][file_path]['status']}")
        
        return all_ok
    
    def check_api_keys(self) -> bool:
        """检查API密钥配置"""
        print("\n🔑 检查API密钥...")
        
        # 检查环境变量
        api_keys = {
            'OPENAI_API_KEY': 'OpenAI',
            'ANTHROPIC_API_KEY': 'Anthropic',
            'GOOGLE_API_KEY': 'Google',
            'DEEPSEEK_API_KEY': 'DeepSeek',
            'GROQ_API_KEY': 'Groq',
        }
        
        configured_count = 0
        
        for env_var, service in api_keys.items():
            is_set = env_var in os.environ or self._check_env_file(env_var)
            
            self.results['api_keys'][service] = {
                'env_var': env_var,
                'configured': is_set,
                'status': '✅' if is_set else '⚠️'
            }
            
            if is_set:
                configured_count += 1
            
            print(f"  {service}: {self.results['api_keys'][service]['status']}")
        
        if configured_count == 0:
            self.errors.append("未配置任何API密钥")
            return False
        elif configured_count < len(api_keys):
            self.warnings.append(f"仅配置了 {configured_count}/{len(api_keys)} 个API密钥")
        
        return True
    
    def _check_env_file(self, key: str) -> bool:
        """检查.env文件中的密钥"""
        env_path = Path('.env')
        if not env_path.exists():
            return False
        
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip().startswith(f"{key}="):
                        return True
        except Exception:
            pass
        
        return False
    
    def check_system_dependencies(self) -> bool:
        """检查系统依赖"""
        print("\n🖥️ 检查系统依赖...")
        
        # 检查Git
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            git_installed = result.returncode == 0
            git_version = result.stdout.strip() if git_installed else None
        except:
            git_installed = False
            git_version = None
        
        self.results['system_dependencies']['git'] = {
            'installed': git_installed,
            'version': git_version,
            'status': '✅' if git_installed else '⚠️'
        }
        
        if not git_installed:
            self.warnings.append("Git未安装（用于版本控制）")
        
        print(f"  Git: {self.results['system_dependencies']['git']['status']}")
        
        return True
    
    def generate_report(self) -> str:
        """生成检查报告"""
        report = []
        report.append("=" * 60)
        report.append("依赖检查报告")
        report.append("=" * 60)
        
        # 错误
        if self.errors:
            report.append("\n❌ 错误:")
            for error in self.errors:
                report.append(f"  - {error}")
        
        # 警告
        if self.warnings:
            report.append("\n⚠️ 警告:")
            for warning in self.warnings:
                report.append(f"  - {warning}")
        
        # 摘要
        report.append("\n📊 检查摘要:")
        report.append(f"  - 错误数: {len(self.errors)}")
        report.append(f"  - 警告数: {len(self.warnings)}")
        
        if not self.errors:
            report.append("\n✅ 所有必需依赖已满足！")
        else:
            report.append("\n❌ 存在必需依赖缺失，请先解决错误！")
        
        return "\n".join(report)
    
    def save_report(self, output_path: str = "dependency_check_report.json"):
        """保存详细报告到文件"""
        report = {
            'timestamp': str(Path().cwd()),
            'results': self.results,
            'errors': self.errors,
            'warnings': self.warnings,
            'summary': {
                'error_count': len(self.errors),
                'warning_count': len(self.warnings),
                'all_ok': len(self.errors) == 0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细报告已保存到: {output_path}")
    
    def run_full_check(self) -> bool:
        """运行完整检查"""
        print("🔍 开始依赖检查...\n")
        
        checks = [
            self.check_python_version(),
            self.check_required_packages(),
            self.check_optional_packages(),
            self.check_directories(),
            self.check_config_files(),
            self.check_api_keys(),
            self.check_system_dependencies(),
        ]
        
        all_ok = all(checks) and len(self.errors) == 0
        
        print("\n" + self.generate_report())
        self.save_report()
        
        return all_ok


def generate_requirements_txt():
    """生成requirements.txt文件"""
    requirements = [
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.23.0",
        "openai>=1.0.0",
        "anthropic",
        "google-generativeai",
        "python-dotenv",
        "requests>=2.25.0",
        "aiohttp>=3.7.0",
        "# torch>=1.7.0  # 可选，用于高级嵌入",
        "# transformers>=4.0.0  # 可选，用于高级NLP",
    ]
    
    with open("requirements.txt", 'w') as f:
        f.write("\n".join(requirements))
    
    print("✅ 已生成 requirements.txt 文件")


if __name__ == "__main__":
    checker = DependencyChecker()
    success = checker.run_full_check()
    
    # 如果缺少requirements.txt，询问是否生成
    if not Path("requirements.txt").exists():
        print("\n未找到 requirements.txt 文件。")
        response = input("是否生成建议的 requirements.txt？(y/n): ")
        if response.lower() == 'y':
            generate_requirements_txt()
    
    # 返回适当的退出码
    sys.exit(0 if success else 1)