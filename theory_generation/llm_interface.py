#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM接口

提供与大型语言模型交互的统一接口。
支持多种功能：
1. 理论生成
2. 概念提取
3. 已有理论诠释生成
4. 文本嵌入
"""

import os
import json
import re
import time
import openai
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

class LLMInterface:
    """LLM接口类"""
    
    def __init__(self, model_source="openai", model_name="gpt-3.5-turbo", request_interval=1.0):
        """
        初始化LLM接口
        
        Args:
            model_source: 模型来源
            model_name: 模型名称
            request_interval: 请求间隔时间（秒）
        """
        # 设置模型信息
        self.model_source = model_source
        self.model_name = model_name
        self.request_interval = request_interval
        self.last_request_time = 0
        
        # 从环境变量获取API密钥
        self.api_key_openai = os.environ.get("OPENAI_API_KEY")
        self.api_key_deepseek = os.environ.get("DEEPSEEK_API_KEY")
        
        # 初始化客户端
        if self.model_source.lower() == 'openai':
            print(f"[INFO] 使用OpenAI模型: {model_name}")
            self.openai_client = openai.AsyncOpenAI(api_key=self.api_key_openai)
        elif self.model_source.lower() == 'deepseek':
            print(f"[INFO] 使用DeepSeek模型: {model_name}")
            self.openai_client = openai.AsyncOpenAI(
                api_key=self.api_key_deepseek,
                base_url="https://api.deepseek.com/v1"
            )
        
        # 其他属性
        self.enable_fallback = False
        self.raise_api_error = False
    
    def set_model(self, model_source, model_name):
        """设置模型来源和名称"""
        self.model_source = model_source
        self.model_name = model_name
        
        # 重新初始化客户端
        if model_source.lower() == 'openai':
            print(f"[INFO] 使用OpenAI模型: {model_name}")
            self.openai_client = openai.AsyncOpenAI(api_key=self.api_key_openai)
        elif model_source.lower() == 'deepseek':
            print(f"[INFO] 使用DeepSeek模型: {model_name}")
            self.openai_client = openai.AsyncOpenAI(
                api_key=self.api_key_deepseek,
                base_url="https://api.deepseek.com/v1"
            )
        else:
            print(f"[WARN] 未知模型来源: {model_source}")
    
    async def _wait_for_rate_limit(self):
        """等待请求间隔，避免速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval and self.last_request_time > 0:
            wait_time = self.request_interval - time_since_last
            print(f"[INFO] 频率限制: 等待 {wait_time:.2f} 秒...")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def query_async(self, messages, temperature=0.7, model_source=None, model_name=None):
        """
        异步查询LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            model_source: 模型来源，如果为None则使用默认值
            model_name: 模型名称，如果为None则使用默认值
        
        Returns:
            str: LLM回复
        """
        # 使用提供的参数，如果未提供则使用默认值
        source = model_source if model_source else self.model_source
        name = model_name if model_name else self.model_name
        
        # 等待速率限制
        await self._wait_for_rate_limit()
        
        try:
            print(f"[INFO] 调用API: {source}/{name}")
            response = await self.openai_client.chat.completions.create(
                model=name,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
                
        except Exception as e:
            error_message = f"调用 {source}-{name} API失败: {str(e)}"
            print(f"[ERROR] {error_message}")
            
            if self.raise_api_error:
                raise Exception(error_message)
            return f"错误: {error_message}"
    
    def query(self, messages, temperature=0.7, model_source=None, model_name=None):
        """
        同步查询LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            model_source: 模型来源，如果为None则使用默认值
            model_name: 模型名称，如果为None则使用默认值
        
        Returns:
            str: LLM回复
        """
        # 使用提供的参数，如果未提供则使用默认值
        source = model_source if model_source else self.model_source
        name = model_name if model_name else self.model_name
        
        # 创建同步客户端
        if source.lower() == 'openai':
            client = openai.OpenAI(api_key=self.api_key_openai)
        elif source.lower() == 'deepseek':
            client = openai.OpenAI(
                api_key=self.api_key_deepseek,
                base_url="https://api.deepseek.com/v1"
            )
        else:
            raise ValueError(f"不支持的模型来源: {source}")
        
        try:
            print(f"[INFO] 同步调用API: {source}/{name}")
            response = client.chat.completions.create(
                model=name,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
                
        except Exception as e:
            error_message = f"调用 {source}-{name} API失败: {str(e)}"
            print(f"[ERROR] {error_message}")
            
            if self.raise_api_error:
                raise Exception(error_message)
            return f"错误: {error_message}"
    
    def extract_json(self, text):
        """
        从 LLM 回复中提取第一行有效 JSON。
        支持：
          • 单行 JSON
          • 先 derivation 行、再 value 行（两行 JSON）
        返回 Python dict；若找不到则返回 None。
        """
        if text is None:
            return None
            
        # 方式 1：逐行找以 { 开头的片段
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # 方式 2：正则搜索整段 {...}
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        print(f"[WARN] 无法从文本中提取JSON: {text[:100]}...")
        return None
    
    def get_current_model_info(self):
        """获取当前使用的模型信息"""
        return {
            "model_source": self.model_source,
            "model_name": self.model_name,
            "source": self.model_source,
            "name": self.model_name
        }

    async def get_embedding(self, text, model_name=None):
        """
        获取文本的嵌入向量
        
        Args:
            text: 要嵌入的文本
            model_name: 嵌入模型名称，如果为None则使用默认值
        
        Returns:
            list: 嵌入向量
        """
        # 使用提供的模型名称或默认值
        embedding_model = model_name if model_name else self.model_name
        
        # 等待速率限制
        await self._wait_for_rate_limit()
        
        try:
            if self.model_source.lower() == 'openai':
                print(f"[INFO] 使用OpenAI获取嵌入向量: {embedding_model}")
                
                # 使用OpenAI嵌入API
                client = openai.AsyncOpenAI(api_key=self.api_key_openai)
                response = await client.embeddings.create(
                    model=embedding_model,
                    input=text
                )
                return response.data[0].embedding
                
            elif self.model_source.lower() == 'deepseek':
                print(f"[INFO] 使用DeepSeek获取嵌入向量: {embedding_model}")
                
                # 使用DeepSeek嵌入API
                client = openai.AsyncOpenAI(
                    api_key=self.api_key_deepseek,
                    base_url="https://api.deepseek.com/v1"
                )
                response = await client.embeddings.create(
                    model=embedding_model,
                    input=text
                )
                return response.data[0].embedding
                
            else:
                raise ValueError(f"不支持的模型来源用于嵌入: {self.model_source}")
                
        except Exception as e:
            error_message = f"获取嵌入向量失败: {str(e)}"
            print(f"[ERROR] {error_message}")
            
            if self.raise_api_error:
                raise Exception(error_message)
            return None
            
    # ------ 新增方法：用于文献概念提取的功能 ------
            
    def extract_concepts_from_text(self, text, source="unknown"):
        """
        从文本中提取科学概念和公式
        
        Args:
            text: 要提取概念的文本内容
            source: 文本来源标识
            
        Returns:
            Tuple[List[Dict], List[Dict]]: 提取的概念和公式列表
        """
        print(f"[INFO] 从文本中提取概念和公式 (来源: {source})")
        
        # 构建提示词
        prompt = f"""
        从以下文本中提取量子力学相关的概念和数学公式。

        对于每个概念，提供：
        - 名称
        - 定义/解释
        - 相关领域（物理学/哲学/数学等）

        对于每个公式，提供：
        - 名称
        - 数学表达式
        - 解释/含义
        - 相关变量说明

        以JSON格式返回，结构如下：
        {{
          "concepts": [
            {{
              "name": "概念名称",
              "description": "详细解释",
              "domain": "相关领域"
            }}
          ],
          "formulas": [
            {{
              "name": "公式名称",
              "expression": "数学表达式",
              "description": "含义解释",
              "variables": [
                {{
                  "symbol": "符号",
                  "meaning": "含义"
                }}
              ]
            }}
          ]
        }}

        文本内容:
        {text}
        """
        
        # 发送请求到LLM
        try:
            response = self.query(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # 使用较低的温度以获得更确定的结果
            )
            
            # 解析响应
            extracted_data = self.extract_json(response)
            
            if not extracted_data:
                print("[WARN] LLM返回的数据无法解析为JSON")
                return [], []
                
            # 提取概念和公式
            concepts = extracted_data.get("concepts", [])
            formulas = extracted_data.get("formulas", [])
            
            # 添加来源信息
            for concept in concepts:
                concept["source"] = source
            for formula in formulas:
                formula["source"] = source
            
            print(f"[INFO] 提取了 {len(concepts)} 个概念和 {len(formulas)} 个公式")
            return concepts, formulas
            
        except Exception as e:
            print(f"[ERROR] 概念提取失败: {str(e)}")
            return [], []
            
    # ------ 新增方法：用于生成已有诠释理论的功能 ------
    
    def generate_existing_theory(self, theory_name):
        """
        生成已有的量子力学诠释理论
        
        Args:
            theory_name: 要生成的理论名称
            
        Returns:
            Dict: 生成的理论数据
        """
        print(f"[INFO] 生成已有理论: {theory_name}")
        
        # 构建提示词
        prompt = f"""
        请详细描述量子力学中的"{theory_name}"诠释理论，包括以下内容：
        
        1. 理论的主要提出者和历史背景
        2. 核心观点和基本假设
        3. 理论的数学框架
        4. 该理论如何解释量子测量问题
        5. 该理论如何处理量子叠加和纠缠
        6. 与其他量子诠释理论的对比
        7. 该理论的优势和局限性
        
        请以JSON格式返回，结构如下：
        {{
          "theory_name": "理论名称",
          "developers": ["提出者1", "提出者2"],
          "year": "提出年份",
          "description": "理论总体描述",
          "core_principles": ["核心原则1", "核心原则2"],
          "mathematical_framework": "数学框架描述",
          "key_concepts": [
            {{
              "name": "概念名称",
              "description": "详细解释"
            }}
          ],
          "treatment_of_measurement": "测量问题的处理方式",
          "treatment_of_entanglement": "纠缠问题的处理方式",
          "advantages": ["优势1", "优势2"],
          "limitations": ["局限性1", "局限性2"],
          "comparisons": [
            {{
              "other_theory": "其他理论名称",
              "differences": "主要区别"
            }}
          ]
        }}
        """
        
        # 发送请求到LLM
        try:
            response = self.query(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            # 解析响应
            theory_data = self.extract_json(response)
            
            if not theory_data:
                print(f"[WARN] LLM返回的理论数据无法解析为JSON")
                return {"theory_name": theory_name, "error": "生成的理论数据无法解析"}
                
            # 添加来源信息
            theory_data["source"] = "LLM生成的已有理论"
            
            return theory_data
            
        except Exception as e:
            print(f"[ERROR] 生成理论 {theory_name} 失败: {str(e)}")
            return {"theory_name": theory_name, "error": str(e)}
            
    def get_theory_comparison(self, theories: List[str]):
        """
        获取多个量子力学诠释理论的比较分析
        
        Args:
            theories: 要比较的理论名称列表
            
        Returns:
            Dict: 理论比较结果
        """
        print(f"[INFO] 比较理论: {', '.join(theories)}")
        
        # 构建提示词
        prompt = f"""
        请比较以下量子力学诠释理论，突出它们之间的关键异同点：
        {', '.join(theories)}
        
        对于每个理论，分析以下方面：
        1. 核心哲学立场
        2. 对测量问题的处理
        3. 对叠加态的解释
        4. 对量子纠缠的解释
        5. 数学形式化方法
        6. 相对于其他理论的优缺点
        
        请以JSON格式返回结果，包含每个理论的总结和两两比较的结果。
        """
        
        # 发送请求到LLM
        try:
            response = self.query(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # 尝试提取JSON
            comparison_data = self.extract_json(response)
            
            if not comparison_data:
                print(f"[WARN] LLM返回的比较数据无法解析为JSON")
                return {"error": "生成的比较数据无法解析", "raw_response": response}
                
            return comparison_data
            
        except Exception as e:
            print(f"[ERROR] 比较理论失败: {str(e)}")
            return {"error": str(e)}
