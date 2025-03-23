#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 多模态处理
===================
本示例讲解如何使用LangGraph构建多模态处理系统:
1. 图像处理集成 - 分析和理解图像内容
2. 语音处理集成 - 语音识别和合成
3. 多模态协同理解 - 融合不同模态的信息进行综合分析
4. 统一响应生成 - 根据多模态输入生成连贯回应

WHY - 设计思路:
1. 现实世界的信息是多模态的，单一模态处理限制了AI系统的应用范围
2. 多模态协同理解能够提供更全面的信息分析能力
3. 图像和语音等非文本信息需要专门的处理节点
4. 不同模态之间的信息需要有效融合以生成一致的响应
5. 模块化设计使系统能够灵活扩展支持新的模态

HOW - 实现方式:
1. 分别定义图像和语音处理的专门节点
2. 使用多模态模型进行信息融合
3. 设计统一的状态结构存储各模态信息
4. 利用LangGraph的流程控制能力协调各节点处理顺序
5. 通过模拟实现各模态处理功能(实际应用中可替换为真实模型)

WHAT - 功能作用:
通过本示例，你将学习如何构建能够处理文本、图像和语音等多种模态输入的系统。
这类系统可用于多模态对话、内容分析、辅助决策等场景，大大拓展了LLM应用的边界。

学习目标:
- 理解多模态处理的基本架构设计
- 掌握不同模态处理节点的实现方法
- 学习多模态信息融合的技术
- 构建完整的多模态处理流程
"""

from typing import TypedDict, List, Dict, Any, Optional, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import re
import os
import random
import base64
from datetime import datetime

# 模拟导入模块，实际应用中需要安装相应的库
# from PIL import Image
# import numpy as np
# import soundfile as sf
# import librosa

# =================================================================
# 第1部分: 基础组件 - 状态定义和模拟多模态模型
# =================================================================

class MultimodalState(TypedDict):
    """多模态处理系统状态定义
    
    WHY - 设计思路:
    1. 需要统一管理不同模态的输入和处理结果
    2. 需要跟踪整个处理流程的状态
    3. 需要保存中间处理结果供后续节点使用
    
    HOW - 实现方式:
    1. 使用TypedDict定义类型安全的状态结构
    2. 为每种模态设计对应的字段
    3. 保存原始输入和处理后的结果
    4. 设计统一的响应字段
    
    WHAT - 功能作用:
    提供统一的状态结构，存储各模态的输入和处理结果，
    使系统能够有效地管理和协调多模态信息处理流程
    """
    # 输入字段
    query: str  # 文本查询
    image_path: Optional[str]  # 图像路径
    audio_path: Optional[str]  # 音频路径
    
    # 处理结果字段
    image_analysis: Optional[Dict[str, Any]]  # 图像分析结果
    audio_transcript: Optional[str]  # 音频转写结果
    
    # 融合和响应字段
    multimodal_context: Optional[Dict[str, Any]]  # 多模态融合结果
    response: Optional[str]  # 最终响应
    error: Optional[str]  # 错误信息

# 模拟视觉模型
class MockVisionModel:
    """模拟视觉分析模型
    
    WHY - 设计思路:
    1. 需要模拟图像分析功能用于示例
    2. 实际应用会替换为真实的视觉模型API
    
    HOW - 实现方式:
    1. 基于图像路径或文件名生成模拟分析结果
    2. 创建结构化的图像描述和对象检测结果
    
    WHAT - 功能作用:
    提供模拟的图像分析功能，用于演示多模态处理流程
    """
    def __init__(self):
        # 预定义一些图像类别和对象
        self.image_categories = [
            "自然风景", "城市建筑", "人物肖像", "动物", "食物", 
            "交通工具", "室内场景", "运动", "艺术作品"
        ]
        
        self.common_objects = [
            "人", "汽车", "建筑", "树木", "桌子", "椅子", "电脑", 
            "手机", "书本", "动物", "食物", "衣物", "天空", "道路"
        ]
        
        # 特定场景的预设描述
        self.scene_descriptions = {
            "nature": "一幅美丽的自然风景，绿树成荫，蓝天白云。",
            "city": "一座现代化城市的天际线，高楼林立，灯光闪烁。",
            "food": "一盘精美的菜肴，色香味俱全，摆盘精致。",
            "portrait": "一张人物肖像，表情生动，细节清晰。",
            "animal": "一只可爱的动物，姿态优美，神态自然。"
        }
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """分析图像内容(模拟)
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            模拟的图像分析结果
        """
        # 根据文件名生成一致的随机结果
        image_name = os.path.basename(image_path).lower()
        
        # 确定图像类别
        if "nature" in image_name or "landscape" in image_name:
            category = "自然风景"
            description = self.scene_descriptions["nature"]
            objects = ["树木", "山脉", "河流", "天空", "云朵"]
        elif "city" in image_name or "urban" in image_name:
            category = "城市建筑"
            description = self.scene_descriptions["city"]
            objects = ["建筑", "道路", "汽车", "人", "灯光"]
        elif "food" in image_name or "dish" in image_name:
            category = "食物"
            description = self.scene_descriptions["food"]
            objects = ["盘子", "食物", "餐具", "装饰", "桌面"]
        elif "person" in image_name or "portrait" in image_name:
            category = "人物肖像"
            description = self.scene_descriptions["portrait"]
            objects = ["人", "脸部", "衣物", "背景", "饰品"]
        elif "animal" in image_name or "pet" in image_name:
            category = "动物"
            description = self.scene_descriptions["animal"]
            objects = ["动物", "草地", "天空", "树木", "水"]
        else:
            # 随机生成结果
            category = random.choice(self.image_categories)
            description = f"这是一张{category}图像，包含多个视觉元素。"
            objects = random.sample(self.common_objects, min(5, len(self.common_objects)))
        
        # 为对象添加置信度
        detected_objects = [
            {"name": obj, "confidence": round(random.uniform(0.7, 0.99), 2)} 
            for obj in objects
        ]
        
        # 生成颜色信息
        dominant_colors = [
            {"color": color, "percentage": round(random.uniform(0.1, 0.5), 2)} 
            for color in random.sample(["红", "绿", "蓝", "黄", "黑", "白"], 3)
        ]
        
        # 返回结构化的分析结果
        return {
            "category": category,
            "description": description,
            "objects": detected_objects,
            "colors": dominant_colors,
            "image_quality": round(random.uniform(0.5, 1.0), 2),
            "analysis_timestamp": datetime.now().isoformat()
        }

# 模拟语音模型
class MockSpeechModel:
    """模拟语音处理模型
    
    WHY - 设计思路:
    1. 需要模拟语音识别功能用于示例
    2. 实际应用会替换为真实的语音API
    
    HOW - 实现方式:
    1. 基于音频路径或文件名生成模拟转写结果
    2. 创建带有置信度的文本转写
    
    WHAT - 功能作用:
    提供模拟的语音识别功能，用于演示多模态处理流程
    """
    def __init__(self):
        # 预定义一些语音内容模板
        self.speech_templates = {
            "query": [
                "请告诉我这张图片里有什么内容",
                "这张照片是什么时候拍的",
                "能描述一下这个场景吗",
                "这个图像中有几个人",
                "图中最显眼的物体是什么"
            ],
            "greeting": [
                "你好，我想了解一下这张图片",
                "早上好，能帮我分析这个图像吗",
                "嗨，请问你能识别这张照片吗",
                "您好，请告诉我这是什么场景"
            ],
            "instruction": [
                "请分析这张图片并给出详细描述",
                "我需要知道图中包含的主要元素",
                "帮我识别这个图像中的所有物体",
                "需要一份关于这个场景的完整报告"
            ]
        }
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """转写音频内容(模拟)
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            模拟的音频转写结果
        """
        # 根据文件名选择内容类型
        audio_name = os.path.basename(audio_path).lower()
        
        if "query" in audio_name or "question" in audio_name:
            category = "query"
        elif "greeting" in audio_name or "hello" in audio_name:
            category = "greeting"
        elif "instruction" in audio_name or "command" in audio_name:
            category = "instruction"
        else:
            # 随机选择一个类别
            category = random.choice(list(self.speech_templates.keys()))
        
        # 从选定类别中随机选择一个模板
        transcript = random.choice(self.speech_templates[category])
        
        # 生成音频特征
        audio_features = {
            "duration": round(random.uniform(1.5, 10.0), 2),  # 秒
            "language": "中文",
            "speaker_count": random.randint(1, 2),
            "background_noise": random.choice(["低", "中", "高"]),
            "sampling_rate": random.choice([16000, 22050, 44100])
        }
        
        # 返回结构化的转写结果
        return {
            "transcript": transcript,
            "confidence": round(random.uniform(0.8, 0.98), 2),
            "audio_features": audio_features,
            "transcription_timestamp": datetime.now().isoformat()
        }
    
    def synthesize(self, text: str) -> Dict[str, Any]:
        """合成语音(模拟)
        
        Args:
            text: 要转换为语音的文本
            
        Returns:
            模拟的语音合成结果
        """
        # 生成模拟的合成结果
        output_path = f"output_speech_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        
        return {
            "output_path": output_path,
            "duration": round(len(text) * 0.1, 2),  # 简单估算持续时间
            "format": "wav",
            "sample_rate": 22050,
            "synthesis_timestamp": datetime.now().isoformat()
        }

# 模拟多模态模型
class MockMultimodalModel:
    """模拟多模态融合模型
    
    WHY - 设计思路:
    1. 需要模拟多模态信息融合功能
    2. 实际应用会替换为真实的多模态模型
    
    HOW - 实现方式:
    1. 整合文本、图像和音频信息
    2. 生成基于多模态输入的综合响应
    
    WHAT - 功能作用:
    提供模拟的多模态融合功能，综合分析不同模态的信息
    """
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """融合多模态信息并生成响应
        
        Args:
            inputs: 包含各模态信息的输入字典
            
        Returns:
            基于多模态信息的响应文本
        """
        text_query = inputs.get("text", "")
        image_context = inputs.get("image_context", {})
        audio_context = inputs.get("audio_context", {})
        
        # 构建提示模板
        template = """
        你是一个多模态AI助手，需要基于以下信息生成回应：
        
        用户查询: {text_query}
        
        {image_info}
        
        {audio_info}
        
        请综合以上所有信息，生成一个全面、连贯的回应。
        """
        
        # 格式化图像信息
        image_info = ""
        if image_context:
            image_info = f"""
            图像分析结果:
            - 类别: {image_context.get('category', '未知')}
            - 描述: {image_context.get('description', '无描述')}
            - 检测到的物体: {', '.join([obj['name'] for obj in image_context.get('objects', [])])}
            - 主要颜色: {', '.join([color['color'] for color in image_context.get('colors', [])])}
            """
        
        # 格式化音频信息
        audio_info = ""
        if audio_context:
            audio_info = f"""
            音频转写结果:
            - 转写内容: {audio_context.get('transcript', '无内容')}
            - 置信度: {audio_context.get('confidence', 0)}
            - 音频时长: {audio_context.get('audio_features', {}).get('duration', 0)}秒
            """
        
        # 创建提示
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(
            text_query=text_query,
            image_info=image_info,
            audio_info=audio_info
        )
        
        # 使用LLM生成响应
        response = self.llm.invoke(messages)
        return response.content

# 初始化模拟模型
vision_model = MockVisionModel()
speech_model = MockSpeechModel()
multimodal_model = MockMultimodalModel()

# =================================================================
# 第2部分: LangGraph核心逻辑 - 节点函数
# =================================================================

def process_image(state: MultimodalState) -> Dict[str, Any]:
    """图像处理节点
    
    WHY - 设计思路:
    1. 需要专门处理图像输入
    2. 需要提取图像中的视觉信息
    3. 分析结果需要结构化以便后续使用
    
    HOW - 实现方式:
    1. 获取图像路径
    2. 调用视觉模型分析图像
    3. 将分析结果添加到状态中
    
    WHAT - 功能作用:
    处理图像输入，提取视觉特征和内容描述，
    为多模态理解提供视觉信息
    
    Args:
        state: 当前多模态状态
        
    Returns:
        更新后的状态字典
    """
    # 检查是否有图像输入
    if not state.get("image_path"):
        return {"error": "没有提供图像路径"}
    
    try:
        # 获取图像路径
        image_path = state["image_path"]
        
        # 调用视觉模型分析图像
        image_analysis = vision_model.analyze(image_path)
        
        # 返回图像分析结果
        return {"image_analysis": image_analysis}
        
    except Exception as e:
        return {"error": f"图像处理错误: {str(e)}"}

def process_audio(state: MultimodalState) -> Dict[str, Any]:
    """音频处理节点
    
    WHY - 设计思路:
    1. 需要专门处理语音输入
    2. 需要将语音转换为文本便于理解
    3. 保留音频特征以捕捉语调和情感
    
    HOW - 实现方式:
    1. 获取音频路径
    2. 调用语音模型转写音频
    3. 将转写结果添加到状态中
    
    WHAT - 功能作用:
    处理语音输入，提取文本内容和音频特征，
    为多模态理解提供语音信息
    
    Args:
        state: 当前多模态状态
        
    Returns:
        更新后的状态字典
    """
    # 检查是否有音频输入
    if not state.get("audio_path"):
        return {}  # 没有音频输入不是错误，直接返回空结果
    
    try:
        # 获取音频路径
        audio_path = state["audio_path"]
        
        # 调用语音模型转写音频
        audio_result = speech_model.transcribe(audio_path)
        
        # 如果没有文本查询但有语音输入，使用转写结果作为查询
        result = {"audio_transcript": audio_result}
        if not state.get("query") and audio_result.get("transcript"):
            result["query"] = audio_result["transcript"]
        
        return result
        
    except Exception as e:
        return {"error": f"音频处理错误: {str(e)}"}

def multimodal_fusion(state: MultimodalState) -> Dict[str, Any]:
    """多模态融合节点
    
    WHY - 设计思路:
    1. 需要整合来自不同模态的信息
    2. 融合后的信息需要保持语义一致性
    3. 需要处理模态之间的互补和冲突
    
    HOW - 实现方式:
    1. 收集各模态的处理结果
    2. 使用多模态模型融合信息
    3. 生成统一的语义表示
    
    WHAT - 功能作用:
    融合文本查询、图像分析和音频转写信息，
    生成综合的多模态上下文，为生成最终响应做准备
    
    Args:
        state: 当前多模态状态
        
    Returns:
        更新后的状态字典
    """
    # 检查是否有查询
    if not state.get("query"):
        return {"error": "没有提供查询文本或可转写的语音"}
    
    # 收集各模态信息
    query = state["query"]
    image_analysis = state.get("image_analysis", {})
    audio_transcript = state.get("audio_transcript", {})
    
    # 构建多模态上下文
    multimodal_context = {
        "text": {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # 添加图像信息
    if image_analysis:
        multimodal_context["image"] = image_analysis
    
    # 添加音频信息
    if audio_transcript:
        multimodal_context["audio"] = audio_transcript
    
    return {"multimodal_context": multimodal_context}

def generate_response(state: MultimodalState) -> Dict[str, Any]:
    """响应生成节点
    
    WHY - 设计思路:
    1. 需要基于融合的多模态信息生成响应
    2. 响应需要考虑用户的原始查询意图
    3. 响应应当融合各模态的信息
    
    HOW - 实现方式:
    1. 获取多模态上下文
    2. 调用多模态模型生成响应
    3. 格式化为用户友好的输出
    
    WHAT - 功能作用:
    基于多模态融合结果生成连贯、全面的回应，
    为用户提供综合考虑了所有输入模态的解答
    
    Args:
        state: 当前多模态状态
        
    Returns:
        更新后的状态字典
    """
    # 检查是否存在错误
    if state.get("error"):
        return {"response": f"处理遇到问题: {state['error']}"}
    
    # 获取多模态上下文
    context = state.get("multimodal_context", {})
    if not context:
        return {"response": "无法生成响应，缺少多模态上下文"}
    
    try:
        # 准备多模态模型的输入
        model_input = {
            "text": context.get("text", {}).get("query", ""),
            "image_context": context.get("image", {}),
            "audio_context": context.get("audio", {})
        }
        
        # 调用多模态模型生成响应
        response = multimodal_model.invoke(model_input)
        
        return {"response": response}
        
    except Exception as e:
        return {"response": f"生成响应时发生错误: {str(e)}"}

def synthesize_audio_response(state: MultimodalState) -> Dict[str, Any]:
    """响应语音合成节点(可选)
    
    WHY - 设计思路:
    1. 用户可能需要语音形式的回应
    2. 文本响应需要转换为自然的语音
    
    HOW - 实现方式:
    1. 获取文本响应
    2. 调用语音合成模型
    3. 保存合成的音频
    
    WHAT - 功能作用:
    将文本响应转换为语音输出，提供多模态的系统输出能力
    
    Args:
        state: 当前多模态状态
        
    Returns:
        更新后的状态字典，包含合成音频路径
    """
    # 检查是否有文本响应
    if not state.get("response"):
        return {}
    
    try:
        # 获取响应文本
        response_text = state["response"]
        
        # 调用语音合成模型
        synthesis_result = speech_model.synthesize(response_text)
        
        return {"audio_response": synthesis_result}
        
    except Exception as e:
        return {"error": f"语音合成错误: {str(e)}"}

# =================================================================
# 第3部分: 图构建与流程控制
# =================================================================

def build_multimodal_graph() -> StateGraph:
    """构建多模态处理系统的工作流图
    
    WHY - 设计思路:
    1. 需要协调不同模态处理节点的执行顺序
    2. 需要根据输入类型选择适当的处理路径
    3. 需要确保信息在各节点间正确流动
    
    HOW - 实现方式:
    1. 创建基于MultimodalState的StateGraph
    2. 添加各个处理节点
    3. 定义节点间的转换逻辑
    4. 设置条件分支处理不同输入组合
    
    WHAT - 功能作用:
    组装完整的多模态处理系统，定义信息流和处理逻辑，
    协调各节点协同工作
    
    Returns:
        配置好的StateGraph实例
    """
    # 创建状态图
    workflow = StateGraph(MultimodalState)
    
    # 添加节点
    workflow.add_node("process_image", process_image)
    workflow.add_node("process_audio", process_audio)
    workflow.add_node("multimodal_fusion", multimodal_fusion)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("synthesize_audio_response", synthesize_audio_response)
    
    # 设置入口点 - 根据输入类型选择起始节点
    # 这里我们假设总是有文本查询或音频输入作为基础
    workflow.set_entry_point("process_audio")
    
    # 定义边 - 处理流程
    # 首先处理音频（如果有）
    workflow.add_edge("process_audio", "process_image")
    
    # 然后处理图像（如果有）
    workflow.add_edge("process_image", "multimodal_fusion")
    
    # 融合多模态信息
    workflow.add_edge("multimodal_fusion", "generate_response")
    
    # 生成响应
    workflow.add_edge("generate_response", "synthesize_audio_response")
    
    # 最后合成语音（如果需要）
    # 这是最后一个节点，所以不需要额外的边，会自动结束
    
    return workflow

# =================================================================
# 第4部分: 示例运行与结果展示
# =================================================================

def run_multimodal_example(
    query: str = None, 
    image_path: str = None, 
    audio_path: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """运行多模态处理示例
    
    WHY - 设计思路:
    1. 需要一个简单的方式来演示多模态处理
    2. 需要支持不同组合的输入类型
    3. 需要展示完整的处理流程和结果
    
    HOW - 实现方式:
    1. 构建多模态处理图
    2. 准备输入状态
    3. 执行图并收集结果
    
    WHAT - 功能作用:
    提供一个便捷的接口运行多模态处理系统，并展示处理结果
    
    Args:
        query: 文本查询
        image_path: 图像路径
        audio_path: 音频路径
        verbose: 是否打印详细结果
        
    Returns:
        处理结果
    """
    # 构建工作流图
    multimodal_graph = build_multimodal_graph()
    
    # 编译图
    app = multimodal_graph.compile()
    
    # 准备初始状态
    initial_state = {}
    if query:
        initial_state["query"] = query
    if image_path:
        initial_state["image_path"] = image_path
    if audio_path:
        initial_state["audio_path"] = audio_path
    
    # 如果没有任何输入，使用默认值
    if not initial_state:
        initial_state = {
            "query": "请描述这张图片中的内容",
            "image_path": "sample_image.jpg"
        }
    
    # 执行图
    result = app.invoke(initial_state)
    
    # 打印结果
    if verbose:
        print("\n===== 多模态处理结果 =====")
        
        print("\n----- 输入信息 -----")
        if query:
            print(f"文本查询: {query}")
        if image_path:
            print(f"图像路径: {image_path}")
        if audio_path:
            print(f"音频路径: {audio_path}")
        
        print("\n----- 处理结果 -----")
        if result.get("image_analysis"):
            print("\n图像分析结果:")
            img_analysis = result["image_analysis"]
            print(f"- 类别: {img_analysis.get('category', '未知')}")
            print(f"- 描述: {img_analysis.get('description', '无描述')}")
            print("- 检测到的物体:")
            for obj in img_analysis.get("objects", []):
                print(f"  • {obj['name']} (置信度: {obj['confidence']})")
        
        if result.get("audio_transcript"):
            print("\n音频转写结果:")
            audio_result = result["audio_transcript"]
            print(f"- 转写内容: {audio_result.get('transcript', '无内容')}")
            print(f"- 置信度: {audio_result.get('confidence', 0)}")
            features = audio_result.get("audio_features", {})
            print(f"- 音频时长: {features.get('duration', 0)}秒")
            print(f"- 音频语言: {features.get('language', '未知')}")
        
        print("\n----- 最终响应 -----")
        if result.get("response"):
            print(result["response"])
        else:
            print("未生成响应")
        
        if result.get("audio_response"):
            print("\n已生成语音响应:")
            print(f"- 输出文件: {result['audio_response'].get('output_path', '未知')}")
            print(f"- 音频时长: {result['audio_response'].get('duration', 0)}秒")
        
        if result.get("error"):
            print(f"\n处理错误: {result['error']}")
    
    return result

def demonstrate_image_only():
    """演示仅图像输入的处理"""
    print("\n===== 示例1: 仅图像输入 =====")
    run_multimodal_example(
        query="这张图片里有什么内容?",
        image_path="sample_nature.jpg"
    )

def demonstrate_audio_only():
    """演示仅音频输入的处理"""
    print("\n===== 示例2: 仅音频输入 =====")
    run_multimodal_example(
        audio_path="sample_query.wav"
    )

def demonstrate_multimodal():
    """演示完整多模态输入的处理"""
    print("\n===== 示例3: 完整多模态输入 =====")
    run_multimodal_example(
        query="图中的主要物体是什么?",
        image_path="sample_city.jpg",
        audio_path="sample_instruction.wav"
    )

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行各个示例
    2. 需要展示不同输入组合的处理流程
    3. 需要提供学习总结
    
    HOW - 实现方式:
    1. 运行不同输入组合的示例
    2. 展示各示例的处理结果
    3. 提供学习要点总结
    
    WHAT - 功能作用:
    作为程序入口点，演示多模态处理系统的完整功能和不同使用场景
    """
    print("===== LangGraph 多模态处理学习示例 =====\n")
    
    try:
        # 示例1: 仅图像输入
        demonstrate_image_only()
        
        # 示例2: 仅音频输入
        demonstrate_audio_only()
        
        # 示例3: 完整多模态输入
        demonstrate_multimodal()
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 使用LangGraph构建多模态处理系统")
        print("2. 实现图像、语音和文本的协同处理")
        print("3. 设计多模态信息融合的流程")
        print("4. 构建灵活的处理路径以适应不同输入组合")
        print("5. 生成基于多模态信息的综合响应")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 