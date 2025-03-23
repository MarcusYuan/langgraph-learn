#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 内容创作系统
===================
本示例实现一个基于LangGraph的内容创作系统，包含:
1. 创意生成流程 - 大纲生成、内容草稿和完善
2. 内容改进与评价 - 质量评估和多轮修改
3. 风格一致性维护 - 确保内容风格符合要求

WHY - 设计思路:
1. 内容创作需要结构化的流程支持
2. 创作过程需要多次评估和修改
3. 高质量内容需要保持风格一致性
4. 创作系统应能适应不同主题和受众

HOW - 实现方式:
1. 使用LLM生成内容大纲和草稿
2. 设计评估节点对内容质量进行打分
3. 实现修改循环直到内容质量达标
4. 使用条件边控制流程走向

WHAT - 功能作用:
通过本示例，你将学习如何构建一个完整的内容创作系统，
了解大纲生成、草稿撰写、内容评估、修改完善的实现方法，
以及如何使用LangGraph构建有条件循环的创作流程。

学习目标:
- 掌握LangGraph中的条件循环设计
- 了解内容质量评估的实现方法
- 学习如何在创作流程中保持风格一致性
- 理解多步骤创意生成的工作流程
"""

import os
import time
import json
import random
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

# LangGraph相关导入
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 使用Ollama作为本地LLM
try:
    from langchain_ollama import ChatOllama
    print("使用Ollama作为LLM提供者")
except ImportError:
    from langchain_openai import ChatOpenAI
    print("使用OpenAI作为LLM提供者")

# =================================================================
# 第1部分: 基础组件 - 状态定义与LLM初始化
# =================================================================

class ContentCreationState(TypedDict):
    """内容创作系统状态定义
    
    WHY - 设计思路:
    1. 需要存储创作流程的各个阶段产物
    2. 需要记录评估结果和修改历史
    3. 需要保存风格和受众等元数据
    
    HOW - 实现方式:
    1. 使用TypedDict定义类型安全的状态结构
    2. 包含主题、大纲、草稿等关键内容字段
    3. 添加质量评分和修改次数计数
    
    WHAT - 功能作用:
    为整个内容创作系统提供统一的状态管理接口，
    存储创作过程中的各类信息和中间产物
    """
    topic: str  # 创作主题
    audience: str  # 目标受众
    content_type: str  # 内容类型: 博客/文章/社交媒体等
    style: str  # 写作风格
    outline: Optional[str]  # 内容大纲
    draft: Optional[str]  # 内容草稿
    current_content: Optional[str]  # 当前内容版本
    evaluation: Optional[Dict[str, Any]]  # 评估结果
    quality_score: float  # 内容质量评分(0-1)
    revision_count: int  # 修改次数
    revision_history: List[Dict[str, Any]]  # 修改历史
    final_content: Optional[str]  # 最终内容
    metadata: Dict[str, Any]  # 元数据

def initialize_state(topic: str, audience: str, 
                    content_type: str = "博客文章", 
                    style: str = "专业") -> ContentCreationState:
    """初始化内容创作状态
    
    WHY - 设计思路:
    1. 需要为创作过程提供一致的初始状态
    2. 需要设置默认的内容类型和风格
    3. 需要初始化各个阶段的内容为空
    
    HOW - 实现方式:
    1. 接收必要的主题和受众参数
    2. 设置默认的内容类型和风格
    3. 初始化所有必要字段为默认值
    
    WHAT - 功能作用:
    为新的内容创作提供初始状态，确保所有创作过程
    从相同的起点开始，便于状态管理和追踪
    
    Args:
        topic: 创作主题
        audience: 目标受众
        content_type: 内容类型，默认为"博客文章"
        style: 写作风格，默认为"专业"
        
    Returns:
        ContentCreationState: 初始化的状态
    """
    session_id = f"content-{int(time.time())}"
    
    return {
        "topic": topic,
        "audience": audience,
        "content_type": content_type,
        "style": style,
        "outline": None,
        "draft": None,
        "current_content": None,
        "evaluation": None,
        "quality_score": 0.0,
        "revision_count": 0,
        "revision_history": [],
        "final_content": None,
        "metadata": {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
    }

def get_llm(temperature=0.7):
    """获取语言模型实例
    
    WHY - 设计思路:
    1. 需要集中管理LLM的实例创建
    2. 需要提供灵活的温度参数调整
    
    HOW - 实现方式:
    1. 尝试加载本地Ollama模型
    2. 如果失败则使用OpenAI API
    
    WHAT - 功能作用:
    创建并返回语言模型实例，用于各种处理节点
    
    Args:
        temperature: 生成温度，控制创造性，默认0.7
        
    Returns:
        BaseChatModel: 语言模型实例
    """
    # 首先尝试使用Ollama本地模型
    try:
        return ChatOllama(model="llama3", temperature=temperature)
    except:
        # 回退到OpenAI
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(temperature=temperature)
        except:
            # 最后回退到其他模型
            raise Exception("无法加载任何LLM模型，请确保安装了langchain_ollama或langchain_openai") 

# =================================================================
# 第2部分: 创意生成流程 - 大纲和草稿生成
# =================================================================

def generate_outline_node(state: ContentCreationState) -> Dict[str, Any]:
    """生成内容大纲节点
    
    WHY - 设计思路:
    1. 需要生成结构化的内容大纲
    2. 大纲应符合目标受众和内容类型
    3. 需要考虑主题和风格要求
    
    HOW - 实现方式:
    1. 提取主题、受众和内容类型信息
    2. 使用LLM生成适当的内容大纲
    3. 更新状态中的大纲字段
    
    WHAT - 功能作用:
    分析主题和受众需求，生成结构化的内容大纲，
    为后续内容创作提供框架
    
    Args:
        state: 当前内容创作状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取主题和受众信息
    topic = state["topic"]
    audience = state["audience"]
    content_type = state["content_type"]
    style = state["style"]
    
    # 使用LLM生成大纲
    llm = get_llm(temperature=0.7)  # 使用稍高的温度以增加创造性
    
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的内容策划和大纲设计专家。你需要为指定主题创建一个详细的内容大纲。
        大纲应该根据目标受众和内容类型进行定制，并符合指定的写作风格。
        提供一个结构清晰的大纲，包括引言、主要部分(至少3-5个主要章节)和结论。
        每个主要部分应包含2-4个子点，简要说明该部分将涵盖的内容。
        大纲应具有逻辑性和连贯性，让读者能够自然地从一个观点过渡到下一个观点。
        """),
        ("human", """请为以下内容创建一个详细的大纲:
        
        主题: {topic}
        目标受众: {audience}
        内容类型: {content_type}
        写作风格: {style}
        
        请提供完整的大纲结构，包括标题、引言、主要部分、子部分和结论。
        """),
    ])
    
    # 创建大纲生成链
    outline_chain = outline_prompt | llm 
    
    # 执行大纲生成
    result = outline_chain.invoke({
        "topic": topic,
        "audience": audience,
        "content_type": content_type,
        "style": style
    })
    
    outline_content = result.content
    
    # 更新状态
    new_state["outline"] = outline_content
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "outline_generated_at": datetime.now().isoformat()
    }
    
    print(f"已生成大纲，长度: {len(outline_content)} 字符")
    
    return new_state

def draft_content_node(state: ContentCreationState) -> Dict[str, Any]:
    """根据大纲创建内容草稿节点
    
    WHY - 设计思路:
    1. 需要基于大纲生成完整的内容草稿
    2. 草稿应符合目标受众和内容类型
    3. 内容应保持指定的写作风格
    
    HOW - 实现方式:
    1. 提取大纲和风格信息
    2. 使用LLM根据大纲生成完整草稿
    3. 更新状态中的草稿和当前内容字段
    
    WHAT - 功能作用:
    基于生成的大纲创建完整的内容草稿，
    为后续评估和修改提供基础
    
    Args:
        state: 当前内容创作状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取大纲和风格信息
    outline = state["outline"]
    topic = state["topic"]
    audience = state["audience"]
    content_type = state["content_type"]
    style = state["style"]
    
    if not outline:
        # 如果没有大纲，返回错误
        return {
            "error": "无法创建草稿: 未找到大纲",
            "draft": None,
            "current_content": None
        }
    
    # 使用LLM生成草稿
    llm = get_llm(temperature=0.7)
    
    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的内容创作者，擅长根据大纲创作高质量内容。
        你需要根据提供的大纲创建一个完整的内容草稿。
        草稿应该遵循大纲的结构，同时注意以下几点:
        1. 使用指定的写作风格
        2. 针对指定的目标受众调整语言和深度
        3. 符合指定的内容类型要求
        4. 内容应当丰富、连贯且有价值
        5. 保持清晰的段落结构和逻辑流程
        6. 提供具体的例子或数据支持你的观点
        
        创建的草稿应该像一个完整的成品，而不仅仅是填充大纲。"""),
        ("human", """请根据以下信息创建内容草稿:
        
        主题: {topic}
        目标受众: {audience}
        内容类型: {content_type}
        写作风格: {style}
        
        大纲:
        {outline}
        
        请创建一个完整、连贯、有深度的草稿，遵循大纲结构但不局限于大纲中的点。
        """),
    ])
    
    # 创建草稿生成链
    draft_chain = draft_prompt | llm
    
    # 执行草稿生成
    result = draft_chain.invoke({
        "topic": topic,
        "audience": audience,
        "content_type": content_type,
        "style": style,
        "outline": outline
    })
    
    draft_content = result.content
    
    # 更新状态
    new_state["draft"] = draft_content
    new_state["current_content"] = draft_content  # 当前内容初始化为草稿
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "draft_generated_at": datetime.now().isoformat()
    }
    
    print(f"已生成草稿，长度: {len(draft_content)} 字符")
    
    return new_state

# =================================================================
# 第3部分: 内容评估与修改 - 质量评价和内容改进
# =================================================================

def evaluate_content_node(state: ContentCreationState) -> Dict[str, Any]:
    """评估内容质量节点
    
    WHY - 设计思路:
    1. 需要客观评估内容质量
    2. 需要从多个维度进行评分
    3. 需要提供具体的改进建议
    
    HOW - 实现方式:
    1. 提取当前内容和目标信息
    2. 使用LLM从多个维度评估内容质量
    3. 生成评分和详细评价结果
    
    WHAT - 功能作用:
    评估当前内容的质量，生成量化评分和
    详细的改进建议，为修改提供依据
    
    Args:
        state: 当前内容创作状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取当前内容和目标信息
    current_content = state["current_content"]
    topic = state["topic"]
    audience = state["audience"]
    content_type = state["content_type"]
    style = state["style"]
    
    if not current_content:
        # 如果没有当前内容，返回错误
        return {
            "error": "无法评估: 未找到当前内容",
            "evaluation": None,
            "quality_score": 0.0
        }
    
    # 定义评估维度
    evaluation_dimensions = [
        "相关性 - 内容与主题的相关程度",
        "深度 - 内容的深度和洞察力",
        "结构 - 内容的组织和流程",
        "目标受众 - 内容对目标受众的适合度",
        "风格一致性 - 内容与指定风格的一致性",
        "可读性 - 内容的清晰度和流畅性",
        "原创性 - 内容的独特性和创新性",
        "价值 - 内容提供的实用价值"
    ]
    
    # 使用LLM进行评估
    llm = get_llm(temperature=0.2)  # 使用低温度以获得更客观的评估
    
    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的内容评估专家，擅长分析和评价内容质量。
        你需要从多个维度对提供的内容进行评估，并提供量化评分和详细反馈。
        你的评估应该客观、公正、全面，注意考虑内容的目标受众和预期风格。
        
        对每个维度，提供1-10的评分(10分为最高)和简短解释。
        最后，给出总体评分(0-1之间的小数)和总体评价，并列出3-5点具体的改进建议。
        
        你的评估将以JSON格式返回，包含各维度评分、总评分和改进建议。"""),
        ("human", """请评估以下内容:
        
        主题: {topic}
        目标受众: {audience}
        内容类型: {content_type}
        写作风格: {style}
        
        评估维度:
        {dimensions}
        
        内容:
        {content}
        
        请提供详细的评估结果，包括各维度评分、总评分(0-1)和具体改进建议。
        以JSON格式返回，包含以下字段: 
        dimensions_scores（各维度评分）, 
        overall_score（总评分，0-1之间的小数）, 
        overall_feedback（总体评价）, 
        improvement_suggestions（改进建议列表）
        """),
    ])
    
    # 创建评估链
    evaluation_chain = evaluation_prompt | llm
    
    # 执行评估
    result = evaluation_chain.invoke({
        "topic": topic,
        "audience": audience,
        "content_type": content_type,
        "style": style,
        "dimensions": "\n".join(evaluation_dimensions),
        "content": current_content
    })
    
    # 尝试解析JSON结果
    try:
        # 尝试从回复中提取JSON
        import re
        json_str = re.search(r'```json\n(.*?)\n```', result.content, re.DOTALL)
        if json_str:
            evaluation_result = json.loads(json_str.group(1))
        else:
            json_str = re.search(r'{.*}', result.content, re.DOTALL)
            if json_str:
                evaluation_result = json.loads(json_str.group(0))
            else:
                # 创建一个简单的评估结果
                evaluation_result = {
                    "dimensions_scores": {"整体评价": 7},
                    "overall_score": 0.7,
                    "overall_feedback": "内容整体表现良好，但存在改进空间。",
                    "improvement_suggestions": ["考虑增加更多具体例子", "可以进一步优化结构"]
                }
    except Exception as e:
        print(f"评估结果解析失败: {str(e)}")
        # 创建一个默认的评估结果
        evaluation_result = {
            "dimensions_scores": {"整体评价": 6},
            "overall_score": 0.6,
            "overall_feedback": "内容质量中等，需要进一步修改完善。",
            "improvement_suggestions": ["考虑增加更多深度", "提高与主题的相关性", "优化内容结构"]
        }
    
    # 确保overall_score在0-1范围内
    overall_score = evaluation_result.get("overall_score", 0.6)
    if isinstance(overall_score, str):
        try:
            overall_score = float(overall_score)
        except:
            overall_score = 0.6
    overall_score = max(0.0, min(1.0, overall_score))
    
    # 更新状态
    new_state["evaluation"] = evaluation_result
    new_state["quality_score"] = overall_score
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "evaluation_at": datetime.now().isoformat()
    }
    
    print(f"内容评估完成，质量评分: {overall_score:.2f}")
    
    return new_state

def revise_content_node(state: ContentCreationState) -> Dict[str, Any]:
    """修改和改进内容节点
    
    WHY - 设计思路:
    1. 需要根据评估结果改进内容
    2. 需要针对具体建议进行修改
    3. 需要保持内容的核心主题和风格
    
    HOW - 实现方式:
    1. 提取当前内容和评估结果
    2. 使用LLM根据评估建议修改内容
    3. 更新状态中的内容版本和修改历史
    
    WHAT - 功能作用:
    根据评估结果和改进建议修改内容，
    提高内容质量，为最终版本做准备
    
    Args:
        state: 当前内容创作状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取当前内容和评估结果
    current_content = state["current_content"]
    evaluation = state["evaluation"]
    topic = state["topic"]
    audience = state["audience"]
    style = state["style"]
    revision_count = state["revision_count"]
    revision_history = state.get("revision_history", [])
    
    if not current_content or not evaluation:
        # 如果没有当前内容或评估结果，返回错误
        return {
            "error": "无法修改: 未找到当前内容或评估结果",
            "current_content": current_content
        }
    
    # 提取改进建议
    improvement_suggestions = evaluation.get("improvement_suggestions", [])
    if isinstance(improvement_suggestions, str):
        improvement_suggestions = [improvement_suggestions]
    
    overall_feedback = evaluation.get("overall_feedback", "内容需要改进")
    
    # 使用LLM进行内容修改
    llm = get_llm(temperature=0.7)  # 使用适中的温度平衡创造性和一致性
    
    revision_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的内容编辑和修改专家，擅长根据反馈改进内容质量。
        你需要根据提供的评估反馈和改进建议，修改和完善给定的内容。
        
        在修改过程中，请注意以下几点:
        1. 保持内容的核心主题和信息
        2. 遵循指定的写作风格
        3. 针对目标受众适当调整内容
        4. 重点解决评估中指出的问题
        5. 保持内容的连贯性和流畅性
        6. 不要删减有价值的信息，而是优化表达和组织
        
        提供完整的修改版本，而不仅仅是修改建议。"""),
        ("human", """请根据以下信息修改内容:
        
        主题: {topic}
        目标受众: {audience}
        写作风格: {style}
        修改次数: {revision_count}
        
        评估反馈: {feedback}
        
        改进建议:
        {suggestions}
        
        原始内容:
        {content}
        
        请提供完整的修改版本，着重解决评估反馈中指出的问题，同时保持内容的核心价值和风格一致性。
        """),
    ])
    
    # 创建修改链
    revision_chain = revision_prompt | llm
    
    # 执行内容修改
    result = revision_chain.invoke({
        "topic": topic,
        "audience": audience,
        "style": style,
        "revision_count": revision_count + 1,
        "feedback": overall_feedback,
        "suggestions": "\n".join([f"- {s}" for s in improvement_suggestions]),
        "content": current_content
    })
    
    revised_content = result.content
    
    # 保存当前版本到修改历史
    current_version = {
        "version": revision_count,
        "content": current_content,
        "evaluation": evaluation,
        "timestamp": datetime.now().isoformat()
    }
    revision_history.append(current_version)
    
    # 更新状态
    new_state["current_content"] = revised_content
    new_state["revision_count"] = revision_count + 1
    new_state["revision_history"] = revision_history
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "revision_at": datetime.now().isoformat()
    }
    
    print(f"内容修改完成，修改次数: {revision_count + 1}")
    
    return new_state 

# =================================================================
# 第4部分: 内容完成和格式化 - 最终内容生成
# =================================================================

def finalize_content_node(state: ContentCreationState) -> Dict[str, Any]:
    """最终内容完成节点
    
    WHY - 设计思路:
    1. 需要对通过评估的内容进行最终完善
    2. 需要确保内容格式规范统一
    3. 需要生成适合发布的最终版本
    
    HOW - 实现方式:
    1. 提取当前内容和元数据
    2. 使用LLM进行最终优化和格式化
    3. 更新状态中的最终内容字段
    
    WHAT - 功能作用:
    对高质量内容进行最终完善和格式化，
    生成准备发布的最终版本
    
    Args:
        state: 当前内容创作状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取当前内容和元数据
    current_content = state["current_content"]
    topic = state["topic"]
    audience = state["audience"]
    content_type = state["content_type"]
    style = state["style"]
    
    if not current_content:
        # 如果没有当前内容，返回错误
        return {
            "error": "无法完成: 未找到当前内容",
            "final_content": None
        }
    
    # 使用LLM进行最终完善
    llm = get_llm(temperature=0.4)  # 使用较低温度确保格式一致性
    
    finalize_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的内容编辑和排版专家，擅长对内容进行最终完善和格式化。
        你需要对已经达到质量标准的内容进行最后的润色、完善和格式规范化，使其达到发布标准。
        
        在最终完善过程中，请注意以下几点:
        1. 检查并修正任何语法、拼写或标点错误
        2. 确保段落结构清晰，标题层级一致
        3. 优化开头和结尾，增强整体连贯性
        4. 添加适当的过渡词汇，提高阅读流畅度
        5. 保持风格一致性和专业性
        6. 根据内容类型进行适当的格式调整
        
        提供完全准备好发布的最终版本。"""),
        ("human", """请对以下内容进行最终完善和格式化:
        
        主题: {topic}
        目标受众: {audience}
        内容类型: {content_type}
        写作风格: {style}
        
        当前内容:
        {content}
        
        请提供完全准备好发布的最终版本，确保格式规范、语言流畅、风格一致。
        """),
    ])
    
    # 创建最终完善链
    finalize_chain = finalize_prompt | llm
    
    # 执行最终完善
    result = finalize_chain.invoke({
        "topic": topic,
        "audience": audience,
        "content_type": content_type,
        "style": style,
        "content": current_content
    })
    
    final_content = result.content
    
    # 更新状态
    new_state["final_content"] = final_content
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "finalized_at": datetime.now().isoformat(),
        "completion_status": "completed"
    }
    
    print(f"内容最终完善完成，最终内容长度: {len(final_content)} 字符")
    
    return new_state

def content_route(state: ContentCreationState) -> str:
    """内容流程路由函数
    
    WHY - 设计思路:
    1. 需要根据内容质量决定下一步操作
    2. 需要限制修改次数避免无限循环
    
    HOW - 实现方式:
    1. 检查内容质量评分
    2. 考虑已进行的修改次数
    3. 决定是进行修改还是完成内容
    
    WHAT - 功能作用:
    作为LangGraph的条件路由函数，决定内容创作
    流程的下一步，确保内容达到质量标准
    
    Args:
        state: 当前内容创作状态
        
    Returns:
        str: 下一个节点的名称
    """
    # 检查质量评分
    quality_score = state.get("quality_score", 0.0)
    revision_count = state.get("revision_count", 0)
    
    # 如果质量评分足够高，或已修改多次，则完成内容
    if quality_score >= 0.8 or revision_count >= 3:
        return "finalize"
    else:
        # 否则进行修改
        return "revise"

# =================================================================
# 第5部分: 图结构构建 - 内容创作流程
# =================================================================

def build_content_creation_graph():
    """构建内容创作系统图
    
    WHY - 设计思路:
    1. 需要将创作流程组织为完整的工作流
    2. 需要实现基于质量评估的条件路由
    3. 需要支持内容修改循环
    
    HOW - 实现方式:
    1. 创建基于ContentCreationState的StateGraph
    2. 添加大纲生成、草稿创建等节点
    3. 实现基于质量评分的条件路由
    4. 设置修改和完成逻辑
    
    WHAT - 功能作用:
    构建一个完整的内容创作系统图，整合大纲生成、
    草稿创建、内容评估、修改完善等功能
    
    Returns:
        StateGraph: 编译后的内容创作系统图
    """
    # 创建图
    workflow = StateGraph(ContentCreationState)
    
    # 添加节点
    workflow.add_node("generate_outline", generate_outline_node)
    workflow.add_node("draft_content", draft_content_node)
    workflow.add_node("evaluate_content", evaluate_content_node)
    workflow.add_node("revise_content", revise_content_node)
    workflow.add_node("finalize_content", finalize_content_node)
    
    # 设置流程
    # 1. 大纲生成 -> 草稿创建
    workflow.add_edge("generate_outline", "draft_content")
    
    # 2. 草稿创建 -> 内容评估
    workflow.add_edge("draft_content", "evaluate_content")
    
    # 3. 内容评估 -> 条件路由(修改或完成)
    workflow.add_conditional_edges(
        "evaluate_content",
        content_route,
        {
            "revise": "revise_content",
            "finalize": "finalize_content"
        }
    )
    
    # 4. 内容修改 -> 内容评估 (创建修改-评估循环)
    workflow.add_edge("revise_content", "evaluate_content")
    
    # 5. 内容完成 -> 结束
    workflow.add_edge("finalize_content", END)
    
    # 设置入口点
    workflow.set_entry_point("generate_outline")
    
    # 编译图
    return workflow.compile(checkpointer=MemorySaver())

# =================================================================
# 第6部分: 示例演示 - 内容创作系统演示界面
# =================================================================

def show_content_creation_examples():
    """展示内容创作系统示例
    
    WHY - 设计思路:
    1. 需要提供交互式的示例演示
    2. 允许用户选择不同的创作主题
    3. 展示创作过程的各个阶段
    
    HOW - 实现方式:
    1. 提供交互式菜单选择不同主题
    2. 调用内容创作系统生成内容
    3. 展示创作过程中的关键数据
    
    WHAT - 功能作用:
    提供一个互动式展示，帮助用户理解内容创作系统的工作流程
    """
    print("\n===== LangGraph 内容创作系统示例 =====")
    
    print("\n本示例展示了基于LangGraph的内容创作系统，演示从主题到最终内容的创作流程。")
    
    # 预定义主题
    sample_topics = [
        {
            "topic": "人工智能在日常生活中的应用",
            "audience": "普通大众",
            "content_type": "科普文章",
            "style": "简明易懂"
        },
        {
            "topic": "远程工作的高效管理策略",
            "audience": "企业管理者",
            "content_type": "专业指南",
            "style": "专业实用"
        },
        {
            "topic": "健康饮食与生活方式",
            "audience": "关注健康的年轻人",
            "content_type": "生活指南",
            "style": "轻松友好"
        },
        {
            "topic": "个人理财基础知识",
            "audience": "理财新手",
            "content_type": "教育内容",
            "style": "通俗易懂"
        }
    ]
    
    while True:
        print("\n请选择创作主题或自定义:")
        for i, topic in enumerate(sample_topics):
            print(f"{i+1}. {topic['topic']} (目标受众: {topic['audience']})")
        print("5. 自定义主题")
        print("0. 返回主菜单")
        
        choice = input("\n您的选择> ")
        
        if choice == "0":
            break
        elif choice in ["1", "2", "3", "4"]:
            selected_topic = sample_topics[int(choice)-1]
            run_content_creation(selected_topic)
        elif choice == "5":
            # 自定义主题
            custom_topic = {}
            custom_topic["topic"] = input("请输入主题: ")
            custom_topic["audience"] = input("请输入目标受众: ")
            custom_topic["content_type"] = input("请输入内容类型(默认为博客文章): ") or "博客文章"
            custom_topic["style"] = input("请输入写作风格(默认为专业): ") or "专业"
            run_content_creation(custom_topic)
        else:
            print("无效选择，请重试")

def run_content_creation(topic_config):
    """执行内容创作流程
    
    WHY - 设计思路:
    1. 需要执行完整的内容创作流程
    2. 需要展示每个阶段的结果
    3. 需要处理可能的错误情况
    
    HOW - 实现方式:
    1. 构建内容创作图
    2. 初始化内容创作状态
    3. 调用LangGraph执行创作流程
    4. 展示创作结果和评估信息
    
    WHAT - 功能作用:
    执行内容创作流程并展示结果，是示例的核心执行函数
    
    Args:
        topic_config: 主题配置，包含topic、audience、content_type和style
    """
    print("\n" + "=" * 50)
    print(f"开始创作主题: '{topic_config['topic']}'")
    print("=" * 50)
    
    # 构建内容创作图
    print("构建内容创作系统图...")
    graph = build_content_creation_graph()
    
    # 创建初始状态
    print("\n初始化内容创作状态...")
    state = initialize_state(
        topic=topic_config["topic"],
        audience=topic_config["audience"],
        content_type=topic_config["content_type"],
        style=topic_config["style"]
    )
    
    print(f"\n主题信息:")
    print(f"- 主题: {topic_config['topic']}")
    print(f"- 目标受众: {topic_config['audience']}")
    print(f"- 内容类型: {topic_config['content_type']}")
    print(f"- 写作风格: {topic_config['style']}")
    
    # 询问是否开始创作
    start = input("\n准备开始内容创作，按回车继续或输入'q'取消: ")
    if start.lower() == 'q':
        print("已取消创作")
        return
    
    # 运行内容创作流程
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    try:
        print("\n开始创作流程，这可能需要几分钟时间...")
        
        # 执行内容创作流程
        result = graph.invoke(state, config=config)
        
        # 打印创作流程结果
        print("\n" + "=" * 50)
        print("内容创作完成!")
        print("=" * 50)
        
        # 打印大纲
        print("\n--- 内容大纲 ---")
        if result.get("outline"):
            outline = result.get("outline")
            print(outline[:300] + "..." if len(outline) > 300 else outline)
        else:
            print("未生成大纲")
        
        # 打印最终内容摘要
        print("\n--- 最终内容摘要 ---")
        final_content = result.get("final_content", "未生成最终内容")
        print(final_content[:300] + "..." if len(final_content) > 300 else final_content)
        
        # 打印评估结果
        print("\n--- 质量评估 ---")
        print(f"质量评分: {result.get('quality_score', 0.0):.2f}")
        print(f"修改次数: {result.get('revision_count', 0)}")
        
        # 询问是否查看详细信息
        view_details = input("\n是否查看详细信息? (y/n): ")
        if view_details.lower() == 'y':
            print("\n--- 详细评估信息 ---")
            evaluation = result.get("evaluation", {})
            if evaluation:
                print(f"总体评价: {evaluation.get('overall_feedback', '未提供')}")
                print("\n改进建议:")
                suggestions = evaluation.get("improvement_suggestions", [])
                if isinstance(suggestions, list):
                    for i, suggestion in enumerate(suggestions):
                        print(f"{i+1}. {suggestion}")
                else:
                    print(suggestions)
            else:
                print("未找到评估信息")
        
        # 询问是否保存结果
        save_result = input("\n是否保存创作结果到文件? (y/n): ")
        if save_result.lower() == 'y':
            # 保存结果到文件
            timestamp = int(time.time())
            output_file = f"content_creation_{timestamp}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# {result.get('topic', '未指定主题')}\n\n")
                f.write(result.get("final_content", "未生成最终内容"))
            
            print(f"\n最终内容已保存到文件: {output_file}")
        
    except Exception as e:
        print(f"\n创作过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    input("\n按回车键返回主菜单...")

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行内容创作示例
    2. 需要适当的错误处理确保示例稳定运行
    3. 需要提供清晰的开始和结束提示
    
    HOW - 实现方式:
    1. 使用try-except包装主要执行逻辑
    2. 提供开始和结束提示
    3. 调用示例展示函数
    4. 总结关键学习点
    
    WHAT - 功能作用:
    作为程序入口点，执行内容创作示例，确保示例执行的稳定性
    """
    print("===== LangGraph 内容创作系统学习示例 =====\n")
    
    try:
        # 运行内容创作示例
        show_content_creation_examples()
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 构建基于LangGraph的内容创作系统")
        print("2. 实现条件循环的创作流程")
        print("3. 设计内容质量评估机制")
        print("4. 创建大纲生成和草稿撰写节点")
        print("5. 实现内容修改和完善流程")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 