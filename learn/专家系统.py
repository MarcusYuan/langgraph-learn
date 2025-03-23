#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 专家系统实现
===================
本示例实现一个基于LangGraph的专家系统，包含:
1. 领域知识编码 - 专业知识表示与访问
2. 推理链与解释 - 推理过程透明化
3. 不确定性处理 - 处理模糊问题和多种可能答案

WHY - 设计思路:
1. 专业领域问题需要系统化推理方法
2. 用户需要了解推理过程和决策依据
3. 现实问题常常存在不确定性和模糊边界
4. 需要整合不同知识源和推理技术

HOW - 实现方式:
1. 构建领域知识表示和检索系统
2. 设计明确的步骤化推理流程
3. 引入不确定性评估和置信度评分
4. 使用思维链(CoT)和自反思进行推理

WHAT - 功能作用:
通过本示例，你将学习如何构建一个完整的专家系统，
了解领域知识表示、推理过程设计、不确定性处理的实现方法，
以及如何使用LangGraph构建可解释的专家推理系统。

学习目标:
- 掌握领域知识表示与检索技术
- 了解如何实现链式推理与思维记录
- 学习如何处理专业问题中的不确定性
- 理解如何构建透明可解释的专家系统
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

class ExpertSystemState(TypedDict):
    """专家系统状态定义
    
    WHY - 设计思路:
    1. 需要存储问题、上下文和推理过程
    2. 需要跟踪知识来源和不确定性
    3. 需要保存推理中间结果和最终解决方案
    
    HOW - 实现方式:
    1. 使用TypedDict定义类型安全的状态结构
    2. 包含问题描述、领域知识等关键字段
    3. 添加推理链和置信度评分
    
    WHAT - 功能作用:
    为整个专家系统提供统一的状态管理接口，
    存储专家推理过程中的问题、知识、推理链
    和不确定性评估等关键信息
    """
    problem: str  # 需要解决的问题
    domain: str  # 领域类别
    context: Optional[Dict[str, Any]]  # 额外的上下文信息
    required_info: Optional[List[str]]  # 所需信息清单
    relevant_knowledge: Optional[List[Dict[str, Any]]]  # 相关领域知识
    reasoning_chain: Optional[List[Dict[str, Any]]]  # 推理步骤链
    alternative_paths: Optional[List[Dict[str, Any]]]  # 替代推理路径
    confidence_scores: Optional[Dict[str, float]]  # 各方面的置信度评分
    solution: Optional[Dict[str, Any]]  # 最终解决方案
    explanation: Optional[str]  # 解决方案解释
    metadata: Dict[str, Any]  # 元数据

def initialize_state(problem: str, domain: str = "通用", 
                    context: Optional[Dict[str, Any]] = None) -> ExpertSystemState:
    """初始化专家系统状态
    
    WHY - 设计思路:
    1. 需要为专家系统提供一致的初始状态
    2. 需要设置默认的领域和处理标记
    3. 需要初始化各个阶段的内容为空
    
    HOW - 实现方式:
    1. 接收问题和领域参数
    2. 设置默认的上下文信息
    3. 初始化所有必要字段为默认值
    
    WHAT - 功能作用:
    为新的专家系统问题提供初始状态，确保所有
    推理过程从相同的起点开始，便于状态管理和追踪
    
    Args:
        problem: 需要解决的问题
        domain: 领域类别，默认为"通用"
        context: 可选的上下文信息
        
    Returns:
        ExpertSystemState: 初始化的状态
    """
    session_id = f"expert-{int(time.time())}"
    
    # 确保上下文信息是字典
    if context is None:
        context = {}
    
    return {
        "problem": problem,
        "domain": domain,
        "context": context,
        "required_info": None,
        "relevant_knowledge": None,
        "reasoning_chain": None,
        "alternative_paths": None,
        "confidence_scores": None,
        "solution": None,
        "explanation": None,
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

# 模拟领域知识库
KNOWLEDGE_BASE = {
    "医学": [
        {"id": "med-001", "topic": "高血压", "content": "高血压是指动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg。主要治疗方式包括生活方式改变和药物治疗。"},
        {"id": "med-002", "topic": "糖尿病", "content": "糖尿病是一种代谢紊乱疾病，特征是血糖水平长期升高。包括1型和2型，治疗方式各有不同。"},
        {"id": "med-003", "topic": "感冒", "content": "感冒是由多种病毒引起的上呼吸道感染，通常症状包括鼻塞、流涕、咳嗽和发热。一般为自限性疾病。"}
    ],
    "法律": [
        {"id": "law-001", "topic": "合同法", "content": "合同是民事主体之间设立、变更、终止民事法律关系的协议。合同的生效需要满足主体适格、意思表示真实等要件。"},
        {"id": "law-002", "topic": "知识产权", "content": "知识产权包括著作权、专利权、商标权等，保护创造性智力成果和商业标识。"},
        {"id": "law-003", "topic": "刑法", "content": "刑法规定犯罪行为及其法律后果，包括刑事责任和刑罚种类。适用罪刑法定、罪责刑相适应等原则。"}
    ],
    "计算机": [
        {"id": "cs-001", "topic": "算法复杂度", "content": "算法复杂度用大O表示法描述算法效率，包括时间复杂度和空间复杂度。常见复杂度有O(1)、O(n)、O(log n)等。"},
        {"id": "cs-002", "topic": "数据结构", "content": "数据结构是数据组织、管理和存储格式，包括数组、链表、树、图等，不同结构适用于不同场景。"},
        {"id": "cs-003", "topic": "网络协议", "content": "网络协议是通信规则的集合，如TCP/IP协议族。TCP提供可靠传输，UDP提供快速但不可靠的传输。"}
    ],
    "通用": [
        {"id": "gen-001", "topic": "问题解决", "content": "问题解决的一般步骤包括:1.明确问题 2.收集信息 3.分析原因 4.制定方案 5.实施方案 6.评估结果"},
        {"id": "gen-002", "topic": "决策方法", "content": "常见决策方法包括利弊分析法、决策矩阵法、德尔菲法等，适用于不同类型的决策问题。"},
        {"id": "gen-003", "topic": "创新思维", "content": "创新思维技术包括头脑风暴、六顶思考帽、SCAMPER等方法，有助于打破思维限制。"}
    ]
} 

# =================================================================
# 第2部分: 信息收集与知识检索 - 定义问题和获取知识
# =================================================================

def gather_information_node(state: ExpertSystemState) -> Dict[str, Any]:
    """收集问题相关信息节点
    
    WHY - 设计思路:
    1. 需要分析问题以确定所需信息
    2. 需要明确问题边界和期望结果
    3. 需要识别问题中的关键要素
    
    HOW - 实现方式:
    1. 提取问题和领域信息
    2. 使用LLM分析问题需求
    3. 生成所需信息的清单
    
    WHAT - 功能作用:
    分析问题，明确问题解决所需的关键信息，
    为后续知识检索和推理提供依据
    
    Args:
        state: 当前专家系统状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取问题和领域信息
    problem = state["problem"]
    domain = state["domain"]
    context = state.get("context", {})
    
    # 使用LLM分析问题
    llm = get_llm(temperature=0.3)  # 使用较低温度以获得精确分析
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的问题分析专家，擅长分析复杂问题并确定解决问题所需的关键信息。
        你需要仔细分析给定的问题，确定为解决该问题需要收集哪些信息。
        
        请考虑以下方面:
        1. 问题的核心是什么
        2. 问题属于哪个领域或子领域
        3. 解决此类问题通常需要哪些关键信息
        4. 问题中可能存在哪些隐含条件或假设
        5. 解决问题可能需要哪些专业知识
        
        生成一个结构化的信息需求列表，每个需求项包括:
        - 所需信息的描述
        - 为什么这个信息对解决问题很重要
        - 如何获取或评估这个信息"""),
        ("human", """请分析以下问题，确定解决所需的关键信息:
        
        问题: {problem}
        领域: {domain}
        上下文信息: {context}
        
        请提供一个结构化的信息需求分析，列出解决此问题所需的全部关键信息。
        以JSON格式返回，包含字段:
        - required_info_list: 所需信息列表
        - problem_classification: 问题分类
        - complexity_assessment: 复杂度评估(1-10)
        """),
    ])
    
    # 创建分析链
    analysis_chain = analysis_prompt | llm
    
    # 执行问题分析
    result = analysis_chain.invoke({
        "problem": problem,
        "domain": domain,
        "context": json.dumps(context, ensure_ascii=False)
    })
    
    # 尝试解析JSON结果
    try:
        # 尝试从回复中提取JSON
        import re
        json_str = re.search(r'```json\n(.*?)\n```', result.content, re.DOTALL)
        if json_str:
            analysis_result = json.loads(json_str.group(1))
        else:
            json_str = re.search(r'{.*}', result.content, re.DOTALL)
            if json_str:
                analysis_result = json.loads(json_str.group(0))
            else:
                # 创建一个简单的分析结果
                analysis_result = {
                    "required_info_list": ["需要明确问题的具体细节", "需要相关领域的基础知识"],
                    "problem_classification": domain,
                    "complexity_assessment": 5
                }
    except Exception as e:
        print(f"分析结果解析失败: {str(e)}")
        # 创建一个默认的分析结果
        analysis_result = {
            "required_info_list": ["需要明确问题的具体细节", "需要相关领域的基础知识"],
            "problem_classification": domain,
            "complexity_assessment": 5
        }
    
    # 提取所需信息列表
    required_info = analysis_result.get("required_info_list", [])
    if isinstance(required_info, str):
        required_info = [required_info]
    
    # 更新状态
    new_state["required_info"] = required_info
    new_state["context"] = {
        **(state.get("context", {})),
        "problem_classification": analysis_result.get("problem_classification", domain),
        "complexity_assessment": analysis_result.get("complexity_assessment", 5)
    }
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "information_gathered_at": datetime.now().isoformat()
    }
    
    print(f"信息需求分析完成，识别出 {len(required_info)} 项关键信息需求")
    
    return new_state

def query_knowledge_base_node(state: ExpertSystemState) -> Dict[str, Any]:
    """查询知识库获取领域知识节点
    
    WHY - 设计思路:
    1. 需要检索与问题相关的专业知识
    2. 需要考虑不同领域的特殊知识
    3. 需要对检索到的知识进行相关性排序
    
    HOW - 实现方式:
    1. 提取问题和领域信息
    2. 检索领域知识库中的相关内容
    3. 对检索结果进行相关性评估和排序
    
    WHAT - 功能作用:
    检索与问题高度相关的专业知识，为专家
    推理提供必要的知识支持
    
    Args:
        state: 当前专家系统状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取问题和领域信息
    problem = state["problem"]
    domain = state["domain"]
    required_info = state.get("required_info", [])
    
    # 使用模拟知识库检索
    knowledge_entries = KNOWLEDGE_BASE.get(domain, KNOWLEDGE_BASE["通用"])
    
    # 使用LLM评估知识相关性
    llm = get_llm(temperature=0.2)  # 低温度以获得客观评估
    
    # 如果没有知识条目，返回空结果
    if not knowledge_entries:
        return {
            "relevant_knowledge": [],
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "knowledge_retrieved_at": datetime.now().isoformat(),
                "knowledge_count": 0
            }
        }
    
    # 准备知识条目文本
    knowledge_texts = []
    for entry in knowledge_entries:
        knowledge_texts.append(f"ID: {entry['id']}\n主题: {entry['topic']}\n内容: {entry['content']}")
    
    knowledge_text = "\n\n".join(knowledge_texts)
    
    # 评估知识相关性
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的知识评估专家，擅长评估知识与问题的相关性。
        你需要评估提供的知识条目对解决特定问题的相关性和有用性。
        
        对每个知识条目，评估其与问题的相关性(0-10分)并说明原因。
        识别最相关的知识条目，并解释它们如何有助于解决问题。"""),
        ("human", """请评估以下知识条目与问题的相关性:
        
        问题: {problem}
        领域: {domain}
        所需信息: {required_info}
        
        知识条目:
        {knowledge}
        
        请对每个知识条目进行评估，返回JSON格式的结果，包含:
        - relevant_entries: 相关条目列表，每个条目包含id、relevance_score(0-10)和relevance_explanation
        - 按相关性降序排列
        """),
    ])
    
    # 创建相关性评估链
    relevance_chain = relevance_prompt | llm
    
    # 执行相关性评估
    result = relevance_chain.invoke({
        "problem": problem,
        "domain": domain,
        "required_info": json.dumps(required_info, ensure_ascii=False),
        "knowledge": knowledge_text
    })
    
    # 尝试解析结果
    try:
        # 尝试从回复中提取JSON
        import re
        json_str = re.search(r'```json\n(.*?)\n```', result.content, re.DOTALL)
        if json_str:
            relevance_result = json.loads(json_str.group(1))
        else:
            json_str = re.search(r'{.*}', result.content, re.DOTALL)
            if json_str:
                relevance_result = json.loads(json_str.group(0))
            else:
                # 创建一个简单的相关性结果
                relevance_result = {
                    "relevant_entries": [
                        {"id": entry["id"], "relevance_score": 5, "relevance_explanation": "可能相关"} 
                        for entry in knowledge_entries
                    ]
                }
    except Exception as e:
        print(f"相关性评估解析失败: {str(e)}")
        # 创建一个默认的相关性结果
        relevance_result = {
            "relevant_entries": [
                {"id": entry["id"], "relevance_score": 5, "relevance_explanation": "可能相关"} 
                for entry in knowledge_entries
            ]
        }
    
    # 提取相关条目
    relevant_entries = relevance_result.get("relevant_entries", [])
    
    # 合并相关知识
    relevant_knowledge = []
    for entry in relevant_entries:
        entry_id = entry.get("id")
        for k_entry in knowledge_entries:
            if k_entry["id"] == entry_id:
                relevant_knowledge.append({
                    "id": entry_id,
                    "topic": k_entry["topic"],
                    "content": k_entry["content"],
                    "relevance_score": entry.get("relevance_score", 5),
                    "relevance_explanation": entry.get("relevance_explanation", "")
                })
                break
    
    # 按相关性排序
    relevant_knowledge.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # 更新状态
    new_state["relevant_knowledge"] = relevant_knowledge
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "knowledge_retrieved_at": datetime.now().isoformat(),
        "knowledge_count": len(relevant_knowledge)
    }
    
    print(f"知识检索完成，找到 {len(relevant_knowledge)} 条相关知识")
    
    return new_state

# =================================================================
# 第3部分: 专家推理与解决方案 - 思维链与解释
# =================================================================

def expert_reasoning_node(state: ExpertSystemState) -> Dict[str, Any]:
    """专家推理过程节点
    
    WHY - 设计思路:
    1. 需要基于问题和知识进行专业推理
    2. 需要明确记录推理过程的每一步
    3. 需要考虑可能的替代推理路径
    
    HOW - 实现方式:
    1. 提取问题、所需信息和相关知识
    2. 使用思维链(CoT)方式引导LLM推理
    3. 记录详细的推理步骤和替代方案
    
    WHAT - 功能作用:
    执行专家级推理过程，分析问题并基于
    相关知识生成解决思路，同时保持推理透明度
    
    Args:
        state: 当前专家系统状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取问题、所需信息和相关知识
    problem = state["problem"]
    domain = state["domain"]
    required_info = state.get("required_info", [])
    relevant_knowledge = state.get("relevant_knowledge", [])
    context = state.get("context", {})
    
    # 准备知识文本
    knowledge_text = ""
    for i, entry in enumerate(relevant_knowledge, 1):
        knowledge_text += f"\n知识{i}: [{entry['topic']}] {entry['content']}"
    
    # 使用LLM进行专家推理
    llm = get_llm(temperature=0.4)  # 平衡创造性和逻辑性
    
    reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的领域专家，擅长通过逐步推理解决复杂问题。
        你需要基于提供的知识和信息，对问题进行深入分析和推理。
        
        请按以下步骤进行推理:
        1. 分析问题的核心要素和约束条件
        2. 确定问题的解决策略和方法
        3. 运用领域知识进行逐步推理
        4. 考虑可能的替代解决路径
        5. 评估不同推理路径的优劣
        6. 明确每一步推理的依据和逻辑
        
        你的推理应该清晰、逻辑严密，并具有很强的可解释性。
        同时考虑问题中可能存在的不确定性和边界情况。"""),
        ("human", """请对以下问题进行专家推理:
        
        问题: {problem}
        领域: {domain}
        所需信息: {required_info}
        上下文: {context}
        
        相关知识:
        {knowledge}
        
        请提供详细的推理过程，包括:
        1. 逐步的思考过程
        2. 每一步推理的依据
        3. 可能的替代思路
        4. 不确定性因素分析
        
        以JSON格式返回，包含:
        - reasoning_steps: 推理步骤列表，每步包含step_number、description、reasoning和evidence
        - alternative_paths: 替代推理路径列表
        - uncertainty_factors: 不确定性因素列表
        """),
    ])
    
    # 创建推理链
    reasoning_chain = reasoning_prompt | llm
    
    # 执行专家推理
    result = reasoning_chain.invoke({
        "problem": problem,
        "domain": domain,
        "required_info": json.dumps(required_info, ensure_ascii=False),
        "context": json.dumps(context, ensure_ascii=False),
        "knowledge": knowledge_text
    })
    
    # 尝试解析结果
    try:
        # 尝试从回复中提取JSON
        import re
        json_str = re.search(r'```json\n(.*?)\n```', result.content, re.DOTALL)
        if json_str:
            reasoning_result = json.loads(json_str.group(1))
        else:
            json_str = re.search(r'{.*}', result.content, re.DOTALL)
            if json_str:
                reasoning_result = json.loads(json_str.group(0))
            else:
                # 创建一个简单的推理结果
                reasoning_result = {
                    "reasoning_steps": [
                        {"step_number": 1, "description": "分析问题", "reasoning": "首先需要理解问题的核心要素", "evidence": "基于问题描述"}
                    ],
                    "alternative_paths": [{"description": "可能的替代思路", "reasoning": "从不同角度考虑问题"}],
                    "uncertainty_factors": ["信息不完整", "多种可能解释"]
                }
    except Exception as e:
        print(f"推理结果解析失败: {str(e)}")
        # 创建一个默认的推理结果
        reasoning_result = {
            "reasoning_steps": [
                {"step_number": 1, "description": "分析问题", "reasoning": "首先需要理解问题的核心要素", "evidence": "基于问题描述"}
            ],
            "alternative_paths": [{"description": "可能的替代思路", "reasoning": "从不同角度考虑问题"}],
            "uncertainty_factors": ["信息不完整", "多种可能解释"]
        }
    
    # 提取推理步骤和替代路径
    reasoning_steps = reasoning_result.get("reasoning_steps", [])
    alternative_paths = reasoning_result.get("alternative_paths", [])
    uncertainty_factors = reasoning_result.get("uncertainty_factors", [])
    
    # 评估推理的置信度
    confidence_scores = {
        "overall": 0.7,  # 默认整体置信度
        "completeness": 0.7,  # 信息完整性
        "consistency": 0.8,  # 推理一致性
        "evidence_strength": 0.7  # 证据强度
    }
    
    # 更新状态
    new_state["reasoning_chain"] = reasoning_steps
    new_state["alternative_paths"] = alternative_paths
    new_state["confidence_scores"] = confidence_scores
    new_state["context"] = {
        **(state.get("context", {})),
        "uncertainty_factors": uncertainty_factors
    }
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "reasoning_performed_at": datetime.now().isoformat(),
        "reasoning_steps_count": len(reasoning_steps)
    }
    
    print(f"专家推理完成，生成了 {len(reasoning_steps)} 个推理步骤")
    
    return new_state

def generate_solution_node(state: ExpertSystemState) -> Dict[str, Any]:
    """生成解决方案节点
    
    WHY - 设计思路:
    1. 需要基于推理过程生成解决方案
    2. 需要提供清晰详细的解释
    3. 需要考虑解决方案的可实施性
    
    HOW - 实现方式:
    1. 提取问题、推理链和置信度
    2. 使用LLM生成最终解决方案
    3. 添加详细解释和实施建议
    
    WHAT - 功能作用:
    基于专家推理过程，生成具体可行的解决方案，
    并提供清晰的解释和实施建议
    
    Args:
        state: 当前专家系统状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取问题、推理链和置信度
    problem = state["problem"]
    domain = state["domain"]
    reasoning_chain = state.get("reasoning_chain", [])
    confidence_scores = state.get("confidence_scores", {})
    relevant_knowledge = state.get("relevant_knowledge", [])
    
    # 如果没有推理链，无法生成解决方案
    if not reasoning_chain:
        return {
            "error": "无法生成解决方案: 缺少推理过程",
            "solution": None,
            "explanation": None
        }
    
    # 准备推理步骤文本
    reasoning_text = ""
    for step in reasoning_chain:
        reasoning_text += f"\n步骤{step.get('step_number', '?')}: {step.get('description', '')}\n"
        reasoning_text += f"推理: {step.get('reasoning', '')}\n"
        reasoning_text += f"依据: {step.get('evidence', '')}\n"
    
    # 使用LLM生成解决方案
    llm = get_llm(temperature=0.3)  # 使用较低温度以保持一致性
    
    solution_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的领域专家，擅长提供清晰、实用的解决方案。
        你需要基于前面的推理过程，生成一个完整的解决方案。
        
        你的解决方案应该:
        1. 直接回应问题的核心要求
        2. 基于推理过程中的逻辑和发现
        3. 考虑可行性和有效性
        4. 提供清晰的实施步骤(如适用)
        5. 解释解决方案的原理和依据
        6. 诚实面对不确定性和局限性
        
        解决方案应该既专业又易于理解，避免过度简化或过度复杂化。"""),
        ("human", """请基于以下推理过程生成解决方案:
        
        问题: {problem}
        领域: {domain}
        
        推理过程:
        {reasoning}
        
        请提供:
        1. 完整的解决方案
        2. 详细的解释和依据
        3. 实施建议(如适用)
        4. 局限性和注意事项
        
        以JSON格式返回，包含:
        - main_solution: 主要解决方案
        - explanation: 详细解释
        - implementation_steps: 实施步骤(如适用)
        - limitations: 局限性和注意事项
        - confidence_level: 置信度(0-1)
        """),
    ])
    
    # 创建解决方案生成链
    solution_chain = solution_prompt | llm
    
    # 执行解决方案生成
    result = solution_chain.invoke({
        "problem": problem,
        "domain": domain,
        "reasoning": reasoning_text
    })
    
    # 尝试解析结果
    try:
        # 尝试从回复中提取JSON
        import re
        json_str = re.search(r'```json\n(.*?)\n```', result.content, re.DOTALL)
        if json_str:
            solution_result = json.loads(json_str.group(1))
        else:
            json_str = re.search(r'{.*}', result.content, re.DOTALL)
            if json_str:
                solution_result = json.loads(json_str.group(0))
            else:
                # 创建一个简单的解决方案结果
                solution_result = {
                    "main_solution": "基于分析，建议采取以下行动...",
                    "explanation": "这个方案之所以有效是因为...",
                    "implementation_steps": ["第一步", "第二步"],
                    "limitations": ["这个方案的局限性在于..."],
                    "confidence_level": 0.7
                }
    except Exception as e:
        print(f"解决方案解析失败: {str(e)}")
        # 创建一个默认的解决方案结果
        solution_result = {
            "main_solution": "基于分析，建议采取以下行动...",
            "explanation": "这个方案之所以有效是因为...",
            "implementation_steps": ["第一步", "第二步"],
            "limitations": ["这个方案的局限性在于..."],
            "confidence_level": 0.7
        }
    
    # 提取解决方案和解释
    solution = {
        "main_solution": solution_result.get("main_solution", ""),
        "implementation_steps": solution_result.get("implementation_steps", []),
        "limitations": solution_result.get("limitations", []),
        "confidence_level": solution_result.get("confidence_level", 0.7)
    }
    
    explanation = solution_result.get("explanation", "")
    
    # 更新状态
    new_state["solution"] = solution
    new_state["explanation"] = explanation
    new_state["confidence_scores"] = {
        **(state.get("confidence_scores", {})),
        "solution_confidence": solution_result.get("confidence_level", 0.7)
    }
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "solution_generated_at": datetime.now().isoformat()
    }
    
    print(f"解决方案生成完成，置信度: {solution.get('confidence_level', 0.7):.2f}")
    
    return new_state 

# =================================================================
# 第4部分: 图结构构建 - 专家系统流程图
# =================================================================

def handle_uncertainty_node(state: ExpertSystemState) -> Dict[str, Any]:
    """处理不确定性节点
    
    WHY - 设计思路:
    1. 需要识别和量化推理过程中的不确定性
    2. 需要评估不同推理路径的可靠性
    
    HOW - 实现方式:
    1. 分析推理步骤中的不确定点
    2. 量化各个方面的置信度
    3. 提供不确定性的具体描述
    
    WHAT - 功能作用:
    评估并量化推理和解决方案中的不确定性，
    增强专家系统的可靠性和透明度
    
    Args:
        state: 当前专家系统状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取推理链和解决方案
    reasoning_chain = state.get("reasoning_chain", [])
    solution = state.get("solution", {})
    
    # 如果没有推理链，无法评估不确定性
    if not reasoning_chain:
        return {
            "confidence_scores": {
                "overall": 0.5,
                "completeness": 0.5,
                "consistency": 0.5,
                "evidence_strength": 0.5,
                "solution_confidence": 0.5
            }
        }
    
    # 使用LLM评估不确定性
    llm = get_llm(temperature=0.2)  # 低温度以获得客观评估
    
    uncertainty_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的不确定性分析专家，擅长识别和量化推理过程中的不确定因素。
        你需要分析给定的推理过程和解决方案，识别可能的不确定性来源，并量化各方面的置信度。
        
        请考虑以下不确定性来源:
        1. 信息的完整性和准确性
        2. 推理逻辑的严密性
        3. 前提假设的合理性
        4. 证据的强度和质量
        5. 解释的唯一性
        
        对每个方面，提供0-1之间的置信度评分(1表示完全确定)。
        同时，提供不确定性的具体描述和可能的影响。"""),
        ("human", """请评估以下推理过程和解决方案中的不确定性:
        
        推理步骤:
        {reasoning}
        
        解决方案:
        {solution}
        
        请提供不确定性分析，包括:
        1. 各方面的置信度评分(0-1)
        2. 不确定性的具体来源和描述
        3. 可能的影响和处理建议
        
        以JSON格式返回，包含:
        - confidence_scores: 各方面的置信度评分
        - uncertainty_sources: 不确定性来源列表
        - potential_impacts: 潜在影响列表
        - recommendations: 处理不确定性的建议
        """),
    ])
    
    # 准备推理步骤文本
    reasoning_text = ""
    for step in reasoning_chain:
        reasoning_text += f"\n步骤{step.get('step_number', '?')}: {step.get('description', '')}\n"
        reasoning_text += f"推理: {step.get('reasoning', '')}\n"
    
    solution_text = json.dumps(solution, ensure_ascii=False)
    
    # 创建不确定性评估链
    uncertainty_chain = uncertainty_prompt | llm
    
    # 执行不确定性评估
    result = uncertainty_chain.invoke({
        "reasoning": reasoning_text,
        "solution": solution_text
    })
    
    # 尝试解析结果
    try:
        # 尝试从回复中提取JSON
        import re
        json_str = re.search(r'```json\n(.*?)\n```', result.content, re.DOTALL)
        if json_str:
            uncertainty_result = json.loads(json_str.group(1))
        else:
            json_str = re.search(r'{.*}', result.content, re.DOTALL)
            if json_str:
                uncertainty_result = json.loads(json_str.group(0))
            else:
                # 创建一个简单的不确定性结果
                uncertainty_result = {
                    "confidence_scores": {
                        "overall": 0.7,
                        "completeness": 0.7,
                        "consistency": 0.8,
                        "evidence_strength": 0.7,
                        "solution_confidence": 0.7
                    },
                    "uncertainty_sources": ["信息不完整", "假设未经充分验证"],
                    "potential_impacts": ["可能影响解决方案的适用范围", "可能需要额外验证"],
                    "recommendations": ["获取更多相关信息", "考虑替代解释"]
                }
    except Exception as e:
        print(f"不确定性评估解析失败: {str(e)}")
        # 创建一个默认的不确定性结果
        uncertainty_result = {
            "confidence_scores": {
                "overall": 0.7,
                "completeness": 0.7,
                "consistency": 0.8,
                "evidence_strength": 0.7,
                "solution_confidence": 0.7
            },
            "uncertainty_sources": ["信息不完整", "假设未经充分验证"],
            "potential_impacts": ["可能影响解决方案的适用范围", "可能需要额外验证"],
            "recommendations": ["获取更多相关信息", "考虑替代解释"]
        }
    
    # 提取置信度评分和不确定性来源
    confidence_scores = uncertainty_result.get("confidence_scores", {})
    uncertainty_sources = uncertainty_result.get("uncertainty_sources", [])
    potential_impacts = uncertainty_result.get("potential_impacts", [])
    recommendations = uncertainty_result.get("recommendations", [])
    
    # 更新状态
    new_state["confidence_scores"] = confidence_scores
    new_state["context"] = {
        **(state.get("context", {})),
        "uncertainty_sources": uncertainty_sources,
        "potential_impacts": potential_impacts,
        "uncertainty_recommendations": recommendations
    }
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "uncertainty_assessed_at": datetime.now().isoformat()
    }
    
    print(f"不确定性评估完成，整体置信度: {confidence_scores.get('overall', 0.7):.2f}")
    
    return new_state

def uncertainty_route(state: ExpertSystemState) -> str:
    """不确定性路由函数
    
    WHY - 设计思路:
    1. 需要基于不确定性程度决定下一步操作
    2. 需要考虑置信度是否足够高
    
    HOW - 实现方式:
    1. 检查整体置信度评分
    2. 基于评分决定是否需要重新收集信息
    
    WHAT - 功能作用:
    作为LangGraph的条件路由函数，根据不确定性
    评估结果决定下一步流程，确保结果可靠性
    
    Args:
        state: 当前专家系统状态
        
    Returns:
        str: 下一个节点的名称
    """
    # 获取置信度评分
    confidence_scores = state.get("confidence_scores", {})
    overall_confidence = confidence_scores.get("overall", 0.0)
    
    # 如果置信度够高，进入生成最终解决方案阶段
    if overall_confidence >= 0.6:
        return "generate_solution"
    else:
        # 置信度不够，需要查询更多知识
        return "query_knowledge_base"

def build_expert_system_graph():
    """构建专家系统图
    
    WHY - 设计思路:
    1. 需要将专家系统的各个节点组织为工作流
    2. 需要实现基于不确定性的条件路由
    3. 需要支持知识检索和推理的循环改进
    
    HOW - 实现方式:
    1. 创建基于ExpertSystemState的StateGraph
    2. 添加信息收集、知识检索等节点
    3. 实现基于不确定性的条件路由
    4. 设置入口点和终点
    
    WHAT - 功能作用:
    构建一个完整的专家系统图，整合信息收集、
    知识检索、专家推理和不确定性处理等功能
    
    Returns:
        StateGraph: 编译后的专家系统图
    """
    # 创建图
    workflow = StateGraph(ExpertSystemState)
    
    # 添加节点
    workflow.add_node("gather_information", gather_information_node)
    workflow.add_node("query_knowledge_base", query_knowledge_base_node)
    workflow.add_node("expert_reasoning", expert_reasoning_node)
    workflow.add_node("handle_uncertainty", handle_uncertainty_node)
    workflow.add_node("generate_solution", generate_solution_node)
    
    # 设置流程
    # 1. 信息收集 -> 知识检索
    workflow.add_edge("gather_information", "query_knowledge_base")
    
    # 2. 知识检索 -> 专家推理
    workflow.add_edge("query_knowledge_base", "expert_reasoning")
    
    # 3. 专家推理 -> 不确定性处理
    workflow.add_edge("expert_reasoning", "handle_uncertainty")
    
    # 4. 不确定性处理 -> 条件路由(生成解决方案或重新检索知识)
    workflow.add_conditional_edges(
        "handle_uncertainty",
        uncertainty_route,
        {
            "generate_solution": "generate_solution",
            "query_knowledge_base": "query_knowledge_base"
        }
    )
    
    # 5. 生成解决方案 -> 结束
    workflow.add_edge("generate_solution", END)
    
    # 设置入口点
    workflow.set_entry_point("gather_information")
    
    # 编译图
    return workflow.compile(checkpointer=MemorySaver())

# =================================================================
# 第5部分: 示例演示 - 专家系统演示界面
# =================================================================

def show_expert_system_examples():
    """展示专家系统示例
    
    WHY - 设计思路:
    1. 需要提供交互式的示例演示
    2. 允许用户选择不同的问题领域
    3. 展示专家系统的推理过程
    
    HOW - 实现方式:
    1. 提供交互式菜单选择不同问题
    2. 调用专家系统处理问题
    3. 展示推理过程和解决方案
    
    WHAT - 功能作用:
    提供一个互动式展示，帮助用户理解专家系统的工作流程
    """
    print("\n===== LangGraph 专家系统示例 =====")
    
    print("\n本示例展示了基于LangGraph的专家系统，演示从问题分析到解决方案生成的专业推理流程。")
    
    # 预定义问题
    sample_problems = [
        {
            "problem": "患者出现持续高血压，收缩压150mmHg，舒张压95mmHg，伴有轻微头痛，该如何处理？",
            "domain": "医学",
            "context": {"patient_age": 45, "patient_gender": "男", "has_medication_history": False}
        },
        {
            "problem": "公司与供应商签订的合同中，对产品质量标准描述不明确，现在发生质量纠纷，如何处理？",
            "domain": "法律",
            "context": {"contract_signed_date": "2023-01-15", "dispute_value": "50000元"}
        },
        {
            "problem": "一个包含10万条记录的数据库查询非常缓慢，如何优化性能？",
            "domain": "计算机",
            "context": {"database_type": "MySQL", "query_type": "复杂JOIN查询", "current_response_time": "15秒"}
        },
        {
            "problem": "团队成员之间沟通不畅，影响项目进度，如何改善团队协作？",
            "domain": "通用",
            "context": {"team_size": 8, "project_duration": "6个月", "remote_work": True}
        }
    ]
    
    while True:
        print("\n请选择问题或自定义:")
        for i, problem in enumerate(sample_problems):
            print(f"{i+1}. [{problem['domain']}] {problem['problem'][:50]}...")
        print("5. 自定义问题")
        print("0. 返回主菜单")
        
        choice = input("\n您的选择> ")
        
        if choice == "0":
            break
        elif choice in ["1", "2", "3", "4"]:
            selected_problem = sample_problems[int(choice)-1]
            run_expert_system(selected_problem)
        elif choice == "5":
            # 自定义问题
            custom_problem = {}
            custom_problem["problem"] = input("请输入问题: ")
            custom_problem["domain"] = input("请输入领域(医学/法律/计算机/通用): ") or "通用"
            context_input = input("请输入上下文信息(可选，格式如: 年龄=45,性别=男): ")
            
            context = {}
            if context_input:
                try:
                    for item in context_input.split(","):
                        key, value = item.split("=")
                        context[key.strip()] = value.strip()
                except:
                    print("上下文格式错误，将使用空上下文")
            
            custom_problem["context"] = context
            run_expert_system(custom_problem)
        else:
            print("无效选择，请重试")

def run_expert_system(problem_config):
    """执行专家系统
    
    WHY - 设计思路:
    1. 需要执行完整的专家系统流程
    2. 需要展示推理过程和解决方案
    3. 需要处理可能的错误情况
    
    HOW - 实现方式:
    1. 构建专家系统图
    2. 初始化专家系统状态
    3. 调用LangGraph执行推理流程
    4. 展示推理过程和解决方案
    
    WHAT - 功能作用:
    执行专家系统流程并展示结果，是示例的核心执行函数
    
    Args:
        problem_config: 问题配置，包含problem、domain和context
    """
    print("\n" + "=" * 50)
    print(f"分析问题: '{problem_config['problem']}'")
    print("=" * 50)
    
    # 构建专家系统图
    print("构建专家系统图...")
    graph = build_expert_system_graph()
    
    # 创建初始状态
    print("\n初始化专家系统状态...")
    state = initialize_state(
        problem=problem_config["problem"],
        domain=problem_config["domain"],
        context=problem_config["context"]
    )
    
    print(f"\n问题信息:")
    print(f"- 问题: {problem_config['problem']}")
    print(f"- 领域: {problem_config['domain']}")
    if problem_config["context"]:
        print(f"- 上下文: {json.dumps(problem_config['context'], ensure_ascii=False)}")
    
    # 询问是否开始分析
    start = input("\n准备开始专家分析，按回车继续或输入'q'取消: ")
    if start.lower() == 'q':
        print("已取消分析")
        return
    
    # 运行专家系统
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    try:
        print("\n开始专家推理，这可能需要几分钟时间...")
        
        # 执行专家系统
        result = graph.invoke(state, config=config)
        
        # 打印专家系统结果
        print("\n" + "=" * 50)
        print("专家分析完成!")
        print("=" * 50)
        
        # 打印所需信息
        print("\n--- 所需信息 ---")
        if result.get("required_info"):
            for i, info in enumerate(result.get("required_info"), 1):
                print(f"{i}. {info}")
        else:
            print("未确定所需信息")
        
        # 打印推理过程摘要
        print("\n--- 推理过程摘要 ---")
        reasoning_chain = result.get("reasoning_chain", [])
        if reasoning_chain:
            for step in reasoning_chain[:3]:  # 只显示前3步
                print(f"步骤{step.get('step_number', '?')}: {step.get('description', '')}")
            if len(reasoning_chain) > 3:
                print(f"...（还有{len(reasoning_chain)-3}个步骤）")
        else:
            print("未生成推理过程")
        
        # 打印解决方案
        print("\n--- 解决方案 ---")
        solution = result.get("solution", {})
        if solution:
            print(f"主要方案: {solution.get('main_solution', '未提供')}")
            
            print("\n实施步骤:")
            implementation_steps = solution.get("implementation_steps", [])
            if implementation_steps:
                for i, step in enumerate(implementation_steps, 1):
                    print(f"{i}. {step}")
            else:
                print("未提供实施步骤")
                
            print("\n局限性和注意事项:")
            limitations = solution.get("limitations", [])
            if limitations:
                for i, limitation in enumerate(limitations, 1):
                    print(f"{i}. {limitation}")
            else:
                print("未提供局限性说明")
        else:
            print("未生成解决方案")
        
        # 打印置信度评估
        print("\n--- 置信度评估 ---")
        confidence_scores = result.get("confidence_scores", {})
        if confidence_scores:
            for aspect, score in confidence_scores.items():
                print(f"{aspect}: {score:.2f}")
        else:
            print("未提供置信度评估")
        
        # 询问是否查看详细信息
        view_details = input("\n是否查看详细解释? (y/n): ")
        if view_details.lower() == 'y':
            print("\n--- 详细解释 ---")
            explanation = result.get("explanation", "未提供解释")
            print(explanation)
            
            print("\n--- 替代解释路径 ---")
            alternative_paths = result.get("alternative_paths", [])
            if alternative_paths:
                for i, path in enumerate(alternative_paths, 1):
                    print(f"\n替代路径 {i}:")
                    print(f"描述: {path.get('description', '未提供')}")
                    print(f"推理: {path.get('reasoning', '未提供')}")
            else:
                print("未提供替代解释路径")
        
        # 询问是否保存结果
        save_result = input("\n是否保存分析结果到文件? (y/n): ")
        if save_result.lower() == 'y':
            # 保存结果到文件
            timestamp = int(time.time())
            output_file = f"expert_analysis_{timestamp}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                output = {
                    "problem": result.get("problem", ""),
                    "domain": result.get("domain", ""),
                    "required_info": result.get("required_info", []),
                    "reasoning_chain": result.get("reasoning_chain", []),
                    "solution": result.get("solution", {}),
                    "explanation": result.get("explanation", ""),
                    "confidence_scores": result.get("confidence_scores", {}),
                    "timestamp": datetime.now().isoformat()
                }
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            print(f"\n分析结果已保存到文件: {output_file}")
        
    except Exception as e:
        print(f"\n分析过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    input("\n按回车键返回主菜单...")

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行专家系统示例
    2. 需要适当的错误处理确保示例稳定运行
    3. 需要提供清晰的开始和结束提示
    
    HOW - 实现方式:
    1. 使用try-except包装主要执行逻辑
    2. 提供开始和结束提示
    3. 调用示例展示函数
    4. 总结关键学习点
    
    WHAT - 功能作用:
    作为程序入口点，执行专家系统示例，确保示例执行的稳定性
    """
    print("===== LangGraph 专家系统学习示例 =====\n")
    
    try:
        # 运行专家系统示例
        show_expert_system_examples()
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 构建基于LangGraph的专家系统")
        print("2. 实现领域知识表示和检索")
        print("3. 设计思维链式推理过程")
        print("4. 处理专家系统中的不确定性")
        print("5. 生成可解释的专业解决方案")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 