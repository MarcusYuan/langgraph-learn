#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 状态回放与历史追踪
===========================
本示例讲解LangGraph中的状态历史记录与回放功能:
1. 记录工作流执行过程中的状态历史
2. 查看与分析历史状态
3. 回放特定状态与调试
4. 历史状态的可视化与分析

WHY - 设计思路:
1. 在复杂应用中，需要了解工作流的执行过程和状态变化以进行调试
2. 对话系统可能需要"回到过去"的功能，重新从特定状态开始
3. 分析状态历史有助于优化工作流设计和识别瓶颈
4. 历史状态追踪对实现高级功能如撤销/重做至关重要

HOW - 实现方式:
1. 使用LangGraph的状态追踪API记录执行历史
2. 通过特定配置启用状态历史记录
3. 提供历史状态的查询、分析和回放接口
4. 结合可视化工具展示执行路径

WHAT - 功能作用:
通过本示例，你将学习如何在LangGraph应用中实现状态历史记录、分析和回放功能，
这对于开发复杂对话系统、调试应用以及构建高级用户体验非常有价值。

学习目标:
- 理解LangGraph中的状态历史记录机制
- 掌握如何查询和分析历史状态
- 学习从特定历史状态继续执行的技术
- 了解如何可视化状态历史以辅助调试
"""

import os
import time
import json
from typing import TypedDict, Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaLLM

# ===========================================================
# 第1部分: 状态定义
# ===========================================================

class HistoryState(TypedDict):
    """对话状态定义
    
    WHY - 设计思路:
    1. 需要一个结构化状态容器存储对话信息和历史相关元数据
    2. 状态中需要包含时间戳以追踪时间线
    3. 需要版本号或ID标识不同状态点
    
    HOW - 实现方式:
    1. 使用TypedDict提供类型安全和代码提示
    2. 定义消息历史字段存储对话上下文
    3. 添加元数据字段存储状态ID、时间戳等
    4. 添加备注字段记录每个状态点的特殊信息
    
    WHAT - 功能作用:
    提供一个类型安全的状态结构，包含完整的对话历史和元数据信息，
    支持状态回放、历史追踪和调试功能
    """
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # 消息历史
    metadata: Dict[str, Any]  # 元数据(包含状态ID、时间戳等)
    remarks: Optional[str]  # 状态备注，记录特殊信息
    iteration: Optional[int]  # 迭代次数，用于追踪状态变化
    bookmarks: Optional[List[str]]  # 书签列表，标记重要状态点

# ===========================================================
# 第2部分: 状态初始化与操作
# ===========================================================

def initialize_state() -> HistoryState:
    """初始化对话状态
    
    WHY - 设计思路:
    1. 每次会话开始需要一个干净且带有初始元数据的状态
    2. 需要自动生成唯一标识符用于状态追踪
    3. 需要记录创建时间以支持时间线分析
    
    HOW - 实现方式:
    1. 创建包含所有必要字段的HistoryState字典
    2. 使用当前时间戳作为会话ID和创建时间
    3. 设置初始迭代次数和空书签列表
    
    WHAT - 功能作用:
    提供一个干净的初始状态，确保每次对话开始时状态含有必要的元数据，
    支持后续的历史追踪和状态回放功能
    
    Returns:
        HistoryState: 初始化的状态对象
    """
    current_time = datetime.now()
    session_id = f"session_{int(current_time.timestamp())}"
    
    return {
        "messages": [
            SystemMessage(content="你是一个有用的AI助手，会详细记录和追踪对话状态。")
        ],
        "metadata": {
            "session_id": session_id,
            "created_at": current_time.isoformat(),
            "state_id": f"{session_id}_0",  # 初始状态ID
            "parent_state_id": None,  # 初始状态没有父状态
        },
        "remarks": "初始状态",
        "iteration": 0,
        "bookmarks": []
    }

def add_message(state: HistoryState, message: Union[HumanMessage, AIMessage, SystemMessage]) -> HistoryState:
    """向状态添加新消息
    
    WHY - 设计思路:
    1. 需要一个标准方法更新消息历史，同时保持状态不可变性
    2. 需要自动更新元数据如状态ID和时间戳
    3. 需要维护父子状态关系，形成状态树结构
    
    HOW - 实现方式:
    1. 复制所有现有状态字段，确保不可变性
    2. 追加新消息到消息历史列表
    3. 更新元数据中的时间戳、状态ID和父状态信息
    4. 递增迭代计数器，维护状态序列关系
    
    WHAT - 功能作用:
    提供一个标准化的消息添加函数，在添加消息的同时维护状态元数据，
    确保状态历史的完整性和可追踪性
    
    Args:
        state: 当前状态
        message: 要添加的消息
        
    Returns:
        HistoryState: 更新后的新状态
    """
    # 复制消息列表，保持不可变性
    messages = state["messages"].copy()
    messages.append(message)
    
    # 更新元数据
    metadata = state["metadata"].copy()
    new_iteration = state["iteration"] + 1 if state["iteration"] is not None else 1
    old_state_id = metadata["state_id"]
    metadata["state_id"] = f"{metadata['session_id']}_{new_iteration}"
    metadata["parent_state_id"] = old_state_id
    metadata["updated_at"] = datetime.now().isoformat()
    metadata["message_count"] = len(messages)
    
    # 复制书签列表
    bookmarks = state.get("bookmarks", []).copy() if state.get("bookmarks") else []
    
    return {
        "messages": messages,
        "metadata": metadata,
        "remarks": f"添加{'用户' if isinstance(message, HumanMessage) else 'AI'}消息",
        "iteration": new_iteration,
        "bookmarks": bookmarks
    }

def add_bookmark(state: HistoryState, bookmark_name: str) -> HistoryState:
    """添加状态书签
    
    WHY - 设计思路:
    1. 需要标记重要状态点，便于后续查找和回放
    2. 书签应该有描述性名称，便于理解状态含义
    3. 需要保持状态不可变性，返回新状态
    
    HOW - 实现方式:
    1. 复制并更新书签列表
    2. 确保书签名称唯一，避免重复
    3. 更新状态备注，记录书签添加操作
    
    WHAT - 功能作用:
    允许用户标记重要状态点，便于后续快速查找和回放特定状态，
    增强状态历史的可导航性和可用性
    
    Args:
        state: 当前状态
        bookmark_name: 书签名称
        
    Returns:
        HistoryState: 更新后的新状态
    """
    # 复制书签列表，保持不可变性
    bookmarks = state.get("bookmarks", []).copy() if state.get("bookmarks") else []
    
    # 添加新书签，避免重复
    if bookmark_name not in bookmarks:
        bookmarks.append(bookmark_name)
    
    # 更新元数据
    metadata = state["metadata"].copy()
    metadata["updated_at"] = datetime.now().isoformat()
    
    return {
        "messages": state["messages"],
        "metadata": metadata,
        "remarks": f"添加书签: {bookmark_name}",
        "iteration": state["iteration"],
        "bookmarks": bookmarks
    }

# ===========================================================
# 第3部分: 节点函数定义
# ===========================================================

def generate_response(state: HistoryState) -> HistoryState:
    """生成AI响应的节点函数
    
    WHY - 设计思路:
    1. 需要基于对话历史生成合适的AI回复
    2. 节点函数应更新状态，添加AI消息
    3. 保持纯函数设计，不改变输入状态
    
    HOW - 实现方式:
    1. 从状态提取消息历史
    2. 生成AI回复(这里简化为模拟回复)
    3. 创建AI消息并添加到状态
    4. 返回更新后的新状态
    
    WHAT - 功能作用:
    模拟AI响应生成流程，产生回复并更新状态，
    在实际应用中此处将调用LLM生成真实回复
    
    Args:
        state: 当前状态
        
    Returns:
        HistoryState: 更新后的新状态，包含AI回复
    """
    # 获取最后一条人类消息
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg.content
            break
    
    if not last_message:
        # 如果没有找到人类消息，返回默认回复
        ai_message = AIMessage(content="请问有什么可以帮助你的吗？")
        return add_message(state, ai_message)
    
    # 模拟AI回复生成
    # 在实际应用中，这里会调用LLM生成回复
    responses = [
        f"我理解你说的是关于'{last_message}'的问题。这是一个很好的问题！",
        f"关于'{last_message}'，我有几点想法可以分享...",
        f"根据我的理解，'{last_message}'可以从几个角度来看...",
        f"非常感谢你提问关于'{last_message}'的问题。让我来解答...",
    ]
    
    # 模拟延迟，更真实
    time.sleep(1)
    
    # 随机选择一个回复
    ai_response = random.choice(responses)
    
    # 创建AI消息并更新状态
    ai_message = AIMessage(content=ai_response)
    return add_message(state, ai_message)

def process_user_input(state: HistoryState, user_input: str) -> HistoryState:
    """处理用户输入的节点函数
    
    WHY - 设计思路:
    1. 需要将用户输入转换为消息并添加到状态
    2. 可能需要处理特殊命令，如书签添加
    3. 需要保持状态不可变性设计
    
    HOW - 实现方式:
    1. 检查输入是否为特殊命令
    2. 对于普通输入，创建HumanMessage并添加到状态
    3. 对于特殊命令，执行相应操作(如添加书签)
    
    WHAT - 功能作用:
    处理各类用户输入，支持普通消息和特殊命令，
    更新状态并为下一步处理做准备
    
    Args:
        state: 当前状态
        user_input: 用户输入文本
        
    Returns:
        HistoryState: 更新后的新状态
    """
    # 检查是否是书签命令
    if user_input.startswith("#bookmark "):
        bookmark_name = user_input[10:].strip()
        if bookmark_name:
            return add_bookmark(state, bookmark_name)
        # 如果书签名为空，当作普通消息处理
    
    # 创建用户消息
    user_message = HumanMessage(content=user_input)
    
    # 更新状态
    return add_message(state, user_message)

# ===========================================================
# 第4部分: 图构建
# ===========================================================

def create_history_tracking_graph() -> StateGraph:
    """创建支持历史追踪的对话图
    
    WHY - 设计思路:
    1. 需要一个简单的对话图结构来演示状态历史记录
    2. 图需要包含用户输入和AI响应两个基本节点
    3. 设计需要考虑历史状态的回放能力
    4. LangGraph的StateGraph需要保留执行历史以支持状态回放
    5. 图结构应足够简单，清晰展示核心概念而不引入不必要的复杂性
    
    HOW - 实现方式:
    1. 创建基于HistoryState的StateGraph
    2. 添加回复生成节点
    3. 设置节点间的边，形成完整对话循环
    4. 设置入口点为生成响应节点
    5. 不需要显式配置历史记录功能，LangGraph会自动保留状态变化
    
    WHAT - 功能作用:
    构建一个支持历史追踪的基本对话图，作为状态回放演示的基础，
    提供标准的对话循环结构。图实例会保留执行历史，便于后续通过
    get_state_history方法获取历史状态，实现状态查询、分析和回放。
    
    Returns:
        StateGraph: 编译好的状态图，支持状态历史记录和回放
    """
    # 创建图实例
    workflow = StateGraph(HistoryState)
    
    # 添加节点
    workflow.add_node("generate_response", generate_response)
    
    # 设置边 - 从响应生成到结束
    workflow.add_edge("generate_response", END)
    
    # 设置入口点
    workflow.set_entry_point("generate_response")
    
    # 编译图
    return workflow.compile()

# ===========================================================
# 第5部分: 状态历史分析函数
# ===========================================================

def print_state_history(config: RunnableConfig):
    """打印状态历史记录
    
    WHY - 设计思路:
    1. 需要可视化查看完整的状态历史记录以进行调试和分析
    2. 历史记录应包含关键元数据和状态变化，方便追踪状态演化
    3. 历史数据应有结构化展示，便于开发者快速理解和分析
    4. 状态历史记录是实现回放功能的基础，需要清晰展示
    5. 需要处理图实例不支持历史记录的边缘情况
    
    HOW - 实现方式:
    1. 从配置中使用get_state_history方法获取状态历史
    2. 检查图实例是否支持历史记录功能，提供优雅的回退
    3. 以格式化方式打印每个状态的关键信息（ID、父状态、迭代次数等）
    4. 使用分隔线提高历史记录的可读性
    5. 针对消息内容过长的情况进行截断处理
    6. 特别标记包含书签的状态，突出显示重要状态点
    
    WHAT - 功能作用:
    提供一个直观的状态历史展示功能，帮助开发者理解状态变化、
    调试应用并分析对话流程。通过可视化状态历史，开发者可以：
    1. 追踪状态的演变过程和分支关系
    2. 识别关键状态点和状态转换
    3. 分析状态回放的起点和过程
    4. 调试复杂工作流中的状态问题
    
    Args:
        config: 包含状态历史的运行配置，通常在调用图的invoke或stream方法时提供
    """
    print("\n===== 状态历史记录 =====")
    
    # 获取状态历史
    if not hasattr(graph, "get_state_history"):
        print("当前图实例不支持状态历史记录")
        return
    
    state_history = list(graph.get_state_history(config))
    
    if not state_history:
        print("没有找到状态历史记录")
        return
    
    print(f"共有 {len(state_history)} 个历史状态\n")
    
    # 打印每个状态的关键信息
    for i, state_record in enumerate(state_history):
        values = state_record.values
        next_node = state_record.next
        
        print(f"状态 #{i+1}:")
        print(f"  状态ID: {values['metadata']['state_id']}")
        print(f"  父状态ID: {values['metadata'].get('parent_state_id', 'None')}")
        print(f"  迭代次数: {values['iteration']}")
        print(f"  消息数量: {len(values['messages'])}")
        
        # 显示书签
        if values.get("bookmarks") and len(values["bookmarks"]) > 0:
            print(f"  书签: {', '.join(values['bookmarks'])}")
        
        # 显示备注
        if values.get("remarks"):
            print(f"  备注: {values['remarks']}")
        
        # 显示下一个节点
        print(f"  下一步: {next_node}")
        
        # 显示最后一条消息的概要
        if values["messages"]:
            last_msg = values["messages"][-1]
            msg_type = "系统" if isinstance(last_msg, SystemMessage) else "用户" if isinstance(last_msg, HumanMessage) else "AI"
            content = last_msg.content
            # 如果内容太长，只显示前50个字符
            if len(content) > 50:
                content = content[:47] + "..."
            print(f"  最后消息 ({msg_type}): {content}")
        
        print("-" * 50)

def find_state_by_bookmark(config: RunnableConfig, bookmark_name: str) -> Optional[Any]:
    """根据书签查找状态
    
    WHY - 设计思路:
    1. 在复杂的工作流执行历史中，需要快速定位语义化标记的特定状态
    2. 书签提供了一种基于意义而非技术属性检索历史状态的方式
    3. 找到的状态常用作回放起点，创建新的执行分支
    4. 用户通常按照业务语义而非技术ID来记忆和引用状态
    5. 书签系统支持在长对话历史中设置检查点和重要决策点
    
    HOW - 实现方式:
    1. 首先确认图实例支持历史记录功能
    2. 从配置中获取完整状态历史记录
    3. 遍历所有历史状态记录，检查每个状态的书签列表
    4. 返回第一个包含目标书签的状态记录
    5. 如果未找到匹配书签的状态，返回None表示查找失败
    6. 采用快速失败策略，避免在不支持历史记录的环境中继续操作
    
    WHAT - 功能作用:
    提供基于语义标记的状态检索功能，使开发者能够快速定位和访问
    重要状态点。这使得以下场景成为可能：
    1. 从特定的业务节点重新开始对话
    2. 基于关键决策点创建替代方案
    3. 比较不同执行路径的结果
    4. 实现高级功能如"回到过去"、"另存为"和分支执行
    5. 支持调试和演示特定工作流阶段
    
    Args:
        config: 包含状态历史的运行配置，必须与执行图时使用的配置一致
        bookmark_name: 目标书签名称，通常是具有业务含义的字符串
        
    Returns:
        Optional[Any]: 找到的状态记录对象，包含完整状态值和路由信息；
                     如果未找到匹配书签或不支持历史功能，则返回None
    """
    # 获取状态历史
    if not hasattr(graph, "get_state_history"):
        print("当前图实例不支持状态历史记录")
        return None
    
    state_history = graph.get_state_history(config)
    
    for state_record in state_history:
        values = state_record.values
        bookmarks = values.get("bookmarks", [])
        
        if bookmark_name in bookmarks:
            return state_record
    
    return None

def compare_states(state1: HistoryState, state2: HistoryState) -> Dict[str, Any]:
    """比较两个状态的差异
    
    WHY - 设计思路:
    1. 在状态回放和分支场景中，需要精确了解两个状态之间的具体变化
    2. 直观的差异比较有助于理解状态演化路径和决策点
    3. 差异信息对调试、优化和审计历史记录很有价值
    4. 结构化的差异表示便于程序化处理和可视化
    5. 在复杂工作流中，了解状态分支间的差异可以帮助选择最佳路径
    
    HOW - 实现方式:
    1. 采用分类比较策略，将状态差异分为消息、元数据和其他变化
    2. 检查消息列表长度变化，识别新增消息
    3. 深入比较元数据字段，追踪添加、删除和修改的键值
    4. 比较关键状态字段如备注、迭代次数和书签
    5. 构建结构化的差异报告字典，便于后续处理
    6. 对于消息内容，保留完整内容而非仅比较引用
    
    WHAT - 功能作用:
    提供状态差异分析功能，帮助开发者精确理解状态变化过程。
    这对于以下场景特别有价值：
    1. 调试复杂的工作流逻辑和状态转换
    2. 理解状态回放时的分支变化
    3. 审计和记录关键决策点的状态变化
    4. 识别可能的优化点或问题区域
    5. 可视化多路径执行的差异
    
    Args:
        state1: 第一个状态（通常是原始状态或基准状态）
        state2: 第二个状态（通常是变化后的状态或比较目标）
        
    Returns:
        Dict[str, Any]: 包含差异信息的结构化字典，分为消息变化、
                       元数据变化和其他变化三个主要部分
    """
    # 初始化差异字典
    diff = {
        "message_changes": {},
        "metadata_changes": {},
        "other_changes": {}
    }
    
    # 比较消息列表
    messages1 = state1["messages"]
    messages2 = state2["messages"]
    
    diff["message_changes"]["count_diff"] = len(messages2) - len(messages1)
    
    if len(messages2) > len(messages1):
        # 有新消息添加
        new_messages = messages2[len(messages1):]
        diff["message_changes"]["new_messages"] = [
            {"type": type(msg).__name__, "content": msg.content} 
            for msg in new_messages
        ]
    
    # 比较元数据
    metadata1 = state1["metadata"]
    metadata2 = state2["metadata"]
    
    for key in set(metadata1.keys()) | set(metadata2.keys()):
        if key not in metadata1:
            diff["metadata_changes"][key] = {"added": metadata2[key]}
        elif key not in metadata2:
            diff["metadata_changes"][key] = {"removed": metadata1[key]}
        elif metadata1[key] != metadata2[key]:
            diff["metadata_changes"][key] = {
                "from": metadata1[key],
                "to": metadata2[key]
            }
    
    # 比较其他字段
    for key in ["remarks", "iteration", "bookmarks"]:
        if state1.get(key) != state2.get(key):
            diff["other_changes"][key] = {
                "from": state1.get(key),
                "to": state2.get(key)
            }
    
    return diff

# ===========================================================
# 第6部分: 执行和演示
# ===========================================================

def run_history_tracking_example():
    """运行历史追踪示例
    
    WHY - 设计思路:
    1. 需要通过实际运行的例子展示状态历史记录和回放完整功能流程
    2. 示例应该覆盖从状态创建、记录到回放的全流程，形成闭环
    3. 步骤应该清晰，每个关键功能点都有直观展示
    4. 示例应该模拟真实对话场景，展示实际应用中的使用方式
    5. 需要包含异常路径和分支路径，展示状态回放的强大功能
    
    HOW - 实现方式:
    1. 初始化图结构和初始状态，配置状态追踪
    2. 模拟多轮对话交互，生成丰富的状态历史
    3. 在关键点添加书签，标记重要状态
    4. 展示历史记录查询和可视化功能
    5. 从特定书签状态创建新的执行分支
    6. 比较原始状态路径和新分支路径的差异
    7. 通过打印状态和结果，直观展示各步骤效果
    
    WHAT - 功能作用:
    提供一个端到端的状态历史记录与回放演示，具体包括：
    1. 创建带元数据的对话状态并追踪变化
    2. 使用书签标记关键状态点
    3. 检索和分析历史状态记录
    4. 从历史状态创建新的执行分支
    5. 对比不同执行路径的状态差异
    
    这个示例既是功能展示，也是学习指南，帮助开发者理解
    如何在自己的LangGraph应用中实现状态回放功能。
    """
    print("\n===== 状态历史追踪与回放示例 =====")
    
    # 初始化图和状态
    global graph
    graph = create_history_tracking_graph()
    state = initialize_state()
    
    # 设置追踪配置
    config = {
        "recursion_limit": 25,
        "configurable": {"thread_id": "history_demo"}
    }
    
    # 模拟对话 1
    print("\n--- 模拟对话场景 ---")
    user_input1 = "你好，我想了解LangGraph的状态管理机制"
    print(f"用户: {user_input1}")
    
    # 处理用户输入
    state = process_user_input(state, user_input1)
    
    # 生成回复
    state = graph.invoke(state, config)
    print(f"AI: {state['messages'][-1].content}")
    
    # 添加书签
    state = add_bookmark(state, "初次询问")
    print("(已添加书签: 初次询问)")
    
    # 模拟对话 2
    user_input2 = "状态回放功能有什么用途？"
    print(f"\n用户: {user_input2}")
    
    state = process_user_input(state, user_input2)
    state = graph.invoke(state, config)
    print(f"AI: {state['messages'][-1].content}")
    
    # 模拟对话 3
    user_input3 = "如何实现状态历史的存储和检索？"
    print(f"\n用户: {user_input3}")
    
    state = process_user_input(state, user_input3)
    state = graph.invoke(state, config)
    print(f"AI: {state['messages'][-1].content}")
    
    # 添加书签
    state = add_bookmark(state, "技术细节")
    print("(已添加书签: 技术细节)")
    
    # 打印状态历史
    print_state_history(config)
    
    # 查找特定书签的状态
    print("\n--- 根据书签查找状态 ---")
    bookmark_name = "初次询问"
    bookmarked_state = find_state_by_bookmark(config, bookmark_name)
    
    if bookmarked_state:
        print(f"找到书签 '{bookmark_name}' 对应的状态:")
        print(f"  状态ID: {bookmarked_state.values['metadata']['state_id']}")
        print(f"  消息数量: {len(bookmarked_state.values['messages'])}")
        print(f"  最后消息: {bookmarked_state.values['messages'][-1].content[:50]}...")
    else:
        print(f"未找到书签 '{bookmark_name}' 对应的状态")
    
    # 从特定状态继续执行
    print("\n--- 从特定状态回放 ---")
    if bookmarked_state:
        print(f"从书签 '{bookmark_name}' 状态继续执行...")
        replay_input = "从这个状态继续，我想了解状态回放的高级用例"
        print(f"用户: {replay_input}")
        
        # 从书签状态创建新状态分支
        replay_state = process_user_input(bookmarked_state.values, replay_input)
        
        # 生成回复
        replay_state = graph.invoke(replay_state, config)
        print(f"AI: {replay_state['messages'][-1].content}")
        
        # 比较原始状态和回放后的状态
        print("\n--- 状态比较 ---")
        
        diff = compare_states(state, replay_state)
        print("状态差异分析:")
        print(f"  消息数量变化: {diff['message_changes'].get('count_diff', 0)}")
        
        if "new_messages" in diff["message_changes"]:
            print("  新增消息:")
            for msg in diff["message_changes"]["new_messages"]:
                print(f"    - [{msg['type']}]: {msg['content'][:50]}...")
        
        print("  元数据变化:")
        for key, change in diff["metadata_changes"].items():
            if "from" in change and "to" in change:
                print(f"    - {key}: 从 {change['from']} 变为 {change['to']}")
        
        # 打印更新后的状态历史
        print("\n更新后的状态历史:")
        print_state_history(config)

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行所有状态回放示例
    2. 需要适当的错误处理确保示例运行稳定
    3. 需要清晰的开始和结束提示
    4. 用户需要一个简洁总结，帮助理解所学内容
    5. 错误应被捕获并展示，方便调试和理解
    
    HOW - 实现方式:
    1. 使用try-except包装主要执行逻辑，捕获状态回放过程中的异常
    2. 提供明确的开始和结束提示，包围主要执行流程
    3. 调用具体示例函数执行状态追踪和回放演示
    4. 运行结束后提供学习总结点，强化关键概念
    5. 打印完整的错误栈，帮助定位问题
    
    WHAT - 功能作用:
    提供程序入口点，执行状态历史追踪与回放示例，确保示例运行平稳，
    增强用户体验和代码可靠性。通过统一的入口点，用户可以完整体验
    状态历史记录、书签标记、状态回放和差异分析的全部功能。
    """
    print("===== LangGraph 状态回放与历史追踪示例 =====\n")
    
    try:
        # 运行历史追踪示例
        run_history_tracking_example()
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 记录和查询状态历史")
        print("2. 使用书签标记重要状态")
        print("3. 从历史状态回放执行")
        print("4. 分析不同状态之间的差异")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 