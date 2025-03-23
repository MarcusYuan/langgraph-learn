#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 状态回放与历史追踪
==========================
本示例讲解LangGraph中的状态历史记录和回放功能:
1. 状态历史记录与保存
2. 会话回放与分支执行
3. 状态分析与调试

WHY - 设计思路:
1. 复杂对话和执行流程需要可追溯性和可分析能力
2. 关键决策点需要可视化和可回溯
3. 调试复杂交互过程需要历史状态访问
4. 支持根据用户反馈，从历史节点重启对话
5. 帮助开发者理解和优化对话流程

HOW - 实现方式:
1. 利用LangGraph内置的状态历史记录能力
2. 设计特殊的状态结构用于记录关键点
3. 实现书签功能标记重要状态点
4. 开发状态比较和分析工具
5. 构建状态回放机制

WHAT - 功能作用:
通过本示例，你将学习如何在LangGraph中记录、分析和回放状态历史，
这对于构建可调试、可追溯的复杂对话系统至关重要，可用于对话修复、
分支对话、"假设如果"场景模拟等高级对话功能。

学习目标:
- 理解LangGraph的状态历史记录机制
- 掌握状态回放和分支执行
- 学习如何分析和比较不同状态
- 实现书签功能用于标记关键状态点
"""

import os
import json
import time
from typing import TypedDict, Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import copy

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_ollama import OllamaLLM

# ===========================================================
# 第1部分: 状态定义
# ===========================================================

class HistoryState(TypedDict):
    """状态历史记录与回放状态定义
    
    WHY - 设计思路:
    1. 需要记录对话历史和状态变化
    2. 需要添加元数据用于分析和调试
    3. 支持在状态中添加书签功能，便于快速定位关键状态
    4. 记录迭代计数器方便追踪对话轮次
    
    HOW - 实现方式:
    1. 使用TypedDict提供类型安全和代码提示
    2. 设计messages字段存储对话历史
    3. 设计metadata字段存储额外信息
    4. 添加bookmarks字段用于标记关键状态点
    5. 添加iteration计数器跟踪对话轮次
    
    WHAT - 功能作用:
    提供一个可追踪、可分析的状态结构，支持历史记录、状态回放
    和状态分析功能，为构建可调试的复杂对话系统提供基础
    """
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # 消息历史
    metadata: Dict[str, Any]  # 元数据
    remarks: Optional[str]  # 状态备注
    iteration: int  # 迭代次数计数器
    bookmarks: List[str]  # 状态书签列表，用于快速定位关键状态

# ===========================================================
# 第2部分: 状态管理函数
# ===========================================================

def initialize_state() -> HistoryState:
    """初始化状态
    
    WHY - 设计思路:
    1. 需要为对话系统提供一个干净的初始状态
    2. 初始状态需要包含基本的元数据和系统提示
    3. 初始化各个字段，确保后续操作安全
    
    HOW - 实现方式:
    1. 创建包含所有必要字段的HistoryState字典
    2. 添加初始系统消息设置对话基调
    3. 初始化元数据、备注和书签列表
    4. 设置迭代计数器为0
    
    WHAT - 功能作用:
    提供对话系统的起点状态，初始化所有必要字段，
    为后续的状态变更和历史记录奠定基础
    
    Returns:
        HistoryState: 初始化的状态
    """
    current_time = datetime.now()
    
    return {
        "messages": [
            SystemMessage(content="这是一个支持状态回放和历史追踪的对话系统。你可以通过特殊命令添加书签，查看历史，或从特定状态重新开始对话。")
        ],
        "metadata": {
            "created_at": current_time.isoformat(),
            "session_id": f"session-{int(time.time())}",
        },
        "remarks": "初始状态",
        "iteration": 0,
        "bookmarks": []
    }

def add_message(state: HistoryState, message: Union[HumanMessage, AIMessage]) -> HistoryState:
    """添加消息到状态
    
    WHY - 设计思路:
    1. 需要保持状态不变性，每次更新都返回新状态
    2. 需要追踪迭代次数，标记对话进展
    3. 消息添加是最常见的状态更新操作，需要特别关注
    
    HOW - 实现方式:
    1. 创建状态的深拷贝，确保不修改原状态
    2. 向消息列表添加新消息
    3. 更新迭代计数器
    4. 可选地添加备注说明此次更新
    
    WHAT - 功能作用:
    在保持状态不变性的前提下，添加新消息到对话历史，
    同时更新迭代计数器，支持对话流程的推进
    
    Args:
        state: 当前状态
        message: 要添加的消息
        
    Returns:
        HistoryState: 更新后的新状态
    """
    # 创建状态的深拷贝，确保不可变性
    new_state = copy.deepcopy(state)
    
    # 添加消息
    new_state["messages"].append(message)
    
    # 更新迭代计数器
    new_state["iteration"] += 1
    
    # 更新备注
    message_type = "用户" if isinstance(message, HumanMessage) else "AI"
    new_state["remarks"] = f"添加了{message_type}消息 (迭代 {new_state['iteration']})"
    
    return new_state

def add_bookmark(state: HistoryState, bookmark_name: str) -> HistoryState:
    """添加书签到状态
    
    WHY - 设计思路:
    1. 需要标记关键状态点便于回溯
    2. 语义化书签比记住状态ID更友好
    3. 支持开发者和用户标记重要节点
    
    HOW - 实现方式:
    1. 创建状态的深拷贝，确保不修改原状态
    2. 向书签列表添加新书签
    3. 可选地添加备注说明此次更新
    4. 确保书签不重复
    
    WHAT - 功能作用:
    允许给状态添加语义化书签，便于后续通过书签名
    快速定位到特定状态，用于回放或分支执行
    
    Args:
        state: 当前状态
        bookmark_name: 书签名称
        
    Returns:
        HistoryState: 更新后的新状态
    """
    # 创建状态的深拷贝，确保不可变性
    new_state = copy.deepcopy(state)
    
    # 确保书签不重复
    if bookmark_name not in new_state["bookmarks"]:
        new_state["bookmarks"].append(bookmark_name)
    
    # 更新备注
    new_state["remarks"] = f"添加了书签: {bookmark_name}"
    
    return new_state

# ===========================================================
# 第3部分: 对话节点函数
# ===========================================================

def get_llm():
    """获取LLM实例
    
    WHY - 设计思路:
    1. 需要一个可复用的LLM获取函数
    2. 便于统一配置和更换底层模型
    
    HOW - 实现方式:
    1. 使用langchain_ollama提供本地LLM能力
    2. 配置合适的参数确保输出质量
    
    WHAT - 功能作用:
    提供一个配置好的LLM实例，供各节点使用，
    确保整个对话系统使用相同的底层模型配置
    """
    return OllamaLLM(
        model="qwen:0.5b",  # 可替换为其他可用模型
        temperature=0.7,
    )

def ai_node(state: HistoryState) -> Dict:
    """AI响应节点
    
    WHY - 设计思路:
    1. 需要一个专门的节点处理AI的响应生成
    2. 响应需要考虑完整的对话历史
    3. 响应生成是对话流程的关键环节
    
    HOW - 实现方式:
    1. 从状态中提取消息历史
    2. 使用LLM基于历史生成响应
    3. 使用add_message函数添加响应到状态
    
    WHAT - 功能作用:
    基于当前状态生成AI响应，并将响应添加到对话历史，
    推进对话流程，同时确保状态更新的一致性
    
    Args:
        state: 当前状态
        
    Returns:
        Dict: 包含更新状态的字典
    """
    print(f"AI思考中...(迭代 {state['iteration']})")
    
    # 获取LLM
    llm = get_llm()
    
    # 生成响应
    response = llm.invoke(state["messages"])
    
    # 添加回复到状态
    new_state = add_message(state, response)
    
    return new_state

def user_input_node(state: HistoryState) -> Dict:
    """用户输入处理节点
    
    WHY - 设计思路:
    1. 需要处理用户输入，包括常规问题和特殊命令
    2. 支持添加书签、查看历史等特殊操作
    3. 确保用户体验的连贯性
    
    HOW - 实现方式:
    1. 获取用户输入
    2. 检查是否为特殊命令（如添加书签）
    3. 根据输入类型执行相应操作
    4. 更新状态并返回
    
    WHAT - 功能作用:
    处理用户输入，支持常规对话和特殊命令操作，
    并将输入或操作结果更新到状态中
    
    Args:
        state: 当前状态
        
    Returns:
        Dict: 包含更新状态的字典
    """
    # 获取用户输入
    user_input = input(f"\n您的输入(迭代 {state['iteration']})> ")
    
    # 检查是否为特殊命令
    if user_input.startswith("/bookmark "):
        # 添加书签
        bookmark_name = user_input[10:].strip()
        if bookmark_name:
            print(f"已添加书签: {bookmark_name}")
            new_state = add_bookmark(state, bookmark_name)
            # 在书签状态添加系统消息提示
            system_msg = SystemMessage(content=f"用户在此处添加了书签: {bookmark_name}")
            new_state = add_message(new_state, system_msg)
            return new_state
    
    # 处理普通消息
    message = HumanMessage(content=user_input)
    new_state = add_message(state, message)
    
    return new_state

# ===========================================================
# 第4部分: 状态历史分析函数
# ===========================================================

def print_state_history(graph):
    """打印状态历史记录
    
    WHY - 设计思路:
    1. 需要直观查看完整的状态历史便于调试和分析
    2. 历史记录需要包含关键元数据和状态变化
    3. 显示格式需要清晰易读，便于开发者理解
    4. 处理不支持历史的图实例
    5. 展示历史记录有助于追踪对话全流程
    
    HOW - 实现方式:
    1. 使用graph.get_state_history获取历史记录
    2. 检查是否支持历史功能
    3. 循环历史记录，格式化显示关键信息
    4. 处理长消息内容，确保显示合理
    
    WHAT - 功能作用:
    提供一个直观的状态历史显示，帮助开发者跟踪状态演变，
    分析对话流程，识别关键节点，调试复杂工作流
    
    Args:
        graph: LangGraph图实例
    """
    print("\n===== 状态历史记录 =====")
    
    # 尝试获取状态历史
    try:
        history = graph.get_state_history()
        if not history:
            print("图实例未记录状态历史或历史为空")
            return
    except (AttributeError, NotImplementedError):
        print("此图实例不支持状态历史功能")
        return
    
    # 打印历史记录
    for i, state_snapshot in enumerate(history):
        # 提取基本信息
        values = state_snapshot.values
        next_nodes = state_snapshot.next
        
        # 打印基本状态信息
        print(f"\n历史记录 #{i+1} - 下一节点: {next_nodes}")
        print(f"  迭代: {values.get('iteration', 'N/A')}")
        print(f"  备注: {values.get('remarks', 'N/A')}")
        
        # 打印书签
        bookmarks = values.get('bookmarks', [])
        if bookmarks:
            print(f"  书签: {', '.join(bookmarks)}")
        
        # 显示最后一条消息
        messages = values.get('messages', [])
        if messages:
            last_msg = messages[-1]
            msg_type = "系统" if isinstance(last_msg, SystemMessage) else "用户" if isinstance(last_msg, HumanMessage) else "AI"
            content = last_msg.content
            
            # 处理长消息
            if len(content) > 100:
                content = content[:97] + "..."
                
            print(f"  最后消息({msg_type}): {content}")
            
        print("-" * 50)

def find_state_by_bookmark(graph, bookmark_name: str):
    """根据书签名查找历史状态
    
    WHY - 设计思路:
    1. 需要快速定位带有特定语义标记的历史状态
    2. 书签提供了有意义的方式来标记和检索状态
    3. 从标记点重新开始是状态回放的重要功能
    4. 语义化查找比基于索引更有意义
    
    HOW - 实现方式:
    1. 从图实例获取完整状态历史
    2. 检查是否支持历史功能
    3. 遍历所有历史状态记录
    4. 检查每个状态的书签列表是否包含目标书签
    5. 返回找到的第一个匹配状态
    
    WHAT - 功能作用:
    提供基于语义的状态检索能力，使开发者能快速定位并访问
    标记过的关键状态点，用于回放、分支或调试
    
    Args:
        graph: LangGraph图实例
        bookmark_name: 要查找的书签名称
        
    Returns:
        找到的状态记录或None
    """
    # 尝试获取状态历史
    try:
        history = graph.get_state_history()
        if not history:
            print("图实例未记录状态历史或历史为空")
            return None
    except (AttributeError, NotImplementedError):
        print("此图实例不支持状态历史功能")
        return None
    
    # 查找带有特定书签的状态
    for state_snapshot in history:
        values = state_snapshot.values
        bookmarks = values.get('bookmarks', [])
        
        if bookmark_name in bookmarks:
            return state_snapshot
    
    print(f"未找到书签名为 '{bookmark_name}' 的状态")
    return None

def compare_states(state1: HistoryState, state2: HistoryState) -> Dict:
    """比较两个状态的差异
    
    WHY - 设计思路:
    1. 需要精确了解两个状态间的差异用于分析
    2. 直观的差异比较有助于理解状态演化路径
    3. 差异比较可帮助识别关键决策点
    
    HOW - 实现方式:
    1. 比较两个状态的各个关键字段
    2. 检查消息历史的变化
    3. 比较元数据和书签的区别
    4. 构建结构化的差异报告
    
    WHAT - 功能作用:
    提供状态差异分析，帮助开发者理解状态变化，
    特别有价值于调试复杂工作流、理解分支变化，
    审计关键决策点、识别优化区域、可视化差异
    
    Args:
        state1: 第一个状态
        state2: 第二个状态
        
    Returns:
        Dict: 包含差异信息的字典
    """
    differences = {
        "message_changes": [],
        "metadata_changes": {},
        "other_changes": {}
    }
    
    # 比较消息历史
    msgs1 = state1.get("messages", [])
    msgs2 = state2.get("messages", [])
    
    # 检查新增消息
    if len(msgs2) > len(msgs1):
        new_messages = msgs2[len(msgs1):]
        for msg in new_messages:
            msg_type = "系统" if isinstance(msg, SystemMessage) else "用户" if isinstance(msg, HumanMessage) else "AI"
            content = msg.content
            if len(content) > 100:
                content = content[:97] + "..."
            differences["message_changes"].append(f"新增{msg_type}消息: {content}")
    
    # 比较元数据
    meta1 = state1.get("metadata", {})
    meta2 = state2.get("metadata", {})
    
    for key in set(meta2.keys()) | set(meta1.keys()):
        if key in meta2 and key in meta1:
            if meta2[key] != meta1[key]:
                differences["metadata_changes"][key] = {
                    "from": meta1[key],
                    "to": meta2[key]
                }
        elif key in meta2:
            differences["metadata_changes"][key] = {
                "from": None,
                "to": meta2[key]
            }
        else:
            differences["metadata_changes"][key] = {
                "from": meta1[key],
                "to": None
            }
    
    # 比较其他字段
    for field in ["remarks", "iteration", "bookmarks"]:
        if state1.get(field) != state2.get(field):
            differences["other_changes"][field] = {
                "from": state1.get(field),
                "to": state2.get(field)
            }
    
    return differences

# ===========================================================
# 第5部分: 图构建
# ===========================================================

def create_history_tracking_graph():
    """创建支持历史追踪的图
    
    WHY - 设计思路:
    1. 需要一个支持状态历史记录和回放的图结构
    2. 图结构需要处理用户输入和AI响应
    3. 必须确保状态变更被正确记录
    4. 图的结构应简单清晰，便于理解核心概念
    5. 需要自动保留执行历史以支持状态回放
    
    HOW - 实现方式:
    1. 创建基于HistoryState的StateGraph
    2. 配置内存存储器用于保存历史记录
    3. 添加用户输入和AI响应节点
    4. 配置节点间的循环关系实现对话
    5. 使用LangGraph的状态历史功能记录所有变更
    
    WHAT - 功能作用:
    提供一个支持历史记录和回放的对话图结构，
    自动保留所有状态变更，支持基于历史的分析和回放，
    是实现状态回放功能的核心组件
    
    Returns:
        编译好的图实例
    """
    # 创建状态图
    workflow = StateGraph(HistoryState)
    
    # 添加节点
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("ai_response", ai_node)
    
    # 设置边 - 循环对话模式
    workflow.add_edge("user_input", "ai_response")
    workflow.add_edge("ai_response", "user_input")
    
    # 设置入口点
    workflow.set_entry_point("user_input")
    
    # 配置内存存储器，用于保存状态历史
    memory = MemorySaver()
    
    # 编译图并配置存储器
    return workflow.compile(checkpointer=memory)

# ===========================================================
# 第6部分: 演示和示例
# ===========================================================

def run_history_tracking_example():
    """运行状态历史追踪示例
    
    WHY - 设计思路:
    1. 需要一个实践示例展示状态历史记录和回放的完整功能
    2. 示例需要覆盖从创建到回放的整个过程
    3. 每个步骤需要有清晰的说明
    4. 模拟真实对话场景，体现功能价值
    
    HOW - 实现方式:
    1. 创建支持历史追踪的图实例
    2. 初始化状态并开始对话
    3. 在关键点添加书签
    4. 展示历史记录和查询功能
    5. 演示从特定状态回放的能力
    6. 比较不同执行路径的状态
    
    WHAT - 功能作用:
    提供状态历史追踪和回放的端到端示例，展示如何创建状态，
    添加书签，查询历史，从历史点回放，以及分析状态差异，
    帮助开发者理解和应用这些高级功能
    """
    print("\n===== 状态历史追踪与回放示例 =====")
    
    # 创建图实例
    graph = create_history_tracking_graph()
    
    # 初始化状态
    initial_state = initialize_state()
    
    print("\n开始对话，您可以使用以下特殊命令:")
    print("  /bookmark <名称> - 为当前状态添加书签")
    print("输入'退出'结束对话")
    
    # 运行对话几轮，以演示历史记录
    current_state = initial_state
    
    # 模拟第一轮对话
    print("\n--- 第一轮对话 ---")
    user_msg = HumanMessage(content="你好，这是一个测试消息")
    current_state = add_message(current_state, user_msg)
    
    # 添加书签
    current_state = add_bookmark(current_state, "对话开始")
    print("系统自动添加书签: 对话开始")
    
    # AI响应
    ai_msg = AIMessage(content="你好！我是AI助手。这是状态历史追踪示例。你可以尝试添加书签，之后我们将演示如何回放历史状态。")
    current_state = add_message(current_state, ai_msg)
    
    # 模拟第二轮对话
    print("\n--- 第二轮对话 ---")
    user_msg = HumanMessage(content="请解释一下什么是状态回放？")
    current_state = add_message(current_state, user_msg)
    
    ai_msg = AIMessage(content="状态回放是指从历史记录的某个特定状态点重新开始对话或任务执行的功能。这对于调试、创建替代对话分支、或者基于用户反馈重新执行十分有用。在LangGraph中，我们可以记录所有状态变更，并从任意历史点恢复和继续执行。")
    current_state = add_message(current_state, ai_msg)
    
    # 添加书签
    current_state = add_bookmark(current_state, "解释状态回放")
    print("系统自动添加书签: 解释状态回放")
    
    # 模拟第三轮对话
    print("\n--- 第三轮对话 ---")
    user_msg = HumanMessage(content="这有什么实际应用场景吗？")
    current_state = add_message(current_state, user_msg)
    
    ai_msg = AIMessage(content="状态回放有很多实际应用场景，例如：\n1. 对话修复：当AI回复不理想时，从之前的状态重新生成回复\n2. 创建平行对话：从关键决策点创建多个对话分支，探索不同方向\n3. 假设分析：模拟'如果用户说X会怎样'的场景\n4. 开发调试：帮助开发者理解复杂对话流程和状态变化\n5. 用户体验：允许用户返回对话历史中的任意点重新开始")
    current_state = add_message(current_state, ai_msg)
    
    # 将状态提交到图
    graph.invoke(current_state)
    
    # 打印状态历史
    print("\n展示完整状态历史:")
    print_state_history(graph)
    
    # 查找特定书签状态
    print("\n查找'对话开始'书签状态:")
    bookmark_state = find_state_by_bookmark(graph, "对话开始")
    
    if bookmark_state:
        print(f"找到书签状态 - 迭代: {bookmark_state.values.get('iteration')}")
        
        # 从书签状态创建新的分支
        print("\n从'对话开始'状态创建新的对话分支:")
        
        # 模拟从书签状态开始的新对话
        branch_state = bookmark_state.values
        
        # 添加不同的用户消息
        branch_user_msg = HumanMessage(content="我想了解状态历史记录的其他功能")
        branch_state = add_message(branch_state, branch_user_msg)
        
        branch_ai_msg = AIMessage(content="状态历史记录还有许多有用功能，如状态比较、差异分析、可视化状态变化等。这些功能对于理解复杂对话系统的行为非常有价值，特别是在调试和优化阶段。")
        branch_state = add_message(branch_state, branch_ai_msg)
        
        # 添加书签
        branch_state = add_bookmark(branch_state, "分支对话")
        
        # 将分支状态提交到图
        graph.invoke(branch_state)
        
        # 再次打印状态历史
        print("\n添加分支后的状态历史:")
        print_state_history(graph)
        
        # 比较原始路径和分支路径
        original_state = current_state
        print("\n比较原始对话和分支对话的差异:")
        differences = compare_states(original_state, branch_state)
        
        # 打印差异
        print("\n消息变化:")
        for change in differences["message_changes"]:
            print(f"  - {change}")
            
        print("\n其他变化:")
        for field, change in differences["other_changes"].items():
            print(f"  - {field}: 从 '{change['from']}' 变为 '{change['to']}'")
    
    print("\n===== 状态历史追踪与回放示例结束 =====")
    
    return graph

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行所有示例
    2. 需要适当的错误处理确保程序稳定
    3. 需要提供清晰的开始和结束提示
    4. 总结学习要点强化理解
    
    HOW - 实现方式:
    1. 使用try-except包装主要执行逻辑
    2. 提供开始和结束提示
    3. 调用具体示例函数
    4. 总结关键学习点
    
    WHAT - 功能作用:
    作为程序入口点，执行状态历史追踪和回放示例，
    确保示例执行的稳定性，增强用户学习体验
    """
    print("===== LangGraph 状态回放与历史追踪示例 =====\n")
    
    try:
        # 运行状态历史追踪示例
        run_history_tracking_example()
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 设计支持历史记录的状态结构")
        print("2. 使用书签标记关键状态点")
        print("3. 查看和分析状态历史记录")
        print("4. 从历史状态点创建分支对话")
        print("5. 比较不同状态路径的差异")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 