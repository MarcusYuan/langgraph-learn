#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 图结构与流程控制详解
=================================
本示例讲解LangGraph中的图结构设计与流程控制:
1. 边的定义与条件跳转
2. 条件分支实现
3. 循环与递归处理
4. 高级流程模式

学习目标:
- 理解图结构的基本组成和类型
- 掌握条件分支和路由设计
- 学习循环和递归实现方式
- 了解复杂流程控制模式
"""

# ====================================================================
# WHY: 需要导入特定的库来支持类型提示、随机功能、时间处理和LangGraph功能
# HOW: 使用Python标准库的typing、random和datetime模块，以及LangGraph的关键组件
# WHAT: 导入所需的工具和类型，为后续代码实现提供基础支持
# ====================================================================
from typing import TypedDict, List, Dict, Any, Optional, Union, Literal
import random
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ===========================================================
# 第1部分: 状态定义
# ===========================================================

# ====================================================================
# WHY: 状态设计是LangGraph图执行的核心
# 1. 使用TypedDict提供静态类型检查和代码提示，减少错误并提高开发效率
# 2. 状态作为图执行的"载体"在节点间传递，包含所有决策所需信息
# 3. 每个字段都有特定用途，支持不同的流程控制需求
# 4. 不可变性设计确保状态变更的可追踪性，有利于调试和理解执行路径
# 5. 松耦合设计使节点函数只依赖状态结构，不直接依赖其他节点
#
# HOW: 明确指定每个字段类型，支持图中的各种操作需求
# 1. messages存储对话历史，支持多种消息类型
# 2. task_type用于路由决策，影响工作流方向
# 3. iteration跟踪循环和递归，控制终止条件
# 4. results存储处理结果，支持节点间数据传递
# 5. error用于错误处理，触发条件边进入错误处理流程
#
# WHAT: 一个功能完备的状态结构，支持复杂的图流程控制
# ====================================================================
class AssistantState(TypedDict):
    """助手状态定义"""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # 对话历史
    task_type: Optional[str]  # 任务类型：搜索、计算、创意、其他
    iteration: Optional[int]  # 迭代次数
    results: Optional[Dict[str, Any]]  # 处理结果
    error: Optional[str]  # 错误信息

# ====================================================================
# WHY: 需要一个函数统一初始化状态，确保一致的初始状态
# HOW: 创建一个包含默认值的AssistantState字典
# WHAT: 返回一个预设的初始状态对象，包含系统消息、空任务类型和计数器
# ====================================================================
def initialize_state() -> AssistantState:
    """初始化助手状态"""
    return {
        "messages": [
            SystemMessage(content="你是一个智能助手，可以处理各类请求。")
        ],
        "task_type": None,
        "iteration": 0,
        "results": {},
        "error": None
    }

# ===========================================================
# 第2部分: 基本节点函数
# ===========================================================

# ====================================================================
# WHY: 需要根据用户输入确定应该执行哪种类型的任务
# HOW: 分析最新的用户消息内容，使用关键词匹配进行简单分类
# WHAT: 识别并返回用户请求的任务类型（搜索、计算、创意或一般请求）
# ====================================================================
def classify_task(state: AssistantState) -> AssistantState:
    """对用户请求进行分类
    
    函数逻辑说明:
    1. 接收当前状态对象作为输入
    2. 从状态的消息历史中提取最新的用户消息
    3. 通过关键词匹配分析用户意图，确定任务类型
    4. 返回更新了task_type字段的新状态对象
    """
    # 首先检查消息历史中是否存在用户消息
    # any函数检查是否至少有一条消息是HumanMessage类型
    # 如果没有用户消息，则将任务类型设为"greeting"(问候)
    if not any(isinstance(msg, HumanMessage) for msg in state["messages"]):
        return {**state, "task_type": "greeting"}  # 使用展开语法创建新状态对象
    
    # 从消息历史中提取最新的用户消息内容
    # 1. reversed()反转消息列表，优先查找最近的消息
    # 2. next()找到第一个满足条件的消息(是HumanMessage类型)
    # 3. 如果没找到，返回空字符串作为默认值
    last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    # 基于关键词匹配的简单分类逻辑
    # 通过检查用户消息中是否包含特定关键词来确定任务类型
    # 使用字符串的lower()方法确保匹配不区分大小写
    if any(keyword in last_user_msg.lower() for keyword in ["搜索", "查找", "找到"]):
        # 包含搜索相关关键词，标记为搜索任务
        task_type = "search"
    elif any(keyword in last_user_msg.lower() for keyword in ["计算", "多少", "等于"]):
        # 包含计算相关关键词，标记为计算任务
        task_type = "calculation"
    elif any(keyword in last_user_msg.lower() for keyword in ["创造", "生成", "写一个"]):
        # 包含创意相关关键词，标记为创意任务
        task_type = "creative"
    else:
        # 未匹配到特定关键词，默认为一般任务
        task_type = "general"
    
    # 返回更新后的状态
    # 关键点: 不修改原始状态对象，而是创建包含新task_type的副本
    # {**state, "task_type": task_type} 使用字典展开语法保留原状态的所有字段，仅更新task_type
    return {**state, "task_type": task_type}

# ====================================================================
# WHY: 需要处理搜索类型的用户请求
# HOW: 模拟搜索过程并生成搜索结果
# WHAT: 返回带有搜索结果的更新状态
# ====================================================================
def search_task(state: AssistantState) -> AssistantState:
    """处理搜索任务
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 模拟执行搜索操作，生成示例搜索结果
    3. 将结果添加到状态的results字段中
    4. 返回更新后的状态对象
    """
    # 打印执行信息，便于调试和跟踪
    # 包括当前任务类型和迭代次数
    print(f"执行搜索任务，迭代次数: {state['iteration']}")
    
    # 模拟搜索操作 - 在实际应用中，这里会调用真实的搜索API
    # 创建一个包含搜索结果和时间戳的字典
    results = {
        "search_results": [
            "搜索结果1: 这是第一条相关信息",
            "搜索结果2: 这是第二条相关信息"
        ],
        "timestamp": datetime.now().isoformat()  # 添加时间戳记录处理时间
    }
    
    # 返回更新后的状态
    # 1. 使用**state保留原状态的所有字段
    # 2. 更新results字段，合并现有结果(如果有)和新结果
    # 3. {**(state.get("results") or {}), **results} 确保即使原状态没有results也能正常工作
    return {
        **state,
        "results": {**(state.get("results") or {}), **results}
    }

# ====================================================================
# WHY: 需要处理计算类型的用户请求
# HOW: 模拟计算过程并返回结果
# WHAT: 返回带有计算结果的更新状态
# ====================================================================
def calculation_task(state: AssistantState) -> AssistantState:
    """处理计算任务
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 模拟执行计算操作，生成示例计算结果
    3. 将结果添加到状态的results字段中
    4. 返回更新后的状态对象
    """
    # 打印执行信息，用于调试和跟踪
    print(f"执行计算任务，迭代次数: {state['iteration']}")
    
    # 模拟计算操作 - 在实际应用中，这里会包含真实的计算逻辑
    # 为简化示例，这里直接返回固定值42
    results = {
        "calculation_result": 42,  # 示例计算结果
        "timestamp": datetime.now().isoformat()  # 添加时间戳
    }
    
    # 返回更新后的状态，使用字典解包语法
    # 合并现有结果和新结果，保持状态不可变性
    return {
        **state,
        "results": {**(state.get("results") or {}), **results}
    }

# ====================================================================
# WHY: 需要处理创意类型的用户请求
# HOW: 模拟创意内容生成
# WHAT: 返回带有创意内容的更新状态
# ====================================================================
def creative_task(state: AssistantState) -> AssistantState:
    """处理创意任务
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 模拟创意内容生成过程
    3. 将生成的创意内容添加到状态的results字段
    4. 返回更新后的状态对象
    """
    # 打印执行信息，跟踪当前处理的任务和迭代次数
    print(f"执行创意任务，迭代次数: {state['iteration']}")
    
    # 模拟创意生成 - 在实际应用中，这里会调用LLM生成创意内容
    # 创建包含创意结果的字典
    results = {
        "creative_result": "这是一个由AI生成的创意内容示例",  # 示例创意内容
        "timestamp": datetime.now().isoformat()  # 记录生成时间
    }
    
    # 返回更新后的状态
    # 保持原状态的结构并更新results字段
    return {
        **state,
        "results": {**(state.get("results") or {}), **results}
    }

# ====================================================================
# WHY: 需要处理一般类型的用户请求
# HOW: 提供通用回答
# WHAT: 返回带有一般回答的更新状态
# ====================================================================
def general_task(state: AssistantState) -> AssistantState:
    """处理一般任务
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 处理一般性问题并生成回答
    3. 将回答添加到状态的results字段
    4. 返回更新后的状态对象
    """
    # 打印执行信息，用于跟踪流程
    print(f"执行一般任务，迭代次数: {state['iteration']}")
    
    # 模拟一般处理 - 提供通用回答
    # 在实际应用中，这里可能会基于用户问题提供更具体的回应
    results = {
        "general_result": "这是对一般问题的回答",  # 通用回答内容
        "timestamp": datetime.now().isoformat()  # 添加处理时间戳
    }
    
    # 返回更新后的状态
    # 使用不可变更新模式，创建新状态对象
    return {
        **state,
        "results": {**(state.get("results") or {}), **results}
    }

# ====================================================================
# WHY: 需要将任务处理结果转换为用户可读的回复消息
# HOW: 根据任务类型和处理结果，生成相应的回复文本，并添加到消息历史中
# WHAT: 返回包含AI回复消息的更新状态
# ====================================================================
def generate_response(state: AssistantState) -> AssistantState:
    """生成回复
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 根据任务类型和处理结果构建合适的回复文本
    3. 将回复作为AI消息添加到消息历史
    4. 返回更新后的状态对象
    """
    # 从状态中获取任务类型和结果
    # task_type确定使用哪种类型的回复模板
    task_type = state["task_type"]
    # 获取results字典，如果不存在则使用空字典
    results = state.get("results", {})
    
    # 根据任务类型和结果生成回复文本
    # 针对不同类型的任务，从results中提取对应的结果字段
    if task_type == "search" and "search_results" in results:
        # 搜索任务：将搜索结果列表转换为文本
        response = f"搜索结果:\n" + "\n".join(results["search_results"])
    elif task_type == "calculation" and "calculation_result" in results:
        # 计算任务：展示计算结果
        response = f"计算结果: {results['calculation_result']}"
    elif task_type == "creative" and "creative_result" in results:
        # 创意任务：直接使用创意结果文本
        response = results["creative_result"]
    elif task_type == "general" and "general_result" in results:
        # 一般任务：使用一般回答
        response = results["general_result"]
    else:
        # 无法识别任务类型或找不到对应结果时的默认回复
        response = "我无法处理这个请求。"
    
    # 创建新的消息列表，保持状态不可变性
    # 1. 复制原消息列表，避免修改原状态
    # 2. 添加AI回复消息到列表末尾
    new_messages = state["messages"].copy()
    new_messages.append(AIMessage(content=response))
    
    # 返回更新后的状态，仅修改messages字段
    return {
        **state,
        "messages": new_messages
    }

# ====================================================================
# WHY: 需要追踪循环或递归的迭代次数
# HOW: 将迭代计数器递增
# WHAT: 返回迭代计数增加的更新状态
# ====================================================================
def increment_iteration(state: AssistantState) -> AssistantState:
    """增加迭代次数
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 将迭代计数器(iteration)加1
    3. 返回更新后的状态对象
    
    该函数用于循环和递归流程中，跟踪处理轮次并控制终止条件。
    """
    # 返回更新后的状态，仅修改iteration字段
    # 1. 使用**state保留原状态的所有其他字段
    # 2. 读取当前迭代计数(如果不存在则默认为0)并加1
    # 3. 保持状态不可变性，创建新对象而非修改原对象
    return {
        **state,
        "iteration": (state.get("iteration") or 0) + 1
    }

# ====================================================================
# WHY: 需要统一处理各种错误情况
# HOW: 生成错误回复消息并清除错误状态
# WHAT: 返回带有错误回复的更新状态，同时清除错误标记防止重复处理
# ====================================================================
def handle_error(state: AssistantState) -> AssistantState:
    """处理错误
    
    函数逻辑说明:
    1. 接收当前状态对象(包含错误信息)
    2. 创建错误回复消息
    3. 将错误回复添加到消息历史
    4. 清除错误标记，防止重复处理
    5. 返回更新后的状态对象
    """
    # 创建新的消息列表，保持状态不可变性
    # 复制原消息列表，避免修改原状态
    new_messages = state["messages"].copy()
    # 添加标准错误回复消息
    new_messages.append(AIMessage(content="抱歉，处理您的请求时出现错误。"))
    
    # 返回更新后的状态
    # 1. 更新messages字段，添加错误回复
    # 2. 将error字段设为None，表示错误已处理
    # 这样设计可以防止错误处理节点被重复执行
    return {
        **state,
        "messages": new_messages,
        "error": None  # 清除错误状态
    }

# ===========================================================
# 第3部分: 图结构与边的定义
# ===========================================================

print("===== 图结构与边的定义 =====")

# ====================================================================
# WHY: 需要展示最基本的线性流程图结构
# HOW: 创建节点，添加单向边，设置入口点，然后编译并执行图
# WHAT: 实现了一个简单的三节点直线流程图：分类->搜索->回复
# ====================================================================
# 3.1 基本直线图示例
def create_linear_graph():
    """创建简单的直线流程图
    
    函数逻辑说明:
    1. 创建状态图实例
    2. 添加处理节点(classify, search, respond)
    3. 定义节点间的单向连接，形成线性流程
    4. 设置入口点
    5. 编译图结构
    6. 测试图的执行流程
    
    该函数展示了LangGraph最基础的线性工作流结构，
    其中节点按固定顺序依次执行，没有分支或循环。
    """
    # 打印说明，表明正在创建简单线性图
    print("\n创建简单的直线流程图:")
    
    # 创建图实例，指定状态类型为AssistantState
    # StateGraph是LangGraph的核心类，用于定义工作流
    workflow = StateGraph(AssistantState)
    
    # 添加节点 - 每个节点对应一个函数
    # add_node方法将函数与节点名关联
    # 每个节点函数接收状态并返回更新后的状态
    workflow.add_node("classify", classify_task)  # 分类节点 - 确定任务类型
    workflow.add_node("search", search_task)      # 搜索节点 - 处理搜索请求
    workflow.add_node("respond", generate_response)  # 回复节点 - 生成回复消息
    
    # 添加基本边 - 定义节点间的连接关系
    # add_edge方法创建从源节点到目标节点的单向连接
    # 这些边决定了状态对象在图中的流动路径
    workflow.add_edge("classify", "search")  # 从分类到搜索 - 第一步到第二步
    workflow.add_edge("search", "respond")   # 从搜索到回复 - 第二步到第三步
    workflow.add_edge("respond", END)        # 从回复到结束 - 工作流终止标记
    
    # 设置入口点 - 指定工作流的起始节点
    # set_entry_point方法定义状态对象首先进入哪个节点
    workflow.set_entry_point("classify")
    
    # 编译图 - 将定义好的工作流转换为可执行图
    # compile方法返回一个可以接收状态并执行的图对象
    graph = workflow.compile()
    
    # 打印图结构说明，用ASCII字符展示节点间连接
    print("图结构: classify -> search -> respond -> END")
    
    # 测试执行 - 使用示例输入验证图的功能
    # 1. 初始化状态
    state = initialize_state()
    # 2. 添加用户消息
    state["messages"].append(HumanMessage(content="请帮我搜索关于Python的信息"))
    
    # 3. 调用图执行工作流
    # invoke方法接收初始状态，返回最终状态
    final_state = graph.invoke(state)
    # 4. 打印执行结果
    print(f"最终回复: {final_state['messages'][-1].content}")
    
    # 返回编译后的图，便于后续使用
    return graph

# 创建并执行直线图
linear_graph = create_linear_graph()

# ===========================================================
# 第4部分: 条件分支与路由
# ===========================================================

print("\n===== 条件分支与路由 =====")

# ====================================================================
# WHY: 需要根据状态中的任务类型动态决定下一个节点
# HOW: 检查状态中的task_type字段，根据不同值返回对应的节点名称
# WHAT: 实现了任务类型与处理节点之间的映射关系
# ====================================================================
# 4.1 条件路由函数
def route_by_task_type(state: AssistantState) -> str:
    """根据任务类型进行路由，返回下一个节点名称
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 分析状态中的task_type字段
    3. 根据任务类型返回对应的目标节点名称
    
    这是条件路由的核心函数，它决定工作流执行的下一步走向。
    在LangGraph中，条件边使用的路由函数必须返回节点名称而非状态对象。
    """
    # 打印路由决策信息，便于调试和跟踪执行流程
    print(f"路由决策: 任务类型 = {state['task_type']}")
    
    # 获取任务类型，如果不存在则默认为"general"
    # 这确保即使状态中没有task_type字段也能正常路由
    task_type = state.get("task_type", "general")
    
    # 根据任务类型返回对应的节点名称
    # 每个任务类型对应一个专门的处理节点
    if task_type == "search":
        return "search_task"  # 搜索任务处理节点
    elif task_type == "calculation":
        return "calculation_task"  # 计算任务处理节点
    elif task_type == "creative":
        return "creative_task"  # 创意任务处理节点
    else:
        return "general_task"  # 默认的一般任务处理节点

# ====================================================================
# WHY: 需要展示如何使用条件路由函数来创建动态分支
# HOW: 使用add_conditional_edges方法连接源节点和多个可能的目标节点
# WHAT: 实现了一个根据任务类型动态选择处理路径的图结构
# ====================================================================
# 4.2 使用条件路由创建分支图
def create_conditional_graph():
    """创建条件分支流程图
    
    函数逻辑说明:
    1. 创建状态图实例
    2. 添加多种类型的任务处理节点
    3. 使用条件路由函数连接分类节点与各处理节点
    4. 将所有处理结果汇总到回复节点
    5. 编译图并测试不同分支的执行效果
    
    该函数展示了如何使用条件边实现多分支流程，
    根据任务类型动态决定执行路径，是LangGraph中
    实现决策逻辑的核心模式。
    """
    # 打印创建信息
    print("\n创建条件分支流程图:")
    
    # 创建图实例 - 指定状态类型
    workflow = StateGraph(AssistantState)
    
    # 添加各类节点 - 构建完整处理流程
    # 1. 任务分类节点
    workflow.add_node("classify", classify_task)          # 分类节点 - 确定任务类型
    # 2. 各类任务专用处理节点
    workflow.add_node("search_task", search_task)         # 搜索处理节点
    workflow.add_node("calculation_task", calculation_task)  # 计算处理节点
    workflow.add_node("creative_task", creative_task)     # 创意处理节点
    workflow.add_node("general_task", general_task)       # 一般处理节点
    # 3. 回复生成节点
    workflow.add_node("respond", generate_response)       # 回复生成节点
    
    # 添加条件分支 - 核心部分
    # add_conditional_edges方法接收:
    # 1. 源节点名称
    # 2. 路由函数 - 根据状态返回目标节点名称
    # 3. 可能的目标节点映射 - 将路由函数返回值映射到实际节点
    workflow.add_conditional_edges(
        "classify",  # 源节点
        route_by_task_type,  # 路由函数 - 决定下一步去哪个节点
        {  # 目标节点映射表 - 键为路由函数可能的返回值，值为对应节点
            "search_task": "search_task",
            "calculation_task": "calculation_task",
            "creative_task": "creative_task",
            "general_task": "general_task"
        }
    )
    
    # 从各类任务节点连接到响应节点
    # 所有分支最终汇聚到同一个响应生成节点
    workflow.add_edge("search_task", "respond")      # 搜索结果→回复
    workflow.add_edge("calculation_task", "respond") # 计算结果→回复
    workflow.add_edge("creative_task", "respond")    # 创意结果→回复
    workflow.add_edge("general_task", "respond")     # 一般结果→回复
    workflow.add_edge("respond", END)                # 回复→结束
    
    # 设置入口点 - 工作流从分类节点开始
    workflow.set_entry_point("classify")
    
    # 编译图 - 转换为可执行图结构
    graph = workflow.compile()
    
    # 打印图结构 - 使用ASCII图形展示节点连接关系
    # 该图显示了从classify到各个处理节点的条件分支，
    # 以及所有处理节点到respond的汇聚
    print("图结构:")
    print("                  ┌→ search_task ┐")
    print("                  ├→ calculation_task ┤")
    print("classify ┬→┬→┬→┬→ ┼→ creative_task ┼→ respond → END")
    print("                  └→ general_task ┘")
    
    # 测试条件分支执行 - 验证不同输入的路由效果
    print("\n测试条件分支执行:")
    
    # 测试搜索分支 - 包含搜索关键词的请求
    search_state = initialize_state()
    search_state["messages"].append(HumanMessage(content="请帮我搜索关于Python的信息"))
    search_result = graph.invoke(search_state)
    print(f"搜索分支回复: {search_result['messages'][-1].content}")
    
    # 测试计算分支 - 包含计算关键词的请求
    calc_state = initialize_state()
    calc_state["messages"].append(HumanMessage(content="计算 2 + 2 等于多少"))
    calc_result = graph.invoke(calc_state)
    print(f"计算分支回复: {calc_result['messages'][-1].content}")
    
    # 返回编译后的图对象
    return graph

# 创建并执行条件分支图
conditional_graph = create_conditional_graph()

# ====================================================================
# WHY: 需要展示如何使用Lambda函数作为条件判断，实现更灵活的条件边
# HOW: 使用add_edge方法的condition参数设置条件函数
# WHAT: 实现了一个根据错误状态决定执行路径的简单分支结构
# ====================================================================
# 4.3 使用Lambda函数作为条件边
def create_lambda_conditional_graph():
    """使用Lambda函数创建条件边"""
    print("\n创建使用Lambda条件的图:")
    
    # 创建图实例
    workflow = StateGraph(AssistantState)
    
    # 添加节点
    workflow.add_node("process", general_task)      # 处理节点
    workflow.add_node("error_handler", handle_error)  # 错误处理节点
    workflow.add_node("respond", generate_response)   # 回复生成节点
    
    # 添加从process到error_handler的条件边 - 仅当有错误时
    workflow.add_edge(
        "process", 
        "error_handler",
        condition=lambda state: state.get("error") is not None  # 条件：存在错误
    )
    
    # 添加从process到respond的条件边 - 当没有错误时
    workflow.add_edge(
        "process", 
        "respond",
        condition=lambda state: state.get("error") is None  # 条件：无错误
    )
    
    workflow.add_edge("error_handler", END)
    workflow.add_edge("respond", END)
    
    # 设置入口点
    workflow.set_entry_point("process")
    
    # 编译图
    graph = workflow.compile()
    
    print("图结构:")
    print("            ┌→ error_handler → END")
    print("process ┬→┬→┤")
    print("            └→ respond → END")
    
    print("\n(实际执行时只会走一条路径，取决于是否有错误)")
    
    return graph

# 创建Lambda条件图
lambda_graph = create_lambda_conditional_graph()

# ===========================================================
# 第5部分: 循环与递归处理
# ===========================================================

print("\n===== 循环与递归处理 =====")

# ====================================================================
# WHY: 需要一个决策函数来控制循环的终止条件
# HOW: 检查迭代计数，并与最大迭代次数比较来决定是继续循环还是结束
# WHAT: 实现了循环控制逻辑，让图结构可以重复执行特定节点直到满足条件
# ====================================================================
# 5.1 基于条件的循环
def should_continue_iteration(state: AssistantState) -> str:
    """决定是否继续迭代
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 检查迭代计数并与最大允许迭代次数比较
    3. 根据比较结果返回继续迭代或结束迭代的决策
    
    该函数是循环控制的核心，通过返回不同的路由值决定
    是继续执行循环体还是退出循环。在LangGraph中，
    这种模式用于实现有限次数的重复处理。
    """
    # 获取当前迭代次数，默认为0
    iteration = state.get("iteration", 0)
    # 设置最大迭代次数阈值
    max_iterations = 3  # 最多允许3次迭代
    
    # 打印当前迭代状态，便于调试
    print(f"循环判断: 当前迭代次数 = {iteration}, 最大次数 = {max_iterations}")
    
    # 判断是否达到最大迭代次数
    # 1. 如果达到或超过最大次数，返回结束标记
    # 2. 否则返回继续标记
    if iteration >= max_iterations:
        print("迭代完成，进入结束节点")
        return "end_iteration"  # 结束迭代，进入后续处理
    else:
        print("继续迭代")
        return "continue_iteration"  # 继续迭代，回到处理节点

# ====================================================================
# WHY: 需要展示如何实现一个固定次数的循环处理
# HOW: 使用条件边将节点连接成循环，并通过计数控制退出循环
# WHAT: 实现了一个执行固定次数迭代的处理流程图
# ====================================================================
def create_loop_graph():
    """创建循环流程图
    
    函数逻辑说明:
    1. 创建状态图实例
    2. 添加处理节点、计数节点和响应节点
    3. 使用条件边实现循环结构
    4. 设置入口点并编译图
    5. 测试循环执行过程
    
    该函数展示了如何在LangGraph中实现循环结构，
    通过条件判断和回环边使节点可以被多次执行，
    直到满足特定条件为止。
    """
    # 打印创建信息
    print("\n创建循环流程图:")
    
    # 创建图实例
    workflow = StateGraph(AssistantState)
    
    # 添加节点
    # 1. 开始节点 - 简单传递状态的身份函数
    workflow.add_node("start", lambda state: state)  # 简单的开始节点，不修改状态
    # 2. 处理节点 - 每次循环执行的主要逻辑
    workflow.add_node("process", general_task)       # 处理节点 - 模拟任务处理
    # 3. 计数节点 - 增加迭代计数器
    workflow.add_node("increment", increment_iteration)  # 增加迭代计数，跟踪循环次数
    # 4. 响应节点 - 循环结束后生成最终回复
    workflow.add_node("respond", generate_response)  # 生成响应 - 循环结束后执行
    
    # 添加基本边 - 定义初始流程
    workflow.add_edge("start", "process")         # 从开始到处理
    workflow.add_edge("process", "increment")     # 从处理到计数增加
    
    # 添加条件判断 - 决定是循环还是结束
    # 使用add_conditional_edges实现基于条件的分支
    # should_continue_iteration函数根据迭代次数返回决策
    workflow.add_conditional_edges(
        "increment",  # 源节点 - 迭代计数器
        should_continue_iteration,  # 条件函数 - 决定是否继续循环
        {
            "continue_iteration": "process",  # 继续处理 - 回到process形成循环
            "end_iteration": "respond"        # 结束并回复 - 退出循环
        }
    )
    
    # 添加结束边 - 从响应到终止
    workflow.add_edge("respond", END)  # 从响应节点到结束
    
    # 设置入口点 - 工作流从start节点开始
    workflow.set_entry_point("start")
    
    # 编译图 - 转换为可执行结构
    graph = workflow.compile()
    
    # 打印循环图结构 - 使用ASCII图形展示
    # 该图显示了process→increment→process的循环路径
    # 以及从increment到respond的退出路径
    print("循环图结构:")
    print("              ┌───────────────┐")
    print("              │               │")
    print("              ▼               │")
    print("start → process → increment ──┘")
    print("                    │")
    print("                    ▼")
    print("                  respond → END")
    
    # 测试执行 - 验证循环功能
    state = initialize_state()
    state["messages"].append(HumanMessage(content="执行循环测试"))
    
    # 调用图执行 - 状态将在process和increment之间循环
    # 直到达到最大迭代次数
    final_state = graph.invoke(state)
    
    # 打印执行结果 - 验证迭代次数和最终回复
    print(f"循环执行后的最终状态: 迭代次数 = {final_state['iteration']}")
    print(f"最终回复: {final_state['messages'][-1].content}")
    
    # 返回编译后的图对象
    return graph

# 创建并执行循环图
loop_graph = create_loop_graph()

# ====================================================================
# WHY: 需要评估创意结果质量，决定是否需要继续改进
# HOW: 根据迭代次数模拟质量评分，并基于评分返回决策
# WHAT: 提供了一个创意内容质量评估函数，作为递归决策的依据
# ====================================================================
# 5.2 递归处理 - 创意生成循环改进
def evaluate_creative_result(state: AssistantState) -> str:
    """评估创意结果质量，决定是否需要进一步改进
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 从状态中提取迭代次数和当前创意结果
    3. 根据迭代次数模拟质量评分
    4. 基于质量分数和迭代次数决定是继续改进还是完成
    
    这是递归处理模式的核心决策函数，用于判断创意内容
    是否达到质量标准，或是否需要继续改进。在实际应用中，
    可能会使用更复杂的评估逻辑。
    """
    # 从状态中提取迭代次数和创意结果
    iteration = state.get("iteration", 0)
    result = state.get("results", {}).get("creative_result", "")
    
    # 模拟评估过程 - 基于迭代次数计算质量分数
    # 这是一个简化的评估模型:
    # 1. 初始质量为0.5
    # 2. 每次迭代提升0.2分
    # 3. 最高分为0.9分
    quality_score = min(0.5 + iteration * 0.2, 0.9)  # 每次迭代提高质量
    
    # 打印评估信息，便于调试
    print(f"评估创意质量: 迭代={iteration}, 质量分数={quality_score:.2f}")
    
    # 做出决策 - 基于质量分数和最大迭代限制
    # 1. 如果质量分数低于0.8且迭代次数小于3，继续改进
    # 2. 否则完成创作过程
    if quality_score < 0.8 and iteration < 3:
        return "improve"  # 返回改进决策，进入改进节点
    else:
        return "complete"  # 返回完成决策，进入回复节点

# ====================================================================
# WHY: 需要一个函数来不断改进创意内容
# HOW: 获取当前创意结果，添加改进内容，并更新迭代计数
# WHAT: 实现了创意内容的渐进式改进功能
# ====================================================================
def improve_creative_result(state: AssistantState) -> AssistantState:
    """改进创意结果
    
    函数逻辑说明:
    1. 接收当前状态对象
    2. 获取当前创意结果和迭代次数
    3. 基于当前结果生成改进版本
    4. 更新状态中的创意结果和迭代计数
    5. 返回更新后的状态对象
    
    这是递归处理的核心执行函数，负责在每次迭代中
    增强和改进创意内容。在实际应用中，可能会调用
    LLM生成更高质量的内容。
    """
    # 从状态中获取当前结果和迭代次数
    # get方法的第二个参数提供默认值，确保即使字段不存在也能正常工作
    current_result = state.get("results", {}).get("creative_result", "这是初始创意")
    iteration = state.get("iteration", 0)
    
    # 模拟改进过程 - 在现有结果基础上添加改进内容
    # 在实际应用中，这里通常会调用LLM来生成更好的内容
    # 而不是简单地添加固定文本
    improved_result = f"{current_result} [第{iteration+1}次改进: 增加了更多细节和创意元素]"
    
    # 打印改进信息，便于调试
    print(f"改进创意: 迭代={iteration+1}")
    
    # 返回更新后的状态
    # 1. 使用**state保留原状态的所有字段
    # 2. 更新results字段，保留其他结果并更新creative_result
    # 3. 增加迭代计数器
    return {
        **state,
        "results": {
            **(state.get("results") or {}),  # 保留现有结果
            "creative_result": improved_result  # 更新创意结果
        },
        "iteration": iteration + 1  # 增加迭代次数
    }

# ====================================================================
# WHY: 需要展示如何实现递归处理模式
# HOW: 使用条件边和回环构建一个可以重复改进的流程
# WHAT: 实现了一个创意内容生成和质量改进的递归流程图
# ====================================================================
def create_recursive_graph():
    """创建递归处理流程图
    
    函数逻辑说明:
    1. 创建状态图实例
    2. 添加创意生成、评估和改进节点
    3. 构建递归改进循环
    4. 设置入口点并编译图
    5. 测试递归执行过程
    
    该函数展示了如何使用LangGraph实现递归处理模式，
    特别适用于需要多次迭代改进直到满足质量标准的场景，
    如创意内容生成、文本改进等。
    """
    # 打印创建信息
    print("\n创建递归处理流程图:")
    
    # 创建图实例
    workflow = StateGraph(AssistantState)
    
    # 添加节点
    # 1. 分类节点 - 确定任务类型
    workflow.add_node("classify", classify_task)  # 分类节点
    # 2. 创意生成节点 - 初始创意内容
    workflow.add_node("create", creative_task)    # 创意生成节点
    # 3. 评估节点 - 判断质量
    workflow.add_node("evaluate", lambda state: state)  # 评估节点 - 不修改状态，仅用于路由
    # 4. 改进节点 - 优化创意内容
    workflow.add_node("improve", improve_creative_result)  # 改进节点
    # 5. 回复节点 - 生成最终回复
    workflow.add_node("respond", generate_response)  # 回复节点
    
    # 添加基本边 - 初始流程
    workflow.add_edge("classify", "create")  # 从分类到创建 - 生成初始内容
    workflow.add_edge("create", "evaluate")  # 从创建到评估 - 评估初始质量
    
    # 添加条件边 - 评估后决策
    # 使用evaluate_creative_result函数判断质量
    # 并决定是继续改进还是完成创作
    workflow.add_conditional_edges(
        "evaluate",  # 源节点 - 评估节点
        evaluate_creative_result,  # 条件函数 - 评估质量并返回决策
        {
            "improve": "improve",    # 需要改进 - 进入改进节点
            "complete": "respond"    # 完成创作 - 进入回复节点
        }
    )
    
    # 添加改进后的回环 - 形成递归结构
    # 改进节点执行后回到评估节点，形成循环
    # 这是递归模式的核心，使内容能被多次改进
    workflow.add_edge("improve", "evaluate")  # 从改进回到评估，形成循环
    # 添加结束边
    workflow.add_edge("respond", END)  # 回复后结束工作流
    
    # 设置入口点 - 工作流从分类节点开始
    workflow.set_entry_point("classify")
    
    # 编译图 - 转换为可执行结构
    graph = workflow.compile()
    
    # 打印递归图结构 - 使用ASCII图形展示
    # 该图显示了evaluate→improve→evaluate的递归循环
    # 以及从evaluate到respond的退出路径
    print("递归图结构:")
    print("              ┌───────────────┐")
    print("              │               │")
    print("classify → create → evaluate ←─┐")
    print("                    │         │")
    print("                    ▼         │")
    print("                  respond     │")
    print("                    │         │")
    print("                    ▼         │")
    print("                   END    improve")
    print("                            │")
    print("                            └───┘")
    
    # 测试执行 - 验证递归功能
    state = initialize_state()
    state["messages"].append(HumanMessage(content="帮我创作一首诗"))
    state["task_type"] = "creative"  # 设置任务类型以简化示例
    
    # 执行图 - 将触发创意生成、评估和可能的多次改进
    final_state = graph.invoke(state)
    
    # 打印执行结果
    print(f"递归执行后的最终状态: 迭代次数 = {final_state['iteration']}")
    print(f"最终创意结果: {final_state.get('results', {}).get('creative_result', 'N/A')}")
    
    # 返回编译后的图对象
    return graph

# 创建并执行递归图
recursive_graph = create_recursive_graph()

# ===========================================================
# 第6部分: 复杂流程控制模式
# ===========================================================

print("\n===== 复杂流程控制模式 =====")

# ====================================================================
# WHY: 需要展示如何组合多个图形成更复杂的工作流
# HOW: 创建子图并在主图中作为节点使用
# WHAT: 实现了一个带有子图组件的复合图结构，展示模块化设计
# ====================================================================
# 6.1 子图和组合模式
def create_combined_graph():
    """创建组合图示例"""
    print("\n创建组合图:")
    
    # 创建主图
    main_workflow = StateGraph(AssistantState)
    
    # 创建子图 - 处理搜索
    search_workflow = StateGraph(AssistantState)
    search_workflow.add_node("search", search_task)
    search_workflow.add_node("process_results", lambda state: {
        **state,
        "results": {
            **(state.get("results") or {}),
            "processed": True
        }
    })
    search_workflow.add_edge("search", "process_results")
    
    # 设置子图入口和出口
    search_workflow.set_entry_point("search")
    search_workflow.set_finish_point("process_results")
    
    # 编译子图
    search_subgraph = search_workflow.compile()
    
    # 在主图中添加编译后的子图作为节点
    main_workflow.add_node("classify", classify_task)
    main_workflow.add_node("search_flow", search_subgraph)  # 子图作为节点
    main_workflow.add_node("respond", generate_response)
    
    # 连接主图节点
    main_workflow.add_edge("classify", "search_flow")
    main_workflow.add_edge("search_flow", "respond")
    main_workflow.add_edge("respond", END)
    
    # 设置主图入口
    main_workflow.set_entry_point("classify")
    
    # 编译主图
    main_graph = main_workflow.compile()
    
    print("组合图结构:")
    print("classify → [search → process_results] → respond → END")
    print("            └─── search_flow ────┘")
    
    # 测试执行
    state = initialize_state()
    state["messages"].append(HumanMessage(content="请帮我搜索关于Python的信息"))
    
    final_state = main_graph.invoke(state)
    print(f"组合图执行结果: {final_state['messages'][-1].content}")
    print(f"处理标记: {final_state.get('results', {}).get('processed', False)}")
    
    return main_graph

# 创建组合图
combined_graph = create_combined_graph()

# ====================================================================
# WHY: 需要实现多条件复杂路由决策
# HOW: 综合考虑任务类型、迭代次数和错误状态多个条件
# WHAT: 实现了一个复杂的路由决策函数，模拟状态机转换逻辑
# ====================================================================
# 6.2 复杂条件路由 - 状态机模式
def complex_router(state: AssistantState) -> str:
    """复杂路由器 - 结合多个条件"""
    task_type = state.get("task_type", "")
    iteration = state.get("iteration", 0)
    has_error = state.get("error") is not None
    
    print(f"复杂路由: 任务={task_type}, 迭代={iteration}, 错误={has_error}")
    
    # 错误处理优先
    if has_error:
        return "error_handler"
        
    # 基于任务类型和迭代次数的路由
    if task_type == "search":
        return "search_task"
    elif task_type == "calculation":
        return "calculation_task"
    elif task_type == "creative":
        # 创意任务需要根据迭代次数决定处理方式
        if iteration < 2:
            return "basic_creative"
        else:
            return "advanced_creative"
    else:
        return "general_task"

# ====================================================================
# WHY: 需要展示如何构建复杂的条件路由系统
# HOW: 使用复杂的路由函数和多种处理节点
# WHAT: 实现了一个类似状态机的复杂流程图，根据多维条件选择路径
# ====================================================================
def create_complex_routing_graph():
    """创建复杂路由图"""
    print("\n创建复杂路由图:")
    
    # 创建图实例
    workflow = StateGraph(AssistantState)
    
    # 添加节点
    workflow.add_node("start", classify_task)
    workflow.add_node("search_task", search_task)
    workflow.add_node("calculation_task", calculation_task)
    workflow.add_node("basic_creative", creative_task)
    workflow.add_node("advanced_creative", lambda state: {
        **state,
        "results": {
            **(state.get("results") or {}),
            "creative_result": "这是高级创意生成结果，增加了更多复杂元素"
        }
    })
    workflow.add_node("general_task", general_task)
    workflow.add_node("error_handler", handle_error)
    workflow.add_node("respond", generate_response)
    
    # 添加复杂路由
    workflow.add_conditional_edges(
        "start",
        complex_router,
        {
            "search_task": "search_task",
            "calculation_task": "calculation_task",
            "basic_creative": "basic_creative",
            "advanced_creative": "advanced_creative",
            "general_task": "general_task",
            "error_handler": "error_handler"
        }
    )
    
    # 连接到响应节点
    for node in ["search_task", "calculation_task", "basic_creative", 
                 "advanced_creative", "general_task", "error_handler"]:
        workflow.add_edge(node, "respond")
    
    workflow.add_edge("respond", END)
    
    # 设置入口点
    workflow.set_entry_point("start")
    
    # 编译图
    graph = workflow.compile()
    
    print("复杂路由图结构:")
    print("              ┌→ search_task ───────┐")
    print("              ├→ calculation_task ──┤")
    print("              ├→ basic_creative ────┤")
    print("start ┬→┬→┬→┬→┼→ advanced_creative ─┼→ respond → END")
    print("              ├→ general_task ──────┤")
    print("              └→ error_handler ─────┘")
    
    # 测试执行 - 普通创意任务
    basic_state = initialize_state()
    basic_state["messages"].append(HumanMessage(content="帮我创作一首诗"))
    basic_state["task_type"] = "creative"
    basic_state["iteration"] = 1
    
    basic_result = graph.invoke(basic_state)
    print(f"基础创意任务结果: {basic_result['messages'][-1].content}")
    
    # 测试执行 - 高级创意任务
    advanced_state = initialize_state()
    advanced_state["messages"].append(HumanMessage(content="帮我创作一首诗"))
    advanced_state["task_type"] = "creative"
    advanced_state["iteration"] = 3
    
    advanced_result = graph.invoke(advanced_state)
    print(f"高级创意任务结果: {advanced_result['messages'][-1].content}")
    
    return graph

# 创建复杂路由图
complex_routing_graph = create_complex_routing_graph()

# ===========================================================
# 第7部分: 总结
# ===========================================================

# ====================================================================
# WHY: 需要总结本示例展示的主要概念和模式
# HOW: 列出关键点并解释其适用场景
# WHAT: 提供了一个对LangGraph流程控制能力的全面概述
# ====================================================================
print("\n===== 总结 =====")
print("1. 图结构基础: 节点和边是构建工作流的基本元素")
print("2. 条件分支: 使用路由函数和条件边实现分支逻辑")
print("3. 循环与递归: 通过回环边实现重复处理")
print("4. 复杂流程: 子图组合和复杂路由实现高级工作流")

print("\nLangGraph提供了灵活的工具来构建从简单到复杂的工作流程:")
print("- 简单的线性流程适合步骤明确的任务")
print("- 条件分支适合需要根据状态做出决策的场景")
print("- 循环和递归适合需要多次迭代或渐进式改进的任务")
print("- 复杂模式适合构建大型、模块化的系统") 