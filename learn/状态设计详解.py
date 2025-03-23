#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 状态设计详解示例
==================================
本示例展示LangGraph中的状态管理机制:
1. 使用TypedDict设计状态结构
2. 状态传递与更新机制
3. 不可变状态的概念与实践
4. 高级状态设计模式

本例使用一个简单的知识问答系统展示不同的状态设计方法。

学习目标:
- 理解为什么状态设计在LangGraph中如此重要
- 掌握如何使用TypedDict设计强类型状态
- 学习如何正确地更新状态
- 了解不同状态设计模式的适用场景
"""

import os
from typing import TypedDict, List, Dict, Any, Optional, Union, Literal
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

# ===========================================================
# 第1部分: 基本状态设计 - 使用TypedDict
# ===========================================================

# 1.1 简单状态设计
class SimpleState(TypedDict):
    """最简单的状态设计，仅包含消息历史"""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

# 1.2 中等复杂度状态设计
class StandardState(TypedDict):
    """标准状态设计，包含消息历史和元数据"""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    metadata: Dict[str, Any]  # 会话元数据

# 1.3 复杂状态设计
class AdvancedState(TypedDict):
    """高级状态设计，包含多个状态组件"""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # 对话历史
    context: Dict[str, Any]  # 上下文信息
    tools_results: Dict[str, Any]  # 工具调用结果
    memory: List[Dict[str, Any]]  # 长期记忆存储
    current_step: str  # 当前执行步骤
    error: Optional[str]  # 错误信息

print("状态设计中，不同复杂度的状态定义示例:")
print(f"1. 简单状态: {SimpleState.__annotations__}")
print(f"2. 标准状态: {StandardState.__annotations__}")
print(f"3. 高级状态: {AdvancedState.__annotations__}")
print("\n" + "="*50 + "\n")

# ===========================================================
# 第2部分: 状态初始化与构建
# ===========================================================

def initialize_simple_state() -> SimpleState:
    """初始化简单状态"""
    return {
        "messages": [
            SystemMessage(content="你是一个知识问答助手，可以回答用户的问题。")
        ]
    }

def initialize_standard_state() -> StandardState:
    """初始化标准状态"""
    return {
        "messages": [
            SystemMessage(content="你是一个知识问答助手，可以回答用户的问题。")
        ],
        "metadata": {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "user_id": "anonymous",
            "session_start": datetime.now().isoformat(),
            "query_count": 0
        }
    }

def initialize_advanced_state() -> AdvancedState:
    """初始化高级状态"""
    return {
        "messages": [
            SystemMessage(content="你是一个知识问答助手，可以回答用户的问题。")
        ],
        "context": {},
        "tools_results": {},
        "memory": [],
        "current_step": "start",
        "error": None
    }

# 状态初始化示例
simple_state = initialize_simple_state()
standard_state = initialize_standard_state()
advanced_state = initialize_advanced_state()

print("状态初始化示例:")
print(f"1. 简单状态: {simple_state}")
print(f"2. 标准状态: {standard_state}")
print(f"3. 高级状态: {advanced_state}")
print("\n" + "="*50 + "\n")

# ===========================================================
# 第3部分: 状态更新机制 - 保持不可变性
# ===========================================================

def wrong_update_state(state: SimpleState, user_message: str) -> SimpleState:
    """错误的状态更新方式 - 直接修改原状态（不推荐）"""
    # 错误: 直接修改原始状态对象
    state["messages"].append(HumanMessage(content=user_message))
    return state

def correct_update_state(state: SimpleState, user_message: str) -> SimpleState:
    """正确的状态更新方式 - 创建新状态"""
    # 正确: 创建新的状态对象
    new_messages = state["messages"].copy()
    new_messages.append(HumanMessage(content=user_message))
    
    return {
        "messages": new_messages
    }

# 状态更新示例
original_state = initialize_simple_state()
print("原始状态:", original_state)

# 错误的更新方式
wrong_updated = wrong_update_state(original_state, "什么是LangGraph?")
print("错误更新后的状态:", wrong_updated)
print("原始状态 (被修改了):", original_state)
print("状态ID - 原始:", id(original_state), "错误更新后:", id(wrong_updated))
print("消息列表ID - 原始:", id(original_state["messages"]), "错误更新后:", id(wrong_updated["messages"]))
print("结论: 原始状态被直接修改，消息列表是同一个对象")

# 为了演示，重新初始化
original_state = initialize_simple_state()
print("\n重新初始化原始状态:", original_state)

# 正确的更新方式
correct_updated = correct_update_state(original_state, "什么是LangGraph?")
print("正确更新后的状态:", correct_updated)
print("原始状态 (未被修改):", original_state)
print("状态ID - 原始:", id(original_state), "正确更新后:", id(correct_updated))
print("消息列表ID - 原始:", id(original_state["messages"]), "正确更新后:", id(correct_updated["messages"]))
print("结论: 创建了新的状态对象，原始状态保持不变")

print("\n" + "="*50 + "\n")

# ===========================================================
# 第4部分: 常见状态更新模式
# ===========================================================

# 4.1 创建一个通用的状态更新函数
def update_state(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """通用状态更新函数 - 创建状态的深拷贝并应用更新"""
    # 创建一个新的状态字典
    new_state = {}
    
    # 复制原始状态的所有键值对
    for key, value in state.items():
        if isinstance(value, list):
            new_state[key] = value.copy()
        elif isinstance(value, dict):
            new_state[key] = value.copy()
        else:
            new_state[key] = value
    
    # 应用更新
    for key, value in kwargs.items():
        if key in new_state and isinstance(new_state[key], list) and isinstance(value, list):
            # 对于列表，可以选择替换或扩展
            new_state[key].extend(value)
        elif key in new_state and isinstance(new_state[key], dict) and isinstance(value, dict):
            # 对于字典，可以选择合并
            new_state[key].update(value)
        else:
            # 对于其他类型，直接替换
            new_state[key] = value
    
    return new_state

# 4.2 更新特定状态字段的实用函数
def add_message(state: Dict[str, Any], message: Union[HumanMessage, AIMessage, SystemMessage]) -> Dict[str, Any]:
    """添加消息到状态"""
    new_messages = state["messages"].copy()
    new_messages.append(message)
    return update_state(state, messages=new_messages)

def increment_query_count(state: StandardState) -> StandardState:
    """增加查询计数器"""
    new_metadata = state["metadata"].copy()
    new_metadata["query_count"] = new_metadata.get("query_count", 0) + 1
    return update_state(state, metadata=new_metadata)

def set_error(state: AdvancedState, error_message: str) -> AdvancedState:
    """设置错误信息"""
    return update_state(state, error=error_message, current_step="error")

def clear_error(state: AdvancedState) -> AdvancedState:
    """清除错误信息"""
    return update_state(state, error=None)

# 状态更新模式示例
print("状态更新模式示例:")

test_state = initialize_standard_state()
print("初始状态:", test_state)

# 添加用户消息
test_state = add_message(test_state, HumanMessage(content="如何使用LangGraph?"))
print("添加消息后:", test_state)

# 增加查询计数
test_state = increment_query_count(test_state)
print("增加查询计数后:", test_state)

# 高级状态示例 - 设置错误信息
advanced_test = initialize_advanced_state()
print("\n高级状态初始:", advanced_test)

# 设置错误
advanced_test = set_error(advanced_test, "工具调用失败")
print("设置错误后:", advanced_test)

# 清除错误
advanced_test = clear_error(advanced_test)
print("清除错误后:", advanced_test)

print("\n" + "="*50 + "\n")

# ===========================================================
# 第5部分: 将状态与LangGraph结合
# ===========================================================

# 5.1 创建LLM实例
try:
    llm = OllamaLLM(
        base_url="http://localhost:11434",
        model="llama3",
        temperature=0.7,
        request_timeout=30.0,
    )
except:
    # 如果无法连接到Ollama，使用假的LLM响应
    print("无法连接到Ollama服务，将使用模拟的LLM响应进行演示")
    class MockLLM:
        def invoke(self, messages, **kwargs):
            return "这是一个模拟的LLM回复，用于演示状态设计。实际应用中，这里会返回真实的LLM生成内容。"
    llm = MockLLM()

# 5.2 定义节点函数 - 处理不同状态
def process_user_input(state: StandardState) -> StandardState:
    """处理用户输入的节点"""
    print("【状态更新】: process_user_input 节点")
    print("状态内容 (处理前):", state)
    
    # 增加查询计数
    new_state = increment_query_count(state)
    
    print("状态内容 (处理后):", new_state)
    return new_state

def generate_response(state: StandardState, config: RunnableConfig) -> StandardState:
    """生成回复的节点"""
    print("【状态更新】: generate_response 节点")
    print("状态内容 (处理前):", state)
    
    # 使用LLM根据消息历史生成回复
    response = llm.invoke(state["messages"])
    
    # 添加AI回复到消息历史
    new_state = add_message(state, AIMessage(content=response))
    
    # 添加时间戳到元数据
    new_metadata = new_state["metadata"].copy()
    new_metadata["last_response_time"] = datetime.now().isoformat()
    new_state = update_state(new_state, metadata=new_metadata)
    
    print("状态内容 (处理后):", new_state)
    return new_state

def should_continue(state: StandardState) -> Literal["continue", "end"]:
    """决定是否继续对话的路由函数"""
    query_count = state["metadata"].get("query_count", 0)
    
    # 简单示例：当查询次数达到3次时结束
    if query_count >= 3:
        print("【状态路由】: 查询次数已达到3次，结束对话")
        return "end"
    
    print(f"【状态路由】: 查询次数为{query_count}，继续对话")
    return "continue"

# 5.3 构建LangGraph工作流
workflow = StateGraph(StandardState)

# 添加节点
workflow.add_node("process_input", process_user_input)
workflow.add_node("generate_response", generate_response)

# 添加边 - 显示状态流转路径
workflow.add_edge("process_input", "generate_response")

# 添加条件边 - 基于查询次数决定是否结束
workflow.add_conditional_edges(
    "generate_response",
    should_continue,
    {
        "continue": "process_input",
        "end": END
    }
)

# 设置入口点
workflow.set_entry_point("process_input")

# 编译工作流
graph = workflow.compile()

# ===========================================================
# 第6部分: 运行并观察状态变化
# ===========================================================

print("\n运行LangGraph工作流，观察状态变化:")
print("="*50)

# 初始化状态
state = initialize_standard_state()

# 模拟用户输入
user_messages = [
    "什么是LangGraph?",
    "LangGraph与LangChain的关系是什么?",
    "如何在LangGraph中设计良好的状态?"
]

# 运行图并观察状态变化
print("\n初始状态:", state)

for i, user_msg in enumerate(user_messages):
    print(f"\n--- 用户输入 {i+1}: '{user_msg}' ---")
    
    # 添加用户消息到状态
    state = add_message(state, HumanMessage(content=user_msg))
    
    # 运行图
    state = graph.invoke(state)
    
    print(f"\n对话结束后的状态 (回合 {i+1}):\n{state}")
    print("-" * 50)

print("\n" + "="*50)
print("LangGraph状态设计详解示例结束")

# ===========================================================
# 总结:
# 1. 使用TypedDict可以创建强类型的状态定义，有助于代码提示和类型检查
# 2. 不同复杂度的状态设计适用于不同场景
# 3. 状态更新应保持不可变性，创建新状态而不是修改原状态
# 4. 良好的状态更新模式可以简化状态管理
# 5. LangGraph的状态流转是整个工作流的核心
# =========================================================== 