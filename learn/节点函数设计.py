#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 节点函数设计详解
==================================
本示例讲解LangGraph中节点函数的设计原则与模式:
1. 节点函数的输入输出规范
2. 纯函数vs带副作用的节点
3. 错误处理策略
4. 节点函数设计的最佳实践

本例使用智能助手场景展示不同类型的节点函数设计。

学习目标:
- 理解节点函数的基本结构和类型
- 掌握纯函数设计的优点和应用场景
- 学习如何正确处理节点函数中的错误
- 了解节点函数设计的高级模式
"""

import os
import json
import time
import random
from typing import TypedDict, List, Dict, Any, Optional, Tuple, Union, Callable
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

# ===========================================================
# 第1部分: 状态定义 - 作为节点函数的输入和输出
# ===========================================================

class AssistantState(TypedDict):
    """助手状态定义"""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # 对话历史
    context: Optional[Dict[str, Any]]  # 上下文信息
    tools_results: Optional[Dict[str, Any]]  # 工具调用结果
    current_tool: Optional[str]  # 当前使用的工具
    error: Optional[str]  # 错误信息
    thinking: Optional[str]  # 思考过程

def initialize_state() -> AssistantState:
    """初始化助手状态"""
    return {
        "messages": [
            SystemMessage(content="你是一个智能助手，可以回答用户问题、搜索信息和执行计算。")
        ],
        "context": {},
        "tools_results": {},
        "current_tool": None,
        "error": None,
        "thinking": None
    }

# ===========================================================
# 第2部分: 纯函数节点 vs 带副作用的节点
# ===========================================================

print("===== 纯函数节点 vs 带副作用的节点 =====")

# 2.1 纯函数节点 - 相同输入总是产生相同输出，无副作用
def pure_node_example(state: AssistantState) -> AssistantState:
    """纯函数节点示例 - 分析最新消息的意图
    
    纯函数特点:
    1. 不修改输入参数
    2. 不依赖外部状态
    3. 无副作用(不修改外部状态、不进行I/O操作)
    4. 相同输入始终产生相同输出
    """
    # 获取最新消息
    if not state["messages"] or not any(isinstance(msg, HumanMessage) for msg in state["messages"]):
        # 返回新状态，不修改原始状态
        return {
            **state,
            "context": {**(state.get("context") or {}), "intent": "greeting"}
        }
    
    # 获取最新的用户消息
    last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    # 分析意图 (简化版，实际可能使用LLM或分类器)
    intent = "general_query"  # 默认意图
    
    if any(keyword in last_user_msg.lower() for keyword in ["搜索", "查找", "寻找"]):
        intent = "search"
    elif any(keyword in last_user_msg.lower() for keyword in ["计算", "多少", "等于"]):
        intent = "calculation"
    elif any(keyword in last_user_msg.lower() for keyword in ["帮助", "怎么用", "使用说明"]):
        intent = "help"
        
    # 创建新的上下文 (不修改原始上下文)
    new_context = {**(state.get("context") or {}), "intent": intent}
    
    # 返回新状态，不修改原始状态
    return {
        **state,
        "context": new_context
    }

# 2.2 带副作用的节点 - 可能修改外部状态或进行I/O操作
def impure_node_example(state: AssistantState) -> AssistantState:
    """带副作用的节点示例 - 记录日志并调用外部API
    
    副作用包括:
    1. 文件I/O操作
    2. 网络请求
    3. 修改全局变量
    4. 打印输出
    """
    # 获取最新的用户消息
    last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    # 副作用1: 打印日志
    print(f"[LOG] 处理用户消息: '{last_user_msg}'")
    
    # 副作用2: 记录到文件 (实际项目中可能记录到数据库)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": last_user_msg,
        "session_id": "example_session"
    }
    
    try:
        # 副作用3: 文件操作
        # 注: 实际生产环境中应使用适当的日志系统而非直接文件操作
        os.makedirs("logs", exist_ok=True)
        with open("logs/assistant_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[ERROR] 无法写入日志: {str(e)}")
    
    # 副作用4: 模拟网络请求
    try:
        # 模拟网络延迟
        time.sleep(0.1)
        
        # 实际项目中这里可能是真实的API调用
        analytics_data = {
            "message_processed": True,
            "processing_time": 0.1
        }
        
        print(f"[LOG] API调用成功: {analytics_data}")
    except Exception as e:
        print(f"[ERROR] API调用失败: {str(e)}")
    
    # 返回新状态
    return {
        **state,
        "context": {
            **(state.get("context") or {}),
            "last_processed": datetime.now().isoformat()
        }
    }

# 比较两种节点的特点
print("\n纯函数节点特点:")
print("1. 可预测性高 - 相同输入总是产生相同输出")
print("2. 易于测试 - 不需要模拟外部依赖")
print("3. 易于并行化 - 无共享状态导致的竞争条件")
print("4. 易于调试 - 行为由输入完全决定")

print("\n带副作用节点特点:")
print("1. 可以与外部世界交互 - 文件、网络、数据库等")
print("2. 适合日志记录、监控和集成")
print("3. 测试更复杂 - 需要模拟外部依赖")
print("4. 调试更困难 - 行为受外部状态影响")

# 副作用隔离示例 - 将纯逻辑与副作用分离
def isolated_effects_example(state: AssistantState) -> Tuple[AssistantState, Callable[[], None]]:
    """副作用隔离示例 - 分离状态更新和副作用
    
    返回:
        - 更新后的状态
        - 副作用函数(可以在安全的时间点执行)
    """
    # 纯函数部分 - 计算新状态
    last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    new_state = {
        **state,
        "context": {
            **(state.get("context") or {}),
            "last_message": last_user_msg,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # 副作用部分 - 包装在函数中
    def side_effects():
        # 所有副作用都在这里
        print(f"[LOG] 处理消息: '{last_user_msg}'")
        
        # 写入日志文件
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/assistant_log.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()}: {last_user_msg}\n")
        except Exception as e:
            print(f"[ERROR] 日志写入失败: {str(e)}")
    
    # 返回新状态和副作用函数
    return new_state, side_effects

print("\n副作用隔离模式:")
test_state = initialize_state()
test_state["messages"].append(HumanMessage(content="你好，请帮我搜索一下天气"))
new_state, effects_fn = isolated_effects_example(test_state)
print("1. 状态更新与副作用分离")
print("2. 可以在适当的时机执行副作用")
print("3. 保持主逻辑的纯函数特性")
effects_fn()  # 执行副作用

print("\n" + "="*50 + "\n")

# ===========================================================
# 第3部分: 节点函数的输入输出规范
# ===========================================================

print("===== 节点函数的输入输出规范 =====")

# 3.1 基本规范: 接收状态字典，返回状态字典
def basic_node(state: AssistantState) -> AssistantState:
    """基本节点函数 - 接收状态字典，返回状态字典"""
    # 简单处理: 添加时间戳到上下文
    return {
        **state,
        "context": {
            **(state.get("context") or {}),
            "processed_at": datetime.now().isoformat()
        }
    }

# 3.2 接收额外参数的节点
def node_with_config(state: AssistantState, config: RunnableConfig) -> AssistantState:
    """带配置参数的节点函数"""
    # 使用配置中的参数
    tags = config.get("tags", [])
    callbacks = config.get("callbacks", [])
    
    print(f"节点配置: tags={tags}, callbacks有{len(callbacks)}个")
    
    # 处理状态
    return {
        **state,
        "context": {
            **(state.get("context") or {}),
            "used_config": True
        }
    }

# 3.3 部分状态更新
def partial_state_update(state: AssistantState) -> Dict[str, Any]:
    """部分状态更新 - 只返回需要更新的部分"""
    # 只更新thinking字段
    return {
        "thinking": "我应该如何回答这个问题呢..."
    }

# 3.4 条件状态更新
def conditional_update(state: AssistantState) -> Optional[Dict[str, Any]]:
    """条件状态更新 - 有时可能不更新状态"""
    # 获取最新的用户消息
    last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    # 只有当消息包含问号时才更新
    if "?" in last_user_msg:
        return {
            "context": {
                **(state.get("context") or {}),
                "is_question": True
            }
        }
    
    # 返回None表示不更新状态
    return None

print("节点函数的基本输入输出约定:")
print("1. 基本模式: 接收完整状态，返回完整状态")
print("2. 配置参数: 接收状态和运行配置")
print("3. 部分更新: 只返回需要更新的字段")
print("4. 条件更新: 可能返回None表示不更新")

print("\n" + "="*50 + "\n")

# ===========================================================
# 第4部分: 错误处理策略
# ===========================================================

print("===== 错误处理策略 =====")

# 4.1 基本错误处理 - try-except捕获错误
def basic_error_handling(state: AssistantState) -> AssistantState:
    """基本错误处理 - 使用try-except捕获错误"""
    try:
        # 可能出错的操作
        last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                             if isinstance(msg, HumanMessage)), "")
        
        # 模拟可能出错的操作
        result = process_message(last_user_msg)
        
        # 处理成功的情况
        return {
            **state,
            "tools_results": {
                **(state.get("tools_results") or {}),
                "message_processing": result
            }
        }
    except Exception as e:
        # 捕获错误并记录到状态中
        print(f"[ERROR] 处理消息时出错: {str(e)}")
        
        return {
            **state,
            "error": f"处理消息时出错: {str(e)}"
        }

# 模拟可能失败的处理函数
def process_message(message: str) -> Dict[str, Any]:
    """处理消息的函数，可能会失败"""
    # 模拟随机失败
    if random.random() < 0.3:  # 30%的概率失败
        raise ValueError("消息处理失败: 模拟的随机错误")
    
    # 正常情况
    return {
        "processed": True,
        "length": len(message),
        "keywords": [word for word in message.split() if len(word) > 4]
    }

# 4.2 优雅降级 - 出错时使用备用方案
def graceful_degradation(state: AssistantState) -> AssistantState:
    """优雅降级 - 主方法失败时使用备用方法"""
    last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    # 尝试主要处理方法
    try:
        # 模拟主要处理方法
        result = process_message(last_user_msg)
        
        return {
            **state,
            "tools_results": {
                **(state.get("tools_results") or {}),
                "primary_result": result
            }
        }
    except Exception as primary_error:
        print(f"[WARNING] 主要处理方法失败: {str(primary_error)}，尝试备用方法")
        
        # 尝试备用处理方法
        try:
            # 模拟备用处理方法 (更简单但可靠性更高)
            fallback_result = {
                "processed": True,
                "basic_analysis": True,
                "length": len(last_user_msg)
            }
            
            return {
                **state,
                "tools_results": {
                    **(state.get("tools_results") or {}),
                    "fallback_result": fallback_result
                },
                "context": {
                    **(state.get("context") or {}),
                    "used_fallback": True
                }
            }
        except Exception as fallback_error:
            # 如果备用方法也失败，记录错误
            print(f"[ERROR] 备用处理方法也失败: {str(fallback_error)}")
            
            return {
                **state,
                "error": f"处理失败: {str(primary_error)}；备用方法也失败: {str(fallback_error)}"
            }

# 4.3 带恢复机制的错误处理
def recoverable_error_handling(state: AssistantState) -> Union[AssistantState, str]:
    """带恢复机制的错误处理 - 返回状态或特殊节点名称"""
    try:
        # 可能出错的操作
        result = process_complex_task()
        
        # 成功时返回更新的状态
        return {
            **state,
            "tools_results": {
                **(state.get("tools_results") or {}),
                "complex_task": result
            }
        }
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"[ERROR] {error_type}: {error_msg}")
        
        # 根据错误类型返回不同的恢复节点名称
        if "权限" in error_msg or "permission" in error_msg.lower():
            # 权限错误，转到权限处理节点
            return "handle_permission_error"
        elif "超时" in error_msg or "timeout" in error_msg.lower():
            # 超时错误，转到重试节点
            return "retry_node"
        else:
            # 其他错误，记录到状态并继续
            return {
                **state,
                "error": f"{error_type}: {error_msg}",
                "context": {
                    **(state.get("context") or {}),
                    "error_handled": True
                }
            }

# 模拟复杂任务
def process_complex_task() -> Dict[str, Any]:
    """模拟复杂任务处理，可能会出现不同类型的错误"""
    # 模拟不同类型的错误
    error_type = random.choice(["none", "permission", "timeout", "value"])
    
    if error_type == "permission":
        raise PermissionError("没有执行该操作的权限")
    elif error_type == "timeout":
        raise TimeoutError("操作超时")
    elif error_type == "value":
        raise ValueError("无效的输入值")
    
    # 没有错误的情况
    return {"success": True, "result": "任务完成"}

# 自定义异常类
class ValidationError(Exception):
    """表示验证失败的自定义异常"""
    pass

# 4.4 预防性错误处理 - 验证输入
def validate_input(state: AssistantState) -> AssistantState:
    """预防性错误处理 - 在处理前验证输入"""
    # 1. 检查必需的状态字段
    if "messages" not in state:
        return {
            **state,
            "error": "状态缺少必需的messages字段"
        }
    
    # 2. 检查消息历史是否为空
    if not state["messages"]:
        return {
            **state,
            "error": "消息历史为空"
        }
    
    # 3. 检查是否有用户消息
    last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), None)
    
    if last_user_msg is None:
        return {
            **state,
            "error": "没有找到用户消息"
        }
    
    # 4. 验证消息内容
    if len(last_user_msg) < 2:
        return {
            **state,
            "error": "用户消息太短，无法处理"
        }
    
    # 验证通过，处理消息
    return {
        **state,
        "context": {
            **(state.get("context") or {}),
            "validated": True,
            "message_length": len(last_user_msg)
        }
    }

print("错误处理策略:")
print("1. 基本错误处理: 使用try-except捕获错误并记录")
print("2. 优雅降级: 主要方法失败时使用备用方法")
print("3. 带恢复机制: 根据错误类型进行不同处理")
print("4. 预防性错误处理: 在处理前验证输入")

# 测试基本错误处理
test_state = initialize_state()
test_state["messages"].append(HumanMessage(content="测试错误处理"))
result_state = basic_error_handling(test_state)
print("\n基本错误处理测试结果:")
if "error" in result_state and result_state["error"]:
    print(f"- 出现错误: {result_state['error']}")
else:
    print("- 处理成功")

print("\n" + "="*50 + "\n")

# ===========================================================
# 第5部分: 节点函数的高级模式
# ===========================================================

print("===== 节点函数的高级模式 =====")

# 5.1 链式节点 - 将多个处理步骤组合成一个节点
def chained_node(state: AssistantState) -> AssistantState:
    """链式节点 - 将多个处理步骤组合在一个节点中"""
    # 步骤1: 解析用户意图
    state = pure_node_example(state)
    
    # 步骤2: 记录处理时间
    state = {
        **state,
        "context": {
            **(state.get("context") or {}),
            "processed_at": datetime.now().isoformat()
        }
    }
    
    # 步骤3: 根据意图处理消息
    intent = state.get("context", {}).get("intent", "general_query")
    
    if intent == "search":
        # 模拟搜索结果
        state = {
            **state,
            "tools_results": {
                **(state.get("tools_results") or {}),
                "search": {"found": True, "results": ["示例结果1", "示例结果2"]}
            }
        }
    elif intent == "calculation":
        # 模拟计算结果
        state = {
            **state,
            "tools_results": {
                **(state.get("tools_results") or {}),
                "calculation": {"result": 42}
            }
        }
    
    return state

# 5.2 条件节点 - 根据状态决定是否执行
def conditional_node(state: AssistantState) -> AssistantState:
    """条件节点 - 只在满足条件时执行处理"""
    # 检查是否需要处理
    intent = state.get("context", {}).get("intent")
    
    # 如果没有意图或不是搜索意图，则跳过处理
    if not intent or intent != "search":
        print("[INFO] 不是搜索意图，跳过搜索处理")
        return state
    
    # 是搜索意图，执行搜索处理
    print("[INFO] 检测到搜索意图，执行搜索")
    
    # 模拟搜索操作
    return {
        **state,
        "tools_results": {
            **(state.get("tools_results") or {}),
            "search": {"found": True, "results": ["搜索结果1", "搜索结果2"]}
        }
    }

# 5.3 思考-行动-观察模式 (ReAct模式)
def think_node(state: AssistantState) -> AssistantState:
    """思考节点 - 分析当前状态并决定下一步行动"""
    # 获取最新消息
    last_user_msg = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    # 思考过程 (实际中可能使用LLM)
    thinking = f"我需要理解用户的问题: '{last_user_msg}'。"
    
    if "天气" in last_user_msg.lower():
        thinking += " 用户询问天气，我应该使用天气API。"
        action = "weather_api"
    elif "新闻" in last_user_msg.lower():
        thinking += " 用户询问新闻，我应该搜索最新新闻。"
        action = "news_search"
    else:
        thinking += " 这是一般问题，我应该直接回答。"
        action = "direct_answer"
    
    # 更新状态
    return {
        **state,
        "thinking": thinking,
        "current_tool": action
    }

def act_node(state: AssistantState) -> AssistantState:
    """行动节点 - 执行think节点决定的操作"""
    action = state.get("current_tool")
    
    if not action:
        return {
            **state,
            "error": "没有指定操作"
        }
    
    # 执行不同的操作
    if action == "weather_api":
        # 模拟调用天气API
        result = {"temperature": "23°C", "condition": "晴朗"}
    elif action == "news_search":
        # 模拟搜索新闻
        result = {"headlines": ["示例新闻标题1", "示例新闻标题2"]}
    elif action == "direct_answer":
        # 不需要外部工具
        result = {"direct": True}
    else:
        result = {"error": f"未知操作: {action}"}
    
    # 更新状态
    return {
        **state,
        "tools_results": {
            **(state.get("tools_results") or {}),
            action: result
        }
    }

def observe_node(state: AssistantState) -> AssistantState:
    """观察节点 - 分析行动结果并生成回复"""
    # 获取工具结果
    action = state.get("current_tool")
    results = state.get("tools_results", {}).get(action, {})
    
    # 生成回复 (实际中可能使用LLM)
    if action == "weather_api" and results:
        response = f"当前天气是{results.get('temperature', '未知')}，天气状况{results.get('condition', '未知')}。"
    elif action == "news_search" and results:
        headlines = results.get("headlines", [])
        if headlines:
            response = f"以下是最新新闻：\n" + "\n".join([f"- {h}" for h in headlines])
        else:
            response = "抱歉，没有找到相关新闻。"
    elif action == "direct_answer":
        # 这里实际会使用LLM生成回复
        response = "这是一个直接回复的示例。在实际应用中，这里会使用LLM生成回复。"
    else:
        response = "抱歉，我无法处理这个请求。"
    
    # 添加AI消息到历史
    new_messages = state["messages"].copy()
    new_messages.append(AIMessage(content=response))
    
    # 更新状态
    return {
        **state,
        "messages": new_messages
    }

print("节点函数的高级模式:")
print("1. 链式节点: 将多个处理步骤组合成一个节点")
print("2. 条件节点: 根据状态决定是否执行处理")
print("3. ReAct模式: 思考-行动-观察的循环")

# 测试ReAct模式
print("\nReAct模式测试:")
react_state = initialize_state()
react_state["messages"].append(HumanMessage(content="今天北京的天气怎么样？"))

print("1. 思考阶段")
react_state = think_node(react_state)
print(f"- 思考结果: {react_state['thinking']}")
print(f"- 决定操作: {react_state['current_tool']}")

print("2. 行动阶段")
react_state = act_node(react_state)
print(f"- 操作结果: {react_state['tools_results']}")

print("3. 观察阶段")
react_state = observe_node(react_state)
print(f"- 生成回复: {react_state['messages'][-1].content}")

print("\n" + "="*50 + "\n")

# ===========================================================
# 第6部分: 将节点函数组合成图
# ===========================================================

print("===== 将节点函数组合成图 =====")

# 创建LLM实例
try:
    llm = OllamaLLM(
        base_url="http://localhost:11434",
        model="llama3",
        temperature=0.7,
    )
except:
    # 如果无法连接到Ollama，使用假的LLM响应
    print("无法连接到Ollama服务，将使用模拟的LLM响应进行演示")
    class MockLLM:
        def invoke(self, messages, **kwargs):
            return "这是一个模拟的LLM回复，用于演示节点函数。"
    llm = MockLLM()

# 创建图实例
workflow = StateGraph(AssistantState)

# 添加节点
workflow.add_node("process_input", validate_input)  # 验证输入
workflow.add_node("analyze_intent", pure_node_example)  # 分析意图
workflow.add_node("think", think_node)  # 思考
workflow.add_node("act", act_node)  # 行动
workflow.add_node("observe", observe_node)  # 观察

# 添加错误处理节点
workflow.add_node("handle_error", lambda state: {
    **state,
    "messages": state["messages"] + [AIMessage(content="抱歉，我遇到了一个错误。请重新描述您的问题。")],
    "error": None  # 清除错误
})

# 添加边 - 定义节点间的连接
workflow.add_edge("process_input", "analyze_intent")
workflow.add_edge("analyze_intent", "think")
workflow.add_edge("think", "act")
workflow.add_edge("act", "observe")
workflow.add_edge("observe", END)  # 观察后结束

# 添加条件边 - 处理错误情况
workflow.add_conditional_edges(
    "process_input",
    lambda state: "handle_error" if state.get("error") else "analyze_intent"
)

# 设置入口点
workflow.set_entry_point("process_input")

# 编译图
graph = workflow.compile()

print("图结构构建完成:")
print("1. 节点: process_input → analyze_intent → think → act → observe → END")
print("2. 错误处理: process_input → handle_error (当出现错误时)")

# 测试图执行
print("\n测试图执行:")
test_graph_state = initialize_state()
test_graph_state["messages"].append(HumanMessage(content="今天北京的天气怎么样？"))

print("初始状态:")
print(f"- 消息: {[msg.content for msg in test_graph_state['messages']]}")

# 执行图
final_state = graph.invoke(test_graph_state)

print("\n最终状态:")
print(f"- 思考过程: {final_state.get('thinking', 'N/A')}")
print(f"- 使用工具: {final_state.get('current_tool', 'N/A')}")
print(f"- 工具结果: {final_state.get('tools_results', {})}")
print(f"- 最终回复: {final_state['messages'][-1].content}")

print("\n" + "="*50)
print("LangGraph节点函数设计详解示例结束")

# ===========================================================
# 总结:
# 1. 节点函数应遵循明确的输入输出规范
# 2. 纯函数设计有利于测试、调试和并行化
# 3. 错误处理是节点函数设计的重要部分
# 4. 高级模式如链式节点和ReAct模式增强了节点功能
# 5. 节点函数是构建LangGraph应用的基础构件
# =========================================================== 