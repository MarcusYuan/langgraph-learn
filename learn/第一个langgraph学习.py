#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 简单学习案例
使用 Ollama 作为 LLM 提供者
服务器: 192.168.3.3:11434
模型: qwq:latest

==================================
WHY - 为什么使用LangGraph:
==================================
LangGraph是一个专为构建基于LLM的多代理应用设计的框架，它提供了以下优势：
1. 状态管理：有效管理对话和处理过程中的状态
2. 流程控制：使用图结构定义清晰的处理流程
3. 可组合性：轻松组合不同的处理节点形成复杂工作流
4. 可扩展性：方便扩展添加新的功能节点
5. 易于调试：清晰的流程可视化和状态跟踪

==================================
WHAT - 本代码实现了什么:
==================================
本示例实现了一个简单的聊天机器人，它包含以下功能：
1. 使用Ollama作为大语言模型提供者
2. 定义了一个双节点工作流：人类输入 -> AI助手回复
3. 构建了一个完整的状态管理机制
4. 实现了基本的对话交互循环
5. 异常处理与退出机制

==================================
HOW - 实现方式与架构:
==================================
本例使用LangGraph构建有向图工作流：
1. 状态设计：使用TypedDict定义聊天状态，包含消息历史
2. 节点定义：分别定义human和assistant两个核心节点
3. 图构建：设置节点和节点间的连接关系
4. 运行时：通过循环不断调用图进行对话交互

整体架构遵循了"状态-节点-图-执行"的LangGraph标准模式，
适合学习者理解LangGraph的基本概念和使用方法。
"""

import os
import sys
import requests
from typing import Dict, TypedDict, Annotated, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
# 使用langchain_ollama替代已弃用的Ollama类
from langchain_ollama import OllamaLLM

# =====================================================================
# 状态定义
# =====================================================================
# ChatState定义了图执行过程中的状态数据结构
# messages: 存储对话历史记录，包含人类消息和AI回复
class ChatState(TypedDict):
    messages: List[HumanMessage | AIMessage]  # 聊天历史记录

# =====================================================================
# 服务器连接测试
# =====================================================================
def test_ollama_connection(base_url, timeout=5):
    """测试与Ollama服务器的连接
    
    参数:
        base_url: Ollama服务器地址
        timeout: 连接超时时间(秒)
        
    返回:
        成功返回True，失败返回False
    """
    try:
        print(f"正在连接Ollama服务器: {base_url}")
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == 200:
            print("✅ 成功连接到Ollama服务器!")
            available_models = [model["name"] for model in response.json()["models"]]
            print(f"可用模型: {', '.join(available_models)}")
            return True
        else:
            print(f"❌ 服务器返回错误: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Ollama服务器，请确认服务器地址是否正确")
        return False
    except requests.exceptions.Timeout:
        print("❌ 连接超时，请检查网络或增加超时时间")
        return False
    except Exception as e:
        print(f"❌ 连接异常: {str(e)}")
        return False

# =====================================================================
# LLM配置
# =====================================================================
# Ollama服务器地址
OLLAMA_BASE_URL = "http://192.168.3.3:11434"
MODEL_NAME = "qwq:latest"

# 创建Ollama LLM实例，连接到指定服务器上的特定模型
# base_url: Ollama服务器地址
# model: 使用的模型名称
# temperature: 控制输出的随机性，值越高回复越多样化
llm = OllamaLLM(
    base_url=OLLAMA_BASE_URL,       # Ollama服务器地址
    model=MODEL_NAME,               # 使用的模型
    temperature=0.7,                # 温度参数，控制随机性
    request_timeout=20.0,           # 请求超时设置(秒)
)

# =====================================================================
# 节点函数定义
# =====================================================================

# 助手节点：负责处理用户输入并生成AI回复
def assistant(state: ChatState, config: RunnableConfig) -> ChatState:
    """LLM 助手处理用户消息并生成回复
    
    参数:
        state: 当前状态，包含消息历史
        config: 运行时配置参数
        
    返回:
        更新后的状态，包含新增的AI回复
    """
    messages = state["messages"]
    
    print("🤖 AI助手正在思考...")
    try:
        # 使用LLM模型处理消息历史并生成回复
        # invoke方法将整个消息历史传递给模型以保持上下文
        response = llm.invoke(messages)
        print(f"🤖 AI回复: {response}")
        
        # 将生成的回复添加到消息历史中，使用AIMessage包装
        messages.append(AIMessage(content=response))
    except Exception as e:
        print(f"❌ 生成回复时出错: {str(e)}")
        # 添加一个错误消息，确保程序可以继续运行
        messages.append(AIMessage(content=f"抱歉，我遇到了一个错误: {str(e)}"))
    
    # 返回更新后的状态
    return {"messages": messages}

# 人类节点：负责获取用户输入
def human(state: ChatState) -> ChatState:
    """处理人类输入，获取用户消息并添加到状态中
    
    参数:
        state: 当前状态，包含消息历史
        
    返回:
        更新后的状态，包含新增的用户消息
    """
    messages = state["messages"]
    
    # 通过控制台获取用户输入
    user_input = input("\n请输入您的问题: ")
    
    # 将用户输入封装为HumanMessage并添加到消息历史
    messages.append(HumanMessage(content=user_input))
    
    # 返回更新后的状态
    return {"messages": messages}

# =====================================================================
# 图构建函数
# =====================================================================

def build_graph():
    """创建并配置LangGraph工作流图
    
    构建图的步骤:
    1. 初始化状态图
    2. 添加处理节点
    3. 定义节点间连接关系
    4. 指定入口点
    5. 编译图
    
    返回:
        编译后的可执行图
    """
    # 初始化状态图，指定状态类型为ChatState
    workflow = StateGraph(ChatState)
    
    # 添加处理节点，将函数与节点名称关联
    workflow.add_node("human", human)         # 人类输入节点
    workflow.add_node("assistant", assistant) # AI助手节点
    
    # 添加边，定义human节点与assistant节点的连接
    # 表示用户输入后，下一步执行assistant节点
    workflow.add_edge("human", "assistant")
    
    # 添加从assistant回到human的边，形成对话循环
    # 表示AI回复后，继续获取用户输入
    workflow.add_edge("assistant", "human")
    
    # 设置图的入口点为human节点
    # 表示每轮对话都从获取用户输入开始
    workflow.set_entry_point("human")
    
    # 编译工作流图，转换为可执行的计算图
    return workflow.compile()

# =====================================================================
# 主程序
# =====================================================================

def main():
    """主函数，初始化并运行聊天对话循环
    
    流程:
    1. 构建图
    2. 初始化状态
    3. 进入对话循环
    4. 处理异常和退出条件
    """
    # 打印欢迎信息
    print("==== LangGraph 聊天示例 ====")
    print(f"尝试连接 Ollama ({OLLAMA_BASE_URL}) 的 {MODEL_NAME} 模型")
    print("输入 'exit' 或 'quit' 退出")
    
    # 测试与Ollama服务器的连接
    if not test_ollama_connection(OLLAMA_BASE_URL):
        print("\n❌ 无法连接到Ollama服务器，您可以:")
        print("1. 检查服务器地址是否正确")
        print("2. 确认Ollama服务是否运行")
        print("3. 检查网络连接")
        print("4. 修改代码中的OLLAMA_BASE_URL为您的Ollama服务器地址")
        
        use_local = input("\n是否尝试使用本地Ollama服务器 (http://localhost:11434)? (y/n): ")
        if use_local.lower() == 'y':
            global llm
            new_base_url = "http://localhost:11434"
            
            if test_ollama_connection(new_base_url):
                # 重新创建LLM实例，使用本地服务器
                llm = OllamaLLM(
                    base_url=new_base_url,
                    model="llama3", # 尝试使用一个常见模型
                    temperature=0.7,
                    request_timeout=20.0,
                )
            else:
                print("仍然无法连接，退出程序。")
                return
        else:
            print("退出程序。")
            return
    
    # 构建LangGraph工作流图
    graph = build_graph()
    
    # 初始化状态，空消息列表
    state = {"messages": []}
    
    try:
        # 执行工作流直到遇到退出条件
        while True:
            # 通过轮询方式检查用户是否要退出
            # 获取最近的一条用户消息
            if state["messages"]:
                last_human_message = next((msg.content for msg in reversed(state["messages"]) 
                                          if isinstance(msg, HumanMessage)), "")
                if last_human_message.lower() in ["exit", "quit", "退出"]:
                    print("再见!")
                    break
            
            # 使用invoke方法运行图（替代已不存在的step方法）
            # 将当前状态传入图中执行一次完整的节点序列
            state = graph.invoke(state)
                
    except KeyboardInterrupt:
        # 处理用户通过Ctrl+C中断
        print("\n程序被中断，退出中...")
    except Exception as e:
        # 处理其他异常
        print(f"发生错误: {e}")

# 程序入口
if __name__ == "__main__":
    main()
