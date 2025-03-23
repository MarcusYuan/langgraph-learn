#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 交互式UI集成
===================
本示例讲解LangGraph与多种UI框架的集成:
1. Streamlit集成 - 构建Web应用
2. Gradio界面 - 快速原型开发
3. FastAPI集成 - 构建REST API服务

WHY - 设计思路:
1. LangGraph应用需要友好的用户界面以提升用户体验
2. 不同场景需要不同类型的界面(Web应用、API服务等)
3. 需要将LangGraph的状态管理与UI框架结合
4. 流式输出能力需要在UI层面得到支持
5. 需要考虑多用户并发访问的状态隔离

HOW - 实现方式:
1. 使用Streamlit构建交互式Web应用
2. 使用Gradio实现快速原型开发
3. 使用FastAPI提供API服务
4. 设计合理的状态管理机制
5. 实现流式输出的UI展示

WHAT - 功能作用:
通过本示例，你将学习如何将LangGraph应用与各种UI框架集成，
构建友好的用户界面，为终端用户提供良好的交互体验，
并了解在集成过程中的关键考虑点和最佳实践。

学习目标:
- 掌握LangGraph与Streamlit的集成方法
- 了解LangGraph与Gradio的结合方式
- 学习构建基于LangGraph的API服务
- 理解UI集成中的状态管理策略
"""

import os
import copy
import time
from typing import TypedDict, List, Dict, Any, Optional, Generator, Callable
import json
import asyncio
import threading
from datetime import datetime

# LangGraph相关导入
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM

# =================================================================
# 第1部分: 基础组件 - 对话状态和图定义
# =================================================================

class ChatState(TypedDict):
    """对话状态定义
    
    WHY - 设计思路:
    1. 需要存储对话历史以保持上下文
    2. 需要记录元数据便于UI展示和状态管理
    3. 需要支持流式输出的状态标记
    
    HOW - 实现方式:
    1. 使用TypedDict定义类型安全的状态结构
    2. 包含消息历史、元数据和流式输出标记
    3. 设计简洁的结构确保UI层易于使用
    
    WHAT - 功能作用:
    提供UI层和LangGraph之间的数据交换格式，
    确保数据一致性和类型安全
    """
    messages: List[Dict[str, Any]]  # 消息历史: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    metadata: Dict[str, Any]  # 元数据: 会话ID、时间戳等
    streaming: bool  # 流式输出标记

def initialize_state() -> ChatState:
    """初始化对话状态
    
    WHY - 设计思路:
    1. 需要为每个用户会话提供初始状态
    2. 初始状态需要包含系统提示和基本元数据
    
    HOW - 实现方式:
    1. 创建包含空消息列表的状态字典
    2. 添加系统消息设定对话基调
    3. 初始化元数据和流式标记
    
    WHAT - 功能作用:
    为UI应用提供一致的起点状态，确保每个用户会话
    从相同的初始状态开始
    
    Returns:
        ChatState: 初始化的状态
    """
    session_id = f"session-{int(time.time())}"
    
    return {
        "messages": [
            {"role": "system", "content": "你是一个由LangGraph驱动的AI助手，通过UI界面与用户交流。请提供有帮助、安全且友好的回答。"}
        ],
        "metadata": {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        },
        "streaming": False
    }

def get_llm():
    """获取LLM实例
    
    WHY - 设计思路:
    1. 需要统一的LLM配置点
    2. 支持不同UI集成方式使用相同的底层模型
    
    HOW - 实现方式:
    1. 使用OllamaLLM提供本地模型推理
    2. 配置适当的参数确保输出质量
    
    WHAT - 功能作用:
    提供一个配置好的LLM实例，供各UI集成方式使用，
    确保输出一致性和质量
    
    Returns:
        OllamaLLM: LLM实例
    """
    return OllamaLLM(
        model="qwen:0.5b",  # 可替换为其他可用模型
        temperature=0.7,
    )

# =================================================================
# 第2部分: LangGraph核心逻辑 - 节点函数和图构建
# =================================================================

def user_input_node(state: ChatState, message: str) -> ChatState:
    """处理用户输入的节点
    
    WHY - 设计思路:
    1. 需要将UI层的用户输入整合到LangGraph状态中
    2. 需要维护状态的不变性
    
    HOW - 实现方式:
    1. 创建状态的深拷贝确保不可变性
    2. 将用户消息添加到消息历史
    3. 更新元数据时间戳
    
    WHAT - 功能作用:
    处理来自UI层的用户输入，更新状态中的消息历史
    
    Args:
        state: 当前状态
        message: 用户输入文本
        
    Returns:
        ChatState: 更新后的新状态
    """
    new_state = copy.deepcopy(state)
    
    # 添加用户消息
    new_state["messages"].append({"role": "user", "content": message})
    
    # 更新元数据
    new_state["metadata"]["last_updated"] = datetime.now().isoformat()
    
    return new_state

def ai_response_node(state: ChatState) -> ChatState:
    """生成AI响应的节点
    
    WHY - 设计思路:
    1. 需要处理用户输入并生成适当的响应
    2. 需要保持状态的不变性
    3. 响应需要考虑整个对话历史
    
    HOW - 实现方式:
    1. 创建状态的深拷贝确保不可变性
    2. 从状态中提取对话历史
    3. 调用LLM生成响应
    4. 将响应添加到消息历史
    
    WHAT - 功能作用:
    根据对话历史生成AI响应，并更新状态
    
    Args:
        state: 当前状态
        
    Returns:
        ChatState: 更新后的新状态
    """
    new_state = copy.deepcopy(state)
    
    # 获取LLM
    llm = get_llm()
    
    # 提取对话历史
    history = []
    for msg in new_state["messages"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            history.append(SystemMessage(content=msg["content"]))
    
    # 生成AI响应
    response = llm.invoke(history)
    
    # 添加到消息历史
    new_state["messages"].append({"role": "assistant", "content": response.content})
    
    # 更新元数据
    new_state["metadata"]["last_updated"] = datetime.now().isoformat()
    
    return new_state

def create_chat_graph():
    """创建对话图
    
    WHY - 设计思路:
    1. 需要一个封装对话逻辑的图结构
    2. 图需要支持基本的用户输入和AI响应流程
    3. 图应保持简单以便于UI集成
    
    HOW - 实现方式:
    1. 创建基于ChatState的StateGraph
    2. 添加处理用户输入和生成AI响应的节点
    3. 定义节点间的连接关系
    4. 设置入口点
    
    WHAT - 功能作用:
    提供一个封装对话逻辑的图结构，供不同UI框架使用
    
    Returns:
        StateGraph: 创建的对话图
    """
    # 创建状态图
    workflow = StateGraph(ChatState)
    
    # 自定义用户输入节点（接收外部消息）
    def process_user_message(state, input_dict):
        return user_input_node(state, input_dict.get("message", ""))
    
    # 添加节点
    workflow.add_node("user_input", process_user_message)
    workflow.add_node("ai_response", ai_response_node)
    
    # 设置边缘
    workflow.add_edge("user_input", "ai_response")
    workflow.add_edge("ai_response", END)
    
    # 设置入口点
    workflow.set_entry_point("user_input")
    
    # 编译图
    return workflow.compile()

# =================================================================
# 第3部分: Streamlit集成
# =================================================================

def setup_streamlit_ui():
    """
    设置Streamlit UI
    
    WHY - 设计思路:
    1. 需要提供一个基于Web的用户界面
    2. 界面需要支持对话历史显示和用户输入
    3. 需要保持会话状态以支持多轮对话
    
    HOW - 实现方式:
    1. 使用Streamlit的会话状态存储对话状态和图实例
    2. 设置页面布局和UI组件
    3. 处理用户输入并调用LangGraph图
    4. 显示对话历史
    
    WHAT - 功能作用:
    提供一个完整的Streamlit Web应用，展示LangGraph与Streamlit的集成
    """
    # 以下代码需要在Streamlit环境中运行
    streamlit_code = """
import streamlit as st
from datetime import datetime

# 导入上述定义的函数和类
# 确保此脚本中的所有函数和类都已导入
# 在实际使用时可能需要调整导入方式

# 设置页面标题和布局
st.set_page_config(page_title="LangGraph聊天应用", layout="wide")
st.title("LangGraph聊天应用")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.graph = create_chat_graph()
    st.session_state.state = initialize_state()

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 用户输入
user_input = st.chat_input("请输入您的问题")
if user_input:
    # 添加用户消息到UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # 传递给LangGraph处理
    with st.spinner("AI思考中..."):
        # 调用图处理用户输入
        result = st.session_state.graph.invoke(
            {"message": user_input},
            st.session_state.state
        )
        st.session_state.state = result
    
    # 添加AI回复到UI
    ai_message = result["messages"][-1]
    st.session_state.messages.append(ai_message)
    with st.chat_message("assistant"):
        st.write(ai_message["content"])

# 侧边栏添加说明
with st.sidebar:
    st.header("关于")
    st.write("这是一个使用LangGraph和Streamlit构建的聊天应用示例。")
    st.write("LangGraph负责处理对话逻辑，Streamlit提供用户界面。")
    
    st.header("会话信息")
    st.write(f"会话ID: {st.session_state.state['metadata']['session_id']}")
    st.write(f"创建时间: {st.session_state.state['metadata']['created_at']}")
    st.write(f"消息数量: {len(st.session_state.messages)}")
    
    if st.button("清空对话"):
        st.session_state.messages = []
        st.session_state.state = initialize_state()
        st.rerun()
"""
    print("Streamlit UI代码示例:")
    print("-" * 50)
    print(streamlit_code)
    print("-" * 50)
    print("要运行此Streamlit应用，请将上述代码保存到streamlit_app.py文件中，然后执行:")
    print("streamlit run streamlit_app.py")

# =================================================================
# 第4部分: Gradio集成
# =================================================================

def setup_gradio_ui():
    """
    设置Gradio UI
    
    WHY - 设计思路:
    1. 需要提供一个快速搭建的演示界面
    2. 界面需要支持对话历史和用户输入
    3. 需要处理会话状态
    
    HOW - 实现方式:
    1. 定义处理用户输入的回调函数
    2. 设置Gradio聊天界面组件
    3. 集成LangGraph处理逻辑
    
    WHAT - 功能作用:
    提供一个Gradio界面示例，展示LangGraph与Gradio的集成方式
    """
    # 以下代码需要在Gradio环境中运行
    gradio_code = """
import gradio as gr
import copy
import time
from datetime import datetime

# 导入上述定义的函数和类
# 确保此脚本中的所有函数和类都已导入
# 在实际使用时可能需要调整导入方式

# 创建对话图实例
graph = create_chat_graph()

# 保存会话状态的字典
sessions = {}

def respond(message, chat_history, session_id=None):
    # 创建或获取会话状态
    if session_id is None:
        session_id = f"session-{int(time.time())}"
    
    if session_id not in sessions:
        sessions[session_id] = initialize_state()
    
    state = sessions[session_id]
    
    # 调用图处理用户输入
    result = graph.invoke(
        {"message": message},
        state
    )
    
    # 更新会话状态
    sessions[session_id] = result
    
    # 获取AI响应
    ai_response = result["messages"][-1]["content"]
    
    # 更新聊天历史
    chat_history.append((message, ai_response))
    return "", chat_history, session_id

# 创建Gradio界面
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# LangGraph聊天应用")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600)
            with gr.Row():
                msg = gr.Textbox(placeholder="请输入您的问题", scale=3)
                session_id = gr.Textbox(visible=False)
                submit = gr.Button("发送")
        
        with gr.Column(scale=1):
            gr.Markdown("### 会话信息")
            session_info = gr.Markdown("当前没有活动会话")
            clear_btn = gr.Button("清空对话")
    
    # 设置提交事件
    submit_event = submit.click(
        respond, 
        inputs=[msg, chatbot, session_id], 
        outputs=[msg, chatbot, session_id]
    )
    
    # 支持按回车发送
    msg.submit(
        respond, 
        inputs=[msg, chatbot, session_id], 
        outputs=[msg, chatbot, session_id]
    )
    
    # 清空对话
    def clear_chat():
        return "", [], None
    
    clear_btn.click(
        clear_chat,
        inputs=[],
        outputs=[msg, chatbot, session_id]
    )
    
    # 更新会话信息
    def update_session_info(session_id):
        if session_id and session_id in sessions:
            state = sessions[session_id]
            created_at = state["metadata"]["created_at"]
            messages_count = len(state["messages"])
            return f"会话ID: {session_id}\\n创建时间: {created_at}\\n消息数量: {messages_count}"
        return "当前没有活动会话"
    
    submit_event.then(
        update_session_info,
        inputs=[session_id],
        outputs=[session_info]
    )

# 启动Gradio应用
demo.launch()
"""
    print("Gradio UI代码示例:")
    print("-" * 50)
    print(gradio_code)
    print("-" * 50)
    print("要运行此Gradio应用，请将上述代码保存到gradio_app.py文件中，然后执行:")
    print("python gradio_app.py")

# =================================================================
# 第5部分: FastAPI集成 - API服务
# =================================================================

def setup_fastapi_service():
    """
    设置FastAPI服务
    
    WHY - 设计思路:
    1. 需要提供API服务以支持多种客户端集成
    2. 需要处理会话状态管理
    3. 需要支持流式输出
    
    HOW - 实现方式:
    1. 定义API端点处理用户请求
    2. 实现会话管理与状态持久化
    3. 支持普通响应和流式响应两种模式
    
    WHAT - 功能作用:
    提供一个FastAPI服务示例，展示LangGraph与Web API的集成方式
    """
    # 以下代码展示FastAPI集成示例
    fastapi_code = """
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import uuid
import json
import asyncio
from typing import Dict, List, Optional, Any
import copy
import time
from datetime import datetime

# 导入上述定义的函数和类
# 确保此脚本中的所有函数和类都已导入
# 在实际使用时可能需要调整导入方式

# 创建FastAPI应用
app = FastAPI(title="LangGraph API服务")

# 启用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话存储
sessions = {}

# API模型定义
class MessageRequest(BaseModel):
    message: str = Field(..., description="用户消息内容")
    session_id: Optional[str] = Field(None, description="会话ID，如果为空则创建新会话")
    stream: bool = Field(False, description="是否启用流式输出")

class MessageResponse(BaseModel):
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="AI响应内容")
    created_at: str = Field(..., description="响应创建时间")

# 会话管理工具
def get_or_create_session(session_id: Optional[str] = None):
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    
    # 创建新会话
    new_session_id = session_id or str(uuid.uuid4())
    sessions[new_session_id] = initialize_state()
    return new_session_id, sessions[new_session_id]

# 创建对话图实例
graph = create_chat_graph()

# API端点 - 非流式消息
@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    # 获取或创建会话
    session_id, state = get_or_create_session(request.session_id)
    
    # 处理流式请求
    if request.stream:
        return StreamingResponse(
            stream_response(request.message, state, session_id),
            media_type="text/event-stream"
        )
    
    # 调用图处理用户输入
    result = graph.invoke(
        {"message": request.message},
        state
    )
    
    # 更新会话状态
    sessions[session_id] = result
    
    # 获取AI响应
    ai_response = result["messages"][-1]["content"]
    created_at = datetime.now().isoformat()
    
    return MessageResponse(
        session_id=session_id,
        message=ai_response,
        created_at=created_at
    )

# 流式响应生成器
async def stream_response(message: str, state: dict, session_id: str):
    # 在实际应用中，这里应该集成LangGraph的流式响应功能
    # 这里为了演示，使用简单的模拟实现
    
    # 调用图处理用户输入
    result = graph.invoke(
        {"message": message},
        state
    )
    
    # 更新会话状态
    sessions[session_id] = result
    
    # 获取AI响应并模拟流式输出
    ai_response = result["messages"][-1]["content"]
    
    # 每个单词都作为一个流式事件发送
    for word in ai_response.split():
        yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        await asyncio.sleep(0.1)
    
    # 发送完成事件
    yield f"data: {json.dumps({'token': '', 'finished': True})}\n\n"

# 会话管理端点
@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    state = sessions[session_id]
    return {
        "session_id": session_id,
        "metadata": state["metadata"],
        "message_count": len(state["messages"])
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "success", "message": "会话已删除"}

# 启动服务器
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
"""
    print("FastAPI服务代码示例:")
    print("-" * 50)
    print(fastapi_code)
    print("-" * 50)
    print("要运行此FastAPI服务，请将上述代码保存到app.py文件中，然后执行:")
    print("uvicorn app:app --reload")

# =================================================================
# 第6部分: 示例运行
# =================================================================

def show_ui_integration_examples():
    """展示UI集成示例
    
    WHY - 设计思路:
    1. 需要展示不同UI集成方式的关键点
    2. 需要提供用户选择不同示例的能力
    
    HOW - 实现方式:
    1. 展示Streamlit、Gradio和FastAPI集成的关键代码
    2. 提供交互式菜单选择不同示例
    
    WHAT - 功能作用:
    提供一个互动式展示，帮助用户理解不同UI集成方式的特点
    """
    print("\n===== LangGraph UI集成示例 =====")
    
    print("\n本示例展示了LangGraph与多种UI框架的集成方式。")
    print("注意: 实际UI代码需要在相应环境中运行，这里只展示代码样例。")
    
    while True:
        print("\n请选择要查看的UI集成示例:")
        print("1. Streamlit集成")
        print("2. Gradio集成")
        print("3. FastAPI服务")
        print("0. 退出")
        
        choice = input("\n您的选择> ")
        
        if choice == "1":
            setup_streamlit_ui()
        elif choice == "2":
            setup_gradio_ui()
        elif choice == "3":
            setup_fastapi_service()
        elif choice == "0":
            break
        else:
            print("无效选择，请重试")

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行UI集成示例
    2. 需要适当的错误处理确保示例稳定运行
    3. 需要提供清晰的开始和结束提示
    
    HOW - 实现方式:
    1. 使用try-except包装主要执行逻辑
    2. 提供开始和结束提示
    3. 调用示例展示函数
    4. 总结关键学习点
    
    WHAT - 功能作用:
    作为程序入口点，执行UI集成示例，确保示例执行的稳定性
    """
    print("===== LangGraph 交互式UI集成学习示例 =====\n")
    
    try:
        # 运行UI集成示例
        show_ui_integration_examples()
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 将LangGraph与Streamlit集成构建Web应用")
        print("2. 使用Gradio快速搭建LangGraph的演示界面")
        print("3. 构建基于FastAPI的LangGraph服务API")
        print("4. 处理UI集成中的会话状态管理")
        print("5. 实现流式输出的UI展示")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 