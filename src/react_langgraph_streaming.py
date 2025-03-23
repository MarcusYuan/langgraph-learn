from typing import Dict, List, Optional, TypedDict, Any, AsyncIterator
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
import json
import asyncio
from langgraph.checkpoint import MemorySaver

# 定义状态类型
class ReactStreamState(TypedDict):
    # 消息历史
    messages: List[BaseMessage]
    # 组件状态
    counter: int
    # 渲染结果
    ui: str
    # 用户输入
    user_input: Optional[str]
    # 处理状态
    processing: bool

# 初始状态
def get_initial_state() -> ReactStreamState:
    return {
        "messages": [],
        "counter": 0,
        "ui": "",
        "user_input": None,
        "processing": False
    }

# 解析用户输入
def parse_input(state: ReactStreamState) -> Dict:
    """解析用户输入"""
    messages = state["messages"]
    
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {}
    
    user_input = messages[-1].content.strip().lower()
    
    return {"user_input": user_input, "processing": True}

# 更新状态 - 模拟异步处理
async def update_state(state: ReactStreamState) -> AsyncIterator[Dict]:
    """更新组件状态 - 带流式输出"""
    user_input = state["user_input"]
    counter = state["counter"]
    
    # 模拟处理开始
    yield {"ui": "处理中..."}  # 流式输出状态更新
    await asyncio.sleep(0.5)  # 模拟处理延迟
    
    # 根据用户输入更新计数器
    if user_input == "increment":
        # 模拟增量处理
        for i in range(1, 6):  # 模拟5个步骤的处理
            new_counter = counter + (i / 5)  # 逐步增加
            yield {"ui": f"增加中: {new_counter:.1f}"}
            await asyncio.sleep(0.2)  # 短暂延迟
        counter += 1
    elif user_input == "decrement" and counter > 0:
        # 模拟减量处理
        for i in range(1, 6):  # 模拟5个步骤的处理
            new_counter = counter - (i / 5)  # 逐步减少
            yield {"ui": f"减少中: {new_counter:.1f}"}
            await asyncio.sleep(0.2)  # 短暂延迟
        counter -= 1
    elif user_input == "reset":
        # 模拟重置处理
        steps = max(5, counter * 2)  # 根据当前值确定步骤数
        for i in range(1, int(steps) + 1):
            new_counter = counter * (1 - (i / steps))  # 逐步减少到0
            yield {"ui": f"重置中: {new_counter:.1f}"}
            await asyncio.sleep(0.1)  # 短暂延迟
        counter = 0
    
    # 最终状态更新
    yield {"counter": counter, "processing": False}

# 渲染UI
def render(state: ReactStreamState) -> Dict:
    """渲染组件UI"""
    counter = state["counter"]
    processing = state["processing"]
    
    # 构建UI表示
    ui = "===== React Streaming Counter App =====\n\n"
    
    if processing:
        ui += "状态: 处理中...\n\n"
    else:
        ui += "状态: 就绪\n\n"
    
    ui += f"当前计数: {counter}\n\n"
    ui += "可用操作:\n"
    ui += "1. increment - 增加计数器\n"
    ui += "2. decrement - 减少计数器\n"
    ui += "3. reset - 重置计数器为零\n"
    ui += "4. exit - 退出应用程序\n"
    
    return {"ui": ui}

# 生成AI响应
def generate_response(state: ReactStreamState) -> Dict:
    """生成AI响应消息"""
    ui = state["ui"]
    
    # 构建响应消息
    response = ui + "\n\n您想做什么？"
    
    messages = state["messages"].copy()
    messages.append(AIMessage(content=response))
    
    return {"messages": messages}

# 创建图
def create_react_stream_graph():
    # 创建状态图
    graph = StateGraph(ReactStreamState)
    
    # 添加节点
    graph.add_node("parse_input", parse_input)
    graph.add_node("update_state", update_state)
    graph.add_node("render", render)
    graph.add_node("generate_response", generate_response)
    
    # 添加边 - 模拟React的渲染循环
    graph.add_edge(START, "parse_input")
    graph.add_edge("parse_input", "update_state")
    graph.add_edge("update_state", "render")
    graph.add_edge("render", "generate_response")
    graph.add_edge("generate_response", END)
    
    # 创建检查点保存器
    checkpointer = MemorySaver()
    
    # 编译图
    return graph.compile(
        checkpointer=checkpointer,
        stream=True  # 启用流式处理
    ), checkpointer

# 主函数
async def main():
    # 创建图和检查点器
    app, checkpointer = create_react_stream_graph()
    
    # 初始状态
    state = get_initial_state()
    thread_id = "user_stream_123"
    
    print("欢迎使用基于React风格的流式计数器应用！")
    print("输入 'exit' 退出。")
    
    # 初始渲染
    state = await app.ainvoke(
        state,
        {"configurable": {"thread_id": thread_id}}
    )
    print(state["messages"][-1].content)
    
    while True:
        # 获取用户输入
        user_input = input("\n> ")
        
        if user_input.lower() == "exit":
            break
        
        # 更新消息
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        
        try:
            # 流式执行图
            print("\n开始处理...")
            async for chunk in app.astream(
                state,
                {"configurable": {"thread_id": thread_id}}
            ):
                # 如果有UI更新，显示它
                if "ui" in chunk and chunk["ui"] != state.get("ui", ""):
                    print(f"\r{chunk['ui']}", end="", flush=True)
            
            # 获取最终状态
            state = checkpointer.get_state(thread_id)
            print("\n\n处理完成!")
            print(state["messages"][-1].content)
            
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())