from typing import Dict, List, Optional, TypedDict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
import json
import os
from langgraph.checkpoint import MemorySaver

# 定义状态类型
class ReactState(TypedDict):
    # 消息历史
    messages: List[BaseMessage]
    # 组件状态
    count: int
    # 渲染结果
    ui: str
    # 用户输入
    user_input: Optional[str]

# 初始状态
def get_initial_state() -> ReactState:
    return {
        "messages": [],
        "count": 0,
        "ui": "",
        "user_input": None
    }

# 解析用户输入
def parse_input(state: ReactState) -> Dict:
    """解析用户输入"""
    messages = state["messages"]
    
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {}
    
    user_input = messages[-1].content.strip().lower()
    
    return {"user_input": user_input}

# 更新状态 - 类似React的useState
def update_state(state: ReactState) -> Dict:
    """更新组件状态"""
    user_input = state["user_input"]
    count = state["count"]
    
    if user_input == "increment":
        count += 1
    elif user_input == "decrement" and count > 0:
        count -= 1
    elif user_input == "reset":
        count = 0
    
    return {"count": count}

# 渲染UI
def render(state: ReactState) -> Dict:
    """渲染组件UI"""
    count = state["count"]
    
    # 构建UI表示
    ui = "===== React Counter App =====\n\n"
    ui += f"Current count: {count}\n\n"
    ui += "Available actions:\n"
    ui += "1. increment - Increase counter\n"
    ui += "2. decrement - Decrease counter\n"
    ui += "3. reset - Reset counter to zero\n"
    ui += "4. exit - Exit the application\n"
    
    return {"ui": ui}

# 生成AI响应
def generate_response(state: ReactState) -> Dict:
    """生成AI响应消息"""
    ui = state["ui"]
    
    # 构建响应消息
    response = ui + "\n\nWhat would you like to do?"
    
    messages = state["messages"].copy()
    messages.append(AIMessage(content=response))
    
    return {"messages": messages}

# 创建图
def create_react_graph():
    # 创建状态图
    graph = StateGraph(ReactState)
    
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
    
    # 编译图 - 设置断点在update_state之后
    return graph.compile(
        checkpointer=checkpointer,
        breakpoints=["after:update_state"]
    ), checkpointer

# 主函数
def main():
    # 创建图和检查点器
    app, checkpointer = create_react_graph()
    
    # 初始状态
    state = get_initial_state()
    thread_id = "user_123"
    
    print("Welcome to the React-style Counter App using LangGraph with Breakpoints!")
    print("Type 'exit' to quit.")
    
    # 初始渲染
    state = app.invoke(
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
            # 执行图 - 会在update_state后暂停
            state = app.invoke(
                state,
                {"configurable": {"thread_id": thread_id}}
            )
            
            # 检查是否在断点处暂停
            if app.is_paused(thread_id):
                print("\n[BREAKPOINT] Execution paused after update_state.")
                print(f"Current count: {checkpointer.get_state(thread_id)['count']}")
                
                # 模拟人机交互 - 允许修改状态
                modify = input("Do you want to modify the count? (y/n): ")
                if modify.lower() == "y":
                    try:
                        new_count = int(input("Enter new count value: "))
                        # 获取当前状态
                        current_state = checkpointer.get_state(thread_id)
                        # 修改状态
                        current_state["count"] = new_count
                        # 更新状态
                        checkpointer.set_state(thread_id, current_state)
                        print(f"Count updated to: {new_count}")
                    except ValueError:
                        print("Invalid input. Continuing without modification.")
                
                # 继续执行
                print("\n[RESUMING] Continuing execution...")
                state = app.continue_from_pause(
                    thread_id,
                    None
                )
            
            # 打印AI响应
            print(state["messages"][-1].content)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()