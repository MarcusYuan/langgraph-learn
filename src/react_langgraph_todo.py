from typing import Dict, List, Optional, TypedDict, Literal, Any, Union, Callable
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
import json

# 定义状态类型
class TodoItem(TypedDict):
    id: str
    text: str
    completed: bool

class ReactComponentState(TypedDict):
    # 消息历史
    messages: List[BaseMessage]
    # 组件状态
    todos: List[TodoItem]
    # 临时状态（用于存储用户输入等）
    input_value: str
    # 当前活动的副作用
    effects: List[Dict[str, Any]]
    # 渲染结果
    ui: str
    # 当前动作
    action: Optional[str]
    # 动作参数
    action_params: Optional[Dict[str, Any]]

# 初始状态
def get_initial_state() -> ReactComponentState:
    return {
        "messages": [],
        "todos": [],
        "input_value": "",
        "effects": [],
        "ui": "",
        "action": None,
        "action_params": None
    }

# 模拟React的useState钩子
def use_state(state: ReactComponentState) -> Dict:
    """处理状态更新"""
    # 这里我们只返回当前状态，实际更新在action_handler中处理
    return {"todos": state["todos"], "input_value": state["input_value"]}

# 模拟React的useEffect钩子
def use_effect(state: ReactComponentState) -> Dict:
    """处理副作用"""
    effects = state["effects"]
    messages = state["messages"]
    
    # 处理副作用（例如，记录状态变化）
    if effects:
        for effect in effects:
            if effect["type"] == "log":
                print(f"[Log Effect]: {effect['message']}")
            elif effect["type"] == "save":
                print(f"[Save Effect]: Saving todos to storage")
                # 这里可以实现实际的存储逻辑
    
    # 清除已处理的副作用
    return {"effects": []}

# 渲染UI
def render(state: ReactComponentState) -> Dict:
    """渲染组件UI"""
    todos = state["todos"]
    input_value = state["input_value"]
    
    # 构建UI表示
    ui = "===== Todo App =====\n\n"
    
    # 显示待办事项列表
    if todos:
        ui += "Todo Items:\n"
        for i, todo in enumerate(todos):
            status = "[x]" if todo["completed"] else "[ ]"
            ui += f"{i+1}. {status} {todo['text']}\n"
    else:
        ui += "No todos yet. Add some!\n"
    
    # 显示输入框
    ui += f"\nCurrent input: {input_value}\n"
    
    # 显示可用操作
    ui += "\nAvailable actions:\n"
    ui += "1. add - Add a new todo\n"
    ui += "2. toggle <id> - Toggle completion status\n"
    ui += "3. delete <id> - Delete a todo\n"
    ui += "4. input <text> - Set input text\n"
    ui += "5. clear - Clear all todos\n"
    
    return {"ui": ui}

# 解析用户输入
def parse_user_input(state: ReactComponentState) -> Dict:
    """解析用户输入并设置相应的动作"""
    messages = state["messages"]
    
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {}
    
    user_input = messages[-1].content.strip()
    parts = user_input.split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    
    action = None
    action_params = {}
    
    if command == "add":
        action = "add_todo"
        action_params = {"text": state["input_value"]}
    elif command == "toggle" and len(parts) > 1:
        try:
            todo_id = int(parts[1]) - 1  # 转换为0-based索引
            action = "toggle_todo"
            action_params = {"id": todo_id}
        except ValueError:
            pass
    elif command == "delete" and len(parts) > 1:
        try:
            todo_id = int(parts[1]) - 1  # 转换为0-based索引
            action = "delete_todo"
            action_params = {"id": todo_id}
        except ValueError:
            pass
    elif command == "input" and len(parts) > 1:
        action = "set_input"
        action_params = {"value": parts[1]}
    elif command == "clear":
        action = "clear_todos"
    
    return {
        "action": action,
        "action_params": action_params
    }

# 处理动作
def action_handler(state: ReactComponentState) -> Dict:
    """处理用户动作并更新状态"""
    action = state["action"]
    params = state["action_params"]
    
    if not action:
        return {}
    
    todos = state["todos"].copy()
    input_value = state["input_value"]
    effects = state["effects"].copy() if state["effects"] else []
    
    if action == "add_todo" and input_value.strip():
        # 添加新待办事项
        new_todo = {
            "id": str(len(todos) + 1),
            "text": input_value,
            "completed": False
        }
        todos.append(new_todo)
        input_value = ""  # 清空输入
        effects.append({"type": "log", "message": f"Added todo: {new_todo['text']}"})  
    
    elif action == "toggle_todo" and "id" in params:
        # 切换待办事项状态
        todo_id = params["id"]
        if 0 <= todo_id < len(todos):
            todos[todo_id]["completed"] = not todos[todo_id]["completed"]
            status = "completed" if todos[todo_id]["completed"] else "active"
            effects.append({"type": "log", "message": f"Toggled todo {todo_id+1} to {status}"})
    
    elif action == "delete_todo" and "id" in params:
        # 删除待办事项
        todo_id = params["id"]
        if 0 <= todo_id < len(todos):
            deleted_text = todos[todo_id]["text"]
            todos.pop(todo_id)
            effects.append({"type": "log", "message": f"Deleted todo: {deleted_text}"})
    
    elif action == "set_input" and "value" in params:
        # 设置输入值
        input_value = params["value"]
    
    elif action == "clear_todos":
        # 清空所有待办事项
        if todos:
            effects.append({"type": "log", "message": f"Cleared {len(todos)} todos"})
            todos = []
    
    # 如果待办事项发生变化，添加保存副作用
    if action in ["add_todo", "toggle_todo", "delete_todo", "clear_todos"]:
        effects.append({"type": "save"})
    
    return {
        "todos": todos,
        "input_value": input_value,
        "effects": effects,
        "action": None,  # 重置动作
        "action_params": None  # 重置参数
    }

# 生成AI响应
def generate_response(state: ReactComponentState) -> Dict:
    """生成AI响应消息"""
    ui = state["ui"]
    action = state["action"]
    
    # 构建响应消息
    response = ui
    
    if action is None:
        response += "\n\nWhat would you like to do?"
    
    messages = state["messages"].copy()
    messages.append(AIMessage(content=response))
    
    return {"messages": messages}

# 决定下一步
def should_continue(state: ReactComponentState) -> Literal["continue", "end"]:
    """决定是否继续执行循环"""
    # 在实际React中，这相当于检查是否需要重新渲染
    # 这里我们简单地总是返回end，因为我们已经完成了一个渲染循环
    return "end"

# 创建图
def create_react_graph():
    # 创建状态图
    graph = StateGraph(ReactComponentState)
    
    # 添加节点
    graph.add_node("parse_input", parse_user_input)
    graph.add_node("use_state", use_state)
    graph.add_node("action_handler", action_handler)
    graph.add_node("use_effect", use_effect)
    graph.add_node("render", render)
    graph.add_node("generate_response", generate_response)
    
    # 添加边 - 模拟React的渲染循环
    graph.add_edge(START, "parse_input")
    graph.add_edge("parse_input", "use_state")
    graph.add_edge("use_state", "action_handler")
    graph.add_edge("action_handler", "use_effect")
    graph.add_edge("use_effect", "render")
    graph.add_edge("render", "generate_response")
    graph.add_edge("generate_response", END)
    
    # 编译图
    return graph.compile()

# 主函数
def main():
    # 创建图
    app = create_react_graph()
    
    # 初始状态
    state = get_initial_state()
    
    print("Welcome to the React-style Todo App using LangGraph!")
    print("Type 'exit' to quit.")
    
    while True:
        # 获取用户输入
        user_input = input("\n> ")
        
        if user_input.lower() == "exit":
            break
        
        # 更新消息
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        
        # 执行图
        state = app.invoke(state)
        
        # 打印AI响应
        print(state["messages"][-1].content)

if __name__ == "__main__":
    main()