#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 自主代理系统
===================
本示例讲解如何使用LangGraph构建一个自主代理系统:
1. ReAct循环 - 思考(Reasoning)、行动(Acting)和观察(Observing)的循环
2. 工具调用 - 代理与环境交互的能力
3. 自主决策 - 基于状态和目标的决策逻辑
4. 执行控制 - 任务完成判断和循环终止条件

WHY - 设计思路:
1. 自主代理需要持续感知环境并做出反应
2. 代理决策需要明确的思考过程以实现透明性和可调试性
3. 工具调用需要标准化以便代理与环境进行交互
4. 自主循环需要有明确的终止条件避免无限循环
5. 状态管理需要保持完整的执行历史以便分析和调试

HOW - 实现方式:
1. 使用ReAct模式实现思考-行动-观察循环
2. 定义结构化的工具接口供代理调用
3. 使用LLM实现代理的思考和决策过程
4. 通过循环边实现代理的持续运行
5. 设计状态结构记录代理的完整执行历史

WHAT - 功能作用:
通过本示例，你将学习如何构建一个能够自主思考、做出决策、执行行动
并观察结果的智能代理系统。这类系统可用于任务自动化、问题解决、
信息收集和处理等多种场景。

学习目标:
- 理解ReAct模式在代理系统中的应用
- 掌握自主循环的实现方法
- 学习代理与工具交互的机制
- 理解自主决策过程的实现
"""

from typing import TypedDict, List, Dict, Any, Optional, Tuple, Union
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import re
import datetime
import random
import math

# =================================================================
# 第1部分: 基础组件 - 代理状态和工具定义
# =================================================================

class AgentState(TypedDict):
    """代理系统状态定义
    
    WHY - 设计思路:
    1. 需要完整记录代理的任务、思考、行动和观察
    2. 需要保存代理的执行历史以支持分析和可解释性
    3. 需要管理代理的当前状态和终止条件
    
    HOW - 实现方式:
    1. 使用TypedDict定义类型安全的状态结构
    2. 包含任务描述、当前环境状态和代理状态
    3. 记录思考过程、行动选择和观察结果
    4. 保存完整的执行历史供分析使用
    
    WHAT - 功能作用:
    提供统一的状态结构，跟踪代理的完整执行历史和当前状态，
    使系统能够基于历史和当前状态做出下一步决策
    """
    task: str  # 代理需要完成的任务
    environment: Dict[str, Any]  # 环境状态
    thought: Optional[str]  # 当前思考过程
    action: Optional[Dict[str, Any]]  # 当前选择的行动
    observation: Optional[str]  # 当前观察到的结果
    history: List[Dict[str, Any]]  # 完整的执行历史记录
    finished: bool  # 任务是否完成

# 工具定义
class Tool:
    """工具定义
    
    WHY - 设计思路:
    1. 代理需要标准化的工具接口与环境交互
    2. 工具需要有清晰的描述供代理选择
    3. 工具执行需要有明确的输入输出格式
    
    HOW - 实现方式:
    1. 定义包含名称和描述的工具类
    2. 提供执行方法实现工具功能
    3. 处理工具执行的异常情况
    
    WHAT - 功能作用:
    为代理提供与环境交互的标准接口，实现特定功能
    """
    def __init__(self, name: str, description: str, func: callable):
        self.name = name
        self.description = description
        self.func = func
    
    def execute(self, **kwargs) -> str:
        """执行工具功能"""
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

# 工具实现函数
def search_web(query: str) -> str:
    """模拟网络搜索"""
    # 简单模拟，实际应用中可以集成真实搜索API
    topics = {
        "python": "Python是一种通用编程语言，以简洁、易读的语法著称。它支持多种编程范式，包括面向对象、命令式、函数式和过程式。",
        "机器学习": "机器学习是人工智能的一个分支，它使用算法和统计模型让计算机系统能够逐步改进其执行特定任务的性能，而无需明确编程。",
        "气候变化": "气候变化指地球气候系统的长期变化，包括温度、降水模式和极端天气事件的频率和强度。人类活动是当前气候变化的主要驱动因素。",
        "健康饮食": "健康饮食包括均衡摄入各种食物，如水果、蔬菜、全谷物、蛋白质和健康脂肪，同时限制糖、盐和不健康脂肪的摄入。"
    }
    
    # 检查是否有直接匹配
    for key, value in topics.items():
        if key in query.lower():
            return value
    
    # 没有直接匹配时返回通用回答
    return f"关于'{query}'的搜索结果较为广泛，可能涉及多个领域。请提供更具体的查询内容以获取精确信息。"

def calculate(expression: str) -> str:
    """进行数学计算"""
    try:
        # 安全地评估数学表达式，仅允许基本运算
        # 实际应用中应使用更安全的方法
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return "计算表达式包含不允许的字符"
        
        # 替换除法以防止潜在的零除错误
        expression = expression.replace('/', '//')
        
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

def get_date_time() -> str:
    """获取当前日期和时间"""
    now = datetime.datetime.now()
    return f"当前日期时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def get_weather(location: str) -> str:
    """模拟获取天气信息"""
    # 简单模拟，实际应用中可以集成真实天气API
    weathers = ["晴朗", "多云", "小雨", "阵雨", "大雨", "雷雨", "雪"]
    temps = range(0, 35)
    
    weather = random.choice(weathers)
    temp = random.choice(temps)
    
    return f"{location}的天气: {weather}, 温度: {temp}°C"

# 定义可用工具集
available_tools = [
    Tool("search", "搜索网络获取信息", lambda query: search_web(query)),
    Tool("calculate", "进行数学计算", lambda expression: calculate(expression)),
    Tool("datetime", "获取当前日期和时间", lambda: get_date_time()),
    Tool("weather", "获取指定地点的天气信息", lambda location: get_weather(location))
]

# 工具查找函数
def get_tool_by_name(name: str) -> Optional[Tool]:
    """根据名称查找工具"""
    for tool in available_tools:
        if tool.name == name:
            return tool
    return None

# =================================================================
# 第2部分: 提示模板与解析功能
# =================================================================

# 创建提示模板
system_prompt = """你是一个自主代理系统，能够思考、行动并观察结果。你可以使用以下工具:

{tools}

请遵循以下格式进行思考和决策:

1. 分析当前任务和观察，理解需求
2. 决定下一步行动或得出最终结论
3. 如果需要使用工具，请指定工具名称和参数
4. 如果任务已完成，明确指出结论

请记住，每次只能执行一个行动。
"""

think_template = """
任务: {task}

历史记录:
{history}

当前观察: {observation}

请进行思考，分析当前情况，并确定下一步行动方向。提供详细的思考过程。
"""

action_template = """
基于你的思考:

{thought}

请选择下一步行动。你可以使用以下工具之一:
{tools_description}

如果你认为任务已经完成，可以选择'结束任务'。

以JSON格式返回你的决定，格式如下:
```json
{{
  "action": "工具名称或'finish'",
  "action_input": {{具体参数}} 或 "最终答案"
}}
```
"""

# =================================================================
# 第3部分: 工具函数 - 解析和格式化
# =================================================================

def format_history(history: List[Dict[str, Any]]) -> str:
    """将历史记录格式化为可读文本
    
    WHY - 设计思路:
    1. 历史记录需要易于阅读和理解
    2. 需要包含思考、行动和观察的完整循环
    3. 需要清晰的结构供LLM分析
    
    HOW - 实现方式:
    1. 遍历历史记录中的每个条目
    2. 根据条目类型格式化内容
    3. 按时间顺序拼接为文本
    
    WHAT - 功能作用:
    将结构化的历史记录转换为文本格式，供LLM分析和决策
    
    Args:
        history: 历史记录列表
        
    Returns:
        格式化的历史文本
    """
    if not history:
        return "无历史记录"
    
    formatted = []
    for i, entry in enumerate(history):
        if entry["type"] == "thought":
            formatted.append(f"思考 {i+1}: {entry['content']}")
        elif entry["type"] == "action":
            if entry["content"]["action"] == "finish":
                formatted.append(f"决定: 任务完成，结论是 '{entry['content']['action_input']}'")
            else:
                formatted.append(f"行动 {i+1}: 使用工具 '{entry['content']['action']}' 输入参数: {entry['content']['action_input']}")
        elif entry["type"] == "observation":
            formatted.append(f"观察 {i+1}: {entry['content']}")
    
    return "\n".join(formatted)

def format_tools_description() -> str:
    """格式化工具描述列表
    
    WHY - 设计思路:
    1. 代理需要了解所有可用工具的功能
    2. 工具描述需要清晰易懂
    3. 格式需要统一便于代理解析
    
    HOW - 实现方式:
    1. 遍历所有可用工具
    2. 提取名称和描述
    3. 格式化为一致的列表形式
    
    WHAT - 功能作用:
    生成所有可用工具的描述列表，供代理选择使用
    
    Returns:
        格式化的工具描述文本
    """
    descriptions = []
    for tool in available_tools:
        descriptions.append(f"- {tool.name}: {tool.description}")
    
    return "\n".join(descriptions)

def parse_action(action_text: str) -> Dict[str, Any]:
    """解析LLM生成的行动决策
    
    WHY - 设计思路:
    1. 需要从LLM响应中提取结构化的行动决策
    2. 需要处理各种格式的响应内容
    3. 需要提供错误处理确保系统稳定
    
    HOW - 实现方式:
    1. 尝试查找JSON格式的内容
    2. 解析JSON获取行动和参数
    3. 提供默认值处理解析失败的情况
    
    WHAT - 功能作用:
    将LLM响应转换为结构化的行动决策，供系统执行
    
    Args:
        action_text: LLM生成的行动决策文本
        
    Returns:
        解析后的行动决策字典
    """
    try:
        # 尝试在文本中查找JSON
        json_pattern = r'```json\s*({.*?})\s*```|{.*?}'
        json_match = re.search(json_pattern, action_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            return json.loads(json_str)
        else:
            # 尝试直接解析整个文本
            return json.loads(action_text)
    except (json.JSONDecodeError, AttributeError, IndexError):
        # 解析失败时返回默认值
        return {
            "action": "finish",
            "action_input": "无法解析行动，默认结束任务。原始响应: " + action_text[:100] + "..."
        }

# =================================================================
# 第4部分: LangGraph核心逻辑 - 节点函数
# =================================================================

# 初始化LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

def initialize_agent(state: AgentState, task: str) -> AgentState:
    """初始化代理状态
    
    WHY - 设计思路:
    1. 需要为代理提供明确的任务目标
    2. 需要准备初始环境状态
    3. 需要初始化历史记录空间
    
    HOW - 实现方式:
    1. 设置代理任务
    2. 初始化环境状态
    3. 创建空的执行历史
    
    WHAT - 功能作用:
    准备代理的初始状态，设定任务目标和环境状态，为代理运行做好准备
    
    Args:
        state: 当前代理状态
        task: 代理需要完成的任务
        
    Returns:
        初始化后的代理状态
    """
    # 创建初始化状态
    return {
        "task": task,
        "environment": {
            "time_started": datetime.datetime.now().isoformat(),
            "initial_task": task
        },
        "thought": None,
        "action": None,
        "observation": "刚刚开始任务，还没有观察结果。",
        "history": [],
        "finished": False
    }

def agent_think(state: AgentState) -> AgentState:
    """思考阶段：分析当前状态，生成思考过程
    
    WHY - 设计思路:
    1. 代理需要基于当前状态进行推理
    2. 思考过程需要考虑任务目标和历史记录
    3. 思考结果需要提供下一步行动的基础
    
    HOW - 实现方式:
    1. 整理历史记录和当前观察
    2. 使用LLM生成思考内容
    3. 更新状态并记录思考历史
    
    WHAT - 功能作用:
    分析当前任务状态和历史，生成下一步决策的思考过程
    
    Args:
        state: 当前代理状态
        
    Returns:
        更新后的代理状态，包含思考结果
    """
    task = state["task"]
    observation = state["observation"]
    history = state["history"]
    
    # 格式化历史记录
    history_text = format_history(history)
    
    # 创建思考提示
    think_prompt = ChatPromptTemplate.from_template(think_template)
    think_message = think_prompt.format_messages(
        task=task,
        history=history_text,
        observation=observation
    )
    
    # 使用LLM生成思考
    think_response = llm.invoke(think_message)
    thought = think_response.content
    
    # 更新历史记录
    updated_history = history.copy()
    updated_history.append({"type": "thought", "content": thought})
    
    # 返回更新后的状态
    return {
        **state,
        "thought": thought,
        "history": updated_history
    }

def agent_action(state: AgentState) -> AgentState:
    """行动阶段：基于思考选择下一步行动
    
    WHY - 设计思路:
    1. 代理需要基于思考结果决定具体行动
    2. 行动决策需要考虑可用工具
    3. 需要明确是继续执行工具还是任务完成
    
    HOW - 实现方式:
    1. 提供思考结果和可用工具列表
    2. 使用LLM生成行动决策
    3. 解析决策并更新状态
    
    WHAT - 功能作用:
    基于思考过程做出明确的行动决策，选择使用工具或完成任务
    
    Args:
        state: 当前代理状态
        
    Returns:
        更新后的代理状态，包含行动决策
    """
    thought = state["thought"]
    
    # 准备工具描述
    tools_description = format_tools_description()
    
    # 创建行动提示
    action_prompt = ChatPromptTemplate.from_template(action_template)
    action_message = action_prompt.format_messages(
        thought=thought,
        tools_description=tools_description
    )
    
    # 使用LLM生成行动
    action_response = llm.invoke(action_message)
    
    # 解析行动决策
    action = parse_action(action_response.content)
    
    # 检查是否任务完成
    finished = action["action"] == "finish"
    
    # 更新历史记录
    updated_history = state["history"].copy()
    updated_history.append({"type": "action", "content": action})
    
    # 返回更新后的状态
    return {
        **state,
        "action": action,
        "finished": finished,
        "history": updated_history
    }

def agent_observe(state: AgentState) -> AgentState:
    """观察阶段：执行行动并观察结果
    
    WHY - 设计思路:
    1. 代理需要执行选定的行动并获取结果
    2. 需要处理工具调用和执行过程
    3. 观察结果需要供下一次思考使用
    
    HOW - 实现方式:
    1. 获取选定的行动和参数
    2. 调用相应的工具执行操作
    3. 记录执行结果并更新状态
    
    WHAT - 功能作用:
    执行代理选择的行动，获取环境反馈，为下一次决策提供输入
    
    Args:
        state: 当前代理状态
        
    Returns:
        更新后的代理状态，包含观察结果
    """
    action = state["action"]
    
    # 如果决定完成任务，直接返回最终答案作为观察
    if action["action"] == "finish":
        observation = f"任务完成。最终结论: {action['action_input']}"
    else:
        # 查找并执行工具
        tool_name = action["action"]
        tool_input = action["action_input"]
        
        tool = get_tool_by_name(tool_name)
        if tool:
            # 执行工具
            if isinstance(tool_input, dict):
                observation = tool.execute(**tool_input)
            else:
                observation = tool.execute(tool_input)
        else:
            observation = f"错误: 未找到名为'{tool_name}'的工具"
    
    # 更新历史记录
    updated_history = state["history"].copy()
    updated_history.append({"type": "observation", "content": observation})
    
    # 返回更新后的状态
    return {
        **state,
        "observation": observation,
        "history": updated_history
    }

# =================================================================
# 第5部分: 图构建与流程控制
# =================================================================

def should_continue(state: AgentState) -> Union[str, Tuple[str, str]]:
    """决定是否继续代理循环
    
    WHY - 设计思路:
    1. 代理循环需要明确的终止条件
    2. 需要基于状态判断是继续还是结束
    3. 需要区分正常结束和错误终止
    
    HOW - 实现方式:
    1. 检查任务是否标记为完成
    2. 检查循环次数是否超出限制
    3. 返回下一步节点或结束标记
    
    WHAT - 功能作用:
    控制代理循环的流程，决定是继续执行还是结束任务
    
    Args:
        state: 当前代理状态
        
    Returns:
        下一个要执行的节点名称或结束标记
    """
    # 检查是否完成
    if state["finished"]:
        return END
    
    # 检查循环次数限制(防止无限循环)
    if len(state["history"]) > 30:  # 设置合理的次数限制
        return END
    
    # 继续执行循环
    return "think"

def build_agent_graph() -> StateGraph:
    """构建自主代理系统的工作流图
    
    WHY - 设计思路:
    1. 需要将思考、行动和观察组织为完整的循环
    2. 图结构需要支持条件终止
    3. 需要定义初始化和各节点间的转换
    
    HOW - 实现方式:
    1. 创建基于AgentState的StateGraph
    2. 添加初始化、思考、行动和观察节点
    3. 构建循环结构和条件终止
    4. 设置图的入口点
    
    WHAT - 功能作用:
    组装完整的自主代理系统，定义工作流程和节点间的转换逻辑
    
    Returns:
        配置好的StateGraph实例
    """
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("initialize", initialize_agent)
    workflow.add_node("think", agent_think)
    workflow.add_node("action", agent_action)
    workflow.add_node("observe", agent_observe)
    
    # 设置初始节点
    workflow.set_entry_point("initialize")
    
    # 添加边 - 构建循环
    workflow.add_edge("initialize", "think")
    workflow.add_edge("think", "action")
    workflow.add_edge("action", "observe")
    workflow.add_conditional_edges("observe", should_continue)
    
    return workflow

# =================================================================
# 第6部分: 示例运行与结果展示
# =================================================================

def run_agent_example(task: str, verbose: bool = True):
    """运行自主代理示例
    
    WHY - 设计思路:
    1. 需要一个简单的方式来演示代理功能
    2. 需要初始化代理并设定任务
    3. 需要展示代理的完整执行过程
    
    HOW - 实现方式:
    1. 构建并初始化代理图
    2. 设置任务并调用图执行
    3. 格式化并展示结果
    
    WHAT - 功能作用:
    提供一个便捷的接口运行自主代理系统，并展示完整的执行过程
    
    Args:
        task: 代理需要完成的任务
        verbose: 是否打印详细结果
        
    Returns:
        执行结果
    """
    # 构建工作流图
    agent_graph = build_agent_graph()
    
    # 编译图
    app = agent_graph.compile()
    
    # 设置初始状态和任务
    result = app.invoke({"task": task})
    
    # 打印结果
    if verbose:
        print("\n===== 任务 =====")
        print(result["task"])
        
        print("\n===== 执行历史 =====")
        for i, entry in enumerate(result["history"]):
            entry_type = entry["type"]
            if entry_type == "thought":
                print(f"\n----- 思考 {i//3 + 1} -----")
                print(entry["content"])
            elif entry_type == "action":
                action = entry["content"]
                if action["action"] == "finish":
                    print(f"\n----- 最终决策 -----")
                    print(f"任务完成，结论: {action['action_input']}")
                else:
                    print(f"\n----- 行动 {i//3 + 1} -----")
                    print(f"工具: {action['action']}")
                    print(f"参数: {action['action_input']}")
            elif entry_type == "observation":
                print(f"\n----- 观察 {i//3 + 1} -----")
                print(entry["content"])
        
        print("\n===== 最终状态 =====")
        if result["finished"]:
            print("✓ 任务已完成")
            final_action = next((h["content"] for h in reversed(result["history"]) 
                                if h["type"] == "action" and h["content"]["action"] == "finish"), None)
            if final_action:
                print(f"最终结论: {final_action['action_input']}")
        else:
            print("✗ 任务未完成 (可能达到了最大步数限制)")
    
    return result

def demonstrate_stream_execution(task: str):
    """演示流式执行过程
    
    WHY - 设计思路:
    1. 需要展示代理执行的实时状态变化
    2. 流式输出可以帮助理解代理的工作流程
    3. 需要跟踪关键状态指标
    
    HOW - 实现方式:
    1. 使用LangGraph的stream方法
    2. 跟踪每次状态更新
    3. 打印关键状态指标的变化
    
    WHAT - 功能作用:
    展示自主代理系统执行过程中的状态变化，帮助理解系统工作流程
    
    Args:
        task: 代理需要完成的任务
    """
    # 构建工作流图
    agent_graph = build_agent_graph()
    
    # 设置初始状态
    initial_state = {
        "task": task
    }
    
    print("\n===== 流式执行示例 =====")
    print(f"任务: {task}")
    print("执行过程中的状态变化:")
    
    # 流式执行并跟踪状态变化
    for i, event in enumerate(agent_graph.stream(initial_state)):
        node = event.node
        
        if node == "think":
            print(f"\n步骤 {i}: 思考中...")
        elif node == "action":
            action = event.state.get("action", {})
            if action and action.get("action") == "finish":
                print(f"步骤 {i}: 决定完成任务")
            else:
                action_name = action.get("action", "未知") if action else "未知"
                print(f"步骤 {i}: 选择行动 '{action_name}'")
        elif node == "observe":
            print(f"步骤 {i}: 观察结果")
            observation = event.state.get("observation", "")
            print(f"  {observation[:100]}..." if len(observation) > 100 else f"  {observation}")

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行示例
    2. 需要展示系统在不同任务上的表现
    3. 需要展示不同的运行模式
    
    HOW - 实现方式:
    1. 运行不同的任务示例
    2. 展示完整执行和流式执行两种模式
    3. 提供学习总结
    
    WHAT - 功能作用:
    作为程序入口点，展示自主代理系统的完整功能和不同使用方式
    """
    print("===== LangGraph 自主代理系统学习示例 =====\n")
    
    try:
        # 示例1: 信息查询任务
        print("\n示例1: 信息查询任务")
        run_agent_example("查询Python编程语言的基本信息，并计算5+7*3的结果")
        
        # 示例2: 流式执行示例
        print("\n示例2: 决策任务流式执行")
        demonstrate_stream_execution("获取今天的日期时间，然后查询北京的天气")
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 使用LangGraph构建ReAct模式的自主代理系统")
        print("2. 实现思考-行动-观察的循环决策流程")
        print("3. 集成工具让代理能够与环境交互")
        print("4. 构建完整的状态跟踪和历史记录机制")
        print("5. 使用条件路由实现任务完成判断")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 