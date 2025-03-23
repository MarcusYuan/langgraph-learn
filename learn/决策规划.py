#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 决策与规划系统
===================
本示例讲解如何使用LangGraph构建一个决策与规划系统:
1. 目标分解 - 将复杂目标拆解为可执行步骤
2. 计划生成 - 使用LLM生成详细的执行计划
3. 计划执行 - 逐步执行计划中的各个步骤
4. 结果评估 - 评估执行结果并提供调整建议

WHY - 设计思路:
1. 复杂任务需要分解为可执行的小步骤才能有效完成
2. 自动化系统需要能够自主生成执行计划
3. 计划执行过程需要跟踪和管理
4. 执行结果需要评估以便改进和调整
5. 整个过程应形成闭环，从规划到执行到评估

HOW - 实现方式:
1. 使用TypedDict定义规划状态结构
2. 利用LLM生成任务执行计划
3. 通过状态机模式管理计划执行流程
4. 实现评估和总结机制提供反馈
5. 使用条件路由实现动态流程控制

WHAT - 功能作用:
通过本示例，你将学习如何构建一个能够分解目标、制定计划、
执行任务、评估结果并给出改进建议的决策规划系统。
这类系统在项目管理、自动化工作流、智能助手等领域有广泛应用。

学习目标:
- 理解状态设计在复杂系统中的应用
- 掌握基于LLM的计划生成方法
- 学习状态机驱动的流程控制技术
- 理解如何实现系统自评估和优化
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import re

# =================================================================
# 第1部分: 基础组件 - 规划状态和模板定义
# =================================================================

class PlanningState(TypedDict):
    """规划系统状态定义
    
    WHY - 设计思路:
    1. 需要一个结构化的状态来跟踪整个规划和执行过程
    2. 状态需要包含任务、计划、执行进度和结果等关键信息
    3. 需要支持评估和调整的状态字段
    
    HOW - 实现方式:
    1. 使用TypedDict定义类型安全的状态结构
    2. 包含任务描述、计划步骤、当前执行状态等字段
    3. 添加评估结果和最终输出字段
    
    WHAT - 功能作用:
    提供一个统一的状态结构，用于跟踪和管理整个决策规划流程，
    确保系统各组件之间能够一致地传递和处理状态信息
    """
    task: str  # 要完成的任务
    plan: Optional[List[Dict[str, str]]]  # 计划步骤列表
    current_step_index: Optional[int]  # 当前执行的步骤索引
    current_step_result: Optional[str]  # 当前步骤执行结果
    status: Optional[str]  # 任务状态：planning, executing, evaluating, completed, failed
    execution_results: Optional[List[Dict[str, str]]]  # 执行结果列表
    evaluation: Optional[Dict[str, Any]]  # 评估结果
    adjustments: Optional[List[str]]  # 调整建议列表
    final_result: Optional[str]  # 最终执行结果

# 初始化LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 创建提示模板
plan_template = """
你是一个精通任务规划的AI助手。请为以下任务制定详细的步骤计划:

任务: {task}

请提供一个由5-7个步骤组成的计划。每个步骤应包含一个标题和详细描述。
将你的回答格式化为JSON，格式如下:
[
  {{"step": "步骤1标题", "description": "步骤1的详细描述"}},
  {{"step": "步骤2标题", "description": "步骤2的详细描述"}},
  ...
]
"""

execute_template = """
请执行以下任务计划的步骤:

任务: {task}
当前步骤: {current_step}
步骤描述: {step_description}

请描述执行这个步骤的结果。保持简洁但要包含关键信息。
"""

evaluation_template = """
请评估任务执行的结果:

任务: {task}
原始计划:
{original_plan}

执行结果:
{execution_results}

请评估计划执行的成功程度，并提出可能的改进建议。
将你的回答格式化为JSON，格式如下:
{{
  "success_rate": 0-100之间的数字,
  "strengths": ["优点1", "优点2", ...],
  "weaknesses": ["缺点1", "缺点2", ...],
  "suggested_adjustments": ["调整建议1", "调整建议2", ...]
}}
"""

summarize_template = """
请总结以下任务的执行情况:

任务: {task}
计划:
{plan}

执行结果:
{execution_results}

评估:
{evaluation}

调整建议:
{adjustments}

请提供一个简洁的总结，描述任务完成情况、主要成果和经验教训。
"""

# =================================================================
# 第2部分: 工具函数 - 解析和格式化
# =================================================================

def parse_plan_steps(plan_text: str) -> List[Dict[str, str]]:
    """将LLM生成的计划文本解析为结构化的步骤列表
    
    WHY - 设计思路:
    1. LLM生成的计划文本需要结构化处理
    2. 需要处理不同格式的输出和可能的解析错误
    3. 结果需要符合系统预期的步骤结构
    
    HOW - 实现方式:
    1. 尝试直接解析JSON格式的计划
    2. 如果失败，使用正则表达式提取步骤信息
    3. 提供兜底机制处理无法解析的情况
    
    WHAT - 功能作用:
    将LLM生成的文本转换为结构化的步骤列表，便于系统后续处理和执行
    
    Args:
        plan_text: LLM生成的计划文本
        
    Returns:
        结构化的步骤列表
    """
    # 尝试直接解析JSON
    try:
        # 查找文本中的JSON数组
        json_match = re.search(r'\[\s*{.*}\s*\]', plan_text, re.DOTALL)
        if json_match:
            plan_json = json_match.group(0)
            return json.loads(plan_json)
        
        # 如果没有匹配到JSON数组格式，尝试直接解析整个文本
        return json.loads(plan_text)
    except json.JSONDecodeError:
        # 如果JSON解析失败，使用正则表达式提取步骤
        steps = []
        step_pattern = r'步骤\s*(\d+)[:：]?\s*(.*?)(?=步骤\s*\d+|$)'
        matches = re.findall(step_pattern, plan_text, re.DOTALL)
        
        for i, (_, step_text) in enumerate(matches):
            steps.append({
                "step": f"步骤{i+1}",
                "description": step_text.strip()
            })
        
        return steps if steps else [{"step": "未能解析步骤", "description": plan_text}]

# =================================================================
# 第3部分: LangGraph核心逻辑 - 节点函数
# =================================================================

def plan_generation(state: PlanningState) -> PlanningState:
    """生成解决问题的步骤计划
    
    WHY - 设计思路:
    1. 系统需要根据任务自动生成执行计划
    2. 计划应该结构化并易于执行
    3. 需要初始化计划执行的状态
    
    HOW - 实现方式:
    1. 使用提示模板引导LLM生成计划
    2. 解析LLM响应为结构化步骤
    3. 更新状态以启动执行流程
    
    WHAT - 功能作用:
    接收任务描述，生成详细的执行计划，并准备系统开始执行计划
    
    Args:
        state: 当前规划状态
        
    Returns:
        更新后的规划状态，包含执行计划
    """
    task = state["task"]
    
    # 使用LLM生成计划
    plan_prompt = ChatPromptTemplate.from_template(plan_template)
    plan_message = plan_prompt.format_messages(task=task)
    plan_response = llm.invoke(plan_message)
    
    # 解析计划步骤
    steps = parse_plan_steps(plan_response.content)
    
    return {
        **state,
        "plan": steps, 
        "current_step_index": 0,
        "status": "planning",
        "execution_results": []
    }

def execute_step(state: PlanningState) -> PlanningState:
    """执行当前计划步骤
    
    WHY - 设计思路:
    1. 需要逐步执行计划中的每个步骤
    2. 需要记录每个步骤的执行结果
    3. 需要跟踪执行进度
    
    HOW - 实现方式:
    1. 获取当前需要执行的步骤
    2. 使用LLM模拟步骤执行过程
    3. 记录执行结果并更新执行进度
    
    WHAT - 功能作用:
    执行计划中的当前步骤，记录执行结果，并更新执行状态
    
    Args:
        state: 当前规划状态
        
    Returns:
        更新后的规划状态，包含执行结果
    """
    task = state["task"]
    plan = state["plan"]
    current_index = state["current_step_index"]
    
    # 检查是否所有步骤已执行完成
    if current_index >= len(plan):
        return {
            **state,
            "status": "evaluating"
        }
            
    # 获取当前步骤
    current_step = plan[current_index]
    
    # 使用LLM模拟执行步骤
    execute_prompt = ChatPromptTemplate.from_template(execute_template)
    execute_message = execute_prompt.format_messages(
        task=task,
        current_step=current_step["step"],
        step_description=current_step["description"]
    )
    execute_response = llm.invoke(execute_message)
    
    # 更新执行结果
    execution_results = state.get("execution_results", [])
    execution_results.append({
        "step": current_step["step"],
        "result": execute_response.content
    })
    
    return {
        **state,
        "current_step_result": execute_response.content,
        "current_step_index": current_index + 1,
        "execution_results": execution_results,
        "status": "executing"
    }

def evaluate_results(state: PlanningState) -> PlanningState:
    """评估执行结果
    
    WHY - 设计思路:
    1. 完成执行后需要评估整体效果
    2. 需要识别计划的优缺点
    3. 需要生成改进建议
    
    HOW - 实现方式:
    1. 整理计划和执行结果
    2. 使用LLM评估执行情况
    3. 解析评估结果并保存调整建议
    
    WHAT - 功能作用:
    分析执行结果，评估成功程度，识别优缺点，并提出改进建议
    
    Args:
        state: 当前规划状态
        
    Returns:
        更新后的规划状态，包含评估结果
    """
    task = state["task"]
    plan = state["plan"]
    execution_results = state.get("execution_results", [])
    
    # 格式化计划和执行结果为文本
    plan_text = "\n".join([f"{i+1}. {step['step']}: {step['description']}" 
                         for i, step in enumerate(plan)])
    
    results_text = "\n".join([f"{i+1}. {result['step']}: {result['result']}" 
                            for i, result in enumerate(execution_results)])
    
    # 使用LLM评估结果
    eval_prompt = ChatPromptTemplate.from_template(evaluation_template)
    eval_message = eval_prompt.format_messages(
        task=task,
        original_plan=plan_text,
        execution_results=results_text
    )
    eval_response = llm.invoke(eval_message)
    
    # 解析评估结果
    try:
        # 尝试找到JSON格式的评估结果
        json_match = re.search(r'{.*}', eval_response.content, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group(0))
        else:
            evaluation = json.loads(eval_response.content)
    except (json.JSONDecodeError, AttributeError):
        # 如果解析失败，创建简单的评估结果
        evaluation = {
            "success_rate": 50,
            "strengths": ["执行了计划"],
            "weaknesses": ["未能正确解析评估结果"],
            "suggested_adjustments": ["重新评估"]
        }
    
    # 提取调整建议
    adjustments = evaluation.get("suggested_adjustments", [])
    
    return {
        **state,
        "evaluation": evaluation,
        "adjustments": adjustments,
        "status": "evaluated"
    }

def summarize_task(state: PlanningState) -> PlanningState:
    """总结任务执行情况
    
    WHY - 设计思路:
    1. 需要对整个任务执行过程进行总结
    2. 总结应包含计划、执行、评估的完整信息
    3. 需要提炼关键经验和教训
    
    HOW - 实现方式:
    1. 整合计划、执行结果、评估和调整建议
    2. 使用LLM生成综合总结
    3. 将任务标记为已完成
    
    WHAT - 功能作用:
    生成任务执行的综合总结，提炼关键点和经验教训，完成整个规划循环
    
    Args:
        state: 当前规划状态
        
    Returns:
        更新后的规划状态，包含最终总结
    """
    task = state["task"]
    plan = state["plan"]
    execution_results = state.get("execution_results", [])
    evaluation = state.get("evaluation", {})
    adjustments = state.get("adjustments", [])
    
    # 格式化计划和执行结果为文本
    plan_text = "\n".join([f"{i+1}. {step['step']}: {step['description']}" 
                         for i, step in enumerate(plan)])
    
    results_text = "\n".join([f"{i+1}. {result['step']}: {result['result']}" 
                            for i, result in enumerate(execution_results)])
    
    # 格式化评估和调整为文本
    eval_text = f"成功率: {evaluation.get('success_rate', 'N/A')}%\n"
    eval_text += f"优点: {', '.join(evaluation.get('strengths', []))}\n"
    eval_text += f"缺点: {', '.join(evaluation.get('weaknesses', []))}"
    
    adjust_text = "\n".join([f"- {adj}" for adj in adjustments])
    
    # 使用LLM生成总结
    summary_prompt = ChatPromptTemplate.from_template(summarize_template)
    summary_message = summary_prompt.format_messages(
        task=task,
        plan=plan_text,
        execution_results=results_text,
        evaluation=eval_text,
        adjustments=adjust_text
    )
    summary_response = llm.invoke(summary_message)
    
    return {
        **state,
        "final_result": summary_response.content,
        "status": "completed"
    }

# =================================================================
# 第4部分: 图构建与流程控制
# =================================================================

def router(state: PlanningState) -> str:
    """决定下一步操作的路由函数
    
    WHY - 设计思路:
    1. 需要基于当前状态动态决定下一步操作
    2. 流程控制应该遵循规划-执行-评估-总结的逻辑
    3. 执行步骤需要循环直到完成所有计划
    
    HOW - 实现方式:
    1. 检查状态中的状态标记
    2. 根据不同状态返回不同的下一步节点
    3. 处理执行循环和流程终止条件
    
    WHAT - 功能作用:
    控制系统的工作流程，确保按照正确的顺序执行规划、执行、评估和总结
    
    Args:
        state: 当前规划状态
        
    Returns:
        下一步要执行的节点名称
    """
    status = state.get("status", "")
    
    if status == "planning":
        return "execute_step"
    elif status == "executing":
        # 如果还有步骤需要执行，继续执行
        if state["current_step_index"] < len(state["plan"]):
            return "execute_step"
        else:
            return "evaluate_results"
    elif status == "evaluating" or status == "evaluated":
        return "summarize_task"
    elif status == "completed":
        return END
    else:
        # 默认从计划开始
        return "plan_generation"

def build_planning_graph() -> StateGraph:
    """构建决策规划系统的工作流图
    
    WHY - 设计思路:
    1. 需要将各节点组织成完整的工作流
    2. 图结构需要支持条件路由和状态传递
    3. 需要定义明确的入口点和流程
    
    HOW - 实现方式:
    1. 创建基于PlanningState的StateGraph
    2. 添加计划、执行、评估和总结节点
    3. 使用条件边实现动态流程控制
    4. 设置图的入口点
    
    WHAT - 功能作用:
    组装完整的决策规划系统，定义工作流程和节点间的转换逻辑
    
    Returns:
        配置好的StateGraph实例
    """
    # 创建状态图
    workflow = StateGraph(PlanningState)
    
    # 添加节点
    workflow.add_node("plan_generation", plan_generation)
    workflow.add_node("execute_step", execute_step)
    workflow.add_node("evaluate_results", evaluate_results)
    workflow.add_node("summarize_task", summarize_task)
    
    # 设置起始节点
    workflow.set_entry_point("plan_generation")
    
    # 添加边 - 使用路由器函数决定下一步
    workflow.add_conditional_edges("plan_generation", router)
    workflow.add_conditional_edges("execute_step", router)
    workflow.add_conditional_edges("evaluate_results", router)
    workflow.add_conditional_edges("summarize_task", router)
    
    return workflow

# =================================================================
# 第5部分: 示例运行与结果展示
# =================================================================

def run_planning_example(task: str, verbose: bool = True):
    """运行决策规划示例
    
    WHY - 设计思路:
    1. 需要一个简单的方式来演示系统功能
    2. 需要展示系统的输入输出和状态变化
    3. 不同的任务应能得到不同的规划和执行结果
    
    HOW - 实现方式:
    1. 构建并初始化规划图
    2. 设置初始任务并调用图执行
    3. 格式化并展示结果
    
    WHAT - 功能作用:
    提供一个便捷的接口运行决策规划系统，并展示结果
    
    Args:
        task: 要规划的任务描述
        verbose: 是否打印详细结果
        
    Returns:
        执行结果
    """
    # 构建工作流图
    planning_graph = build_planning_graph()
    
    # 编译图
    app = planning_graph.compile()
    
    # 设置初始状态
    initial_state = {
        "task": task
    }
    
    # 执行图并获取结果
    result = app.invoke(initial_state)
    
    # 打印结果
    if verbose:
        print("\n===== 任务 =====")
        print(result["task"])
        
        print("\n===== 计划 =====")
        for i, step in enumerate(result["plan"]):
            print(f"{i+1}. {step['step']}: {step['description']}")
        
        print("\n===== 执行结果 =====")
        for exec_result in result["execution_results"]:
            print(f"- {exec_result['step']}:")
            print(f"  {exec_result['result']}")
        
        print("\n===== 评估 =====")
        eval_result = result["evaluation"]
        print(f"成功率: {eval_result.get('success_rate')}%")
        print("优点:")
        for strength in eval_result.get("strengths", []):
            print(f"- {strength}")
        print("缺点:")
        for weakness in eval_result.get("weaknesses", []):
            print(f"- {weakness}")
        
        print("\n===== 调整建议 =====")
        for adjustment in result["adjustments"]:
            print(f"- {adjustment}")
        
        print("\n===== 最终总结 =====")
        print(result["final_result"])
    
    return result

def demonstrate_stream_execution(task: str):
    """演示流式执行过程
    
    WHY - 设计思路:
    1. 需要展示系统执行的实时状态变化
    2. 流式输出可以帮助理解系统的工作流程
    3. 需要跟踪关键状态指标
    
    HOW - 实现方式:
    1. 使用LangGraph的stream方法
    2. 跟踪每次状态更新
    3. 打印关键状态指标的变化
    
    WHAT - 功能作用:
    展示决策规划系统执行过程中的状态变化，帮助理解系统工作流程
    
    Args:
        task: 要规划的任务描述
    """
    # 构建工作流图
    planning_graph = build_planning_graph()
    
    # 设置初始状态
    initial_state = {
        "task": task
    }
    
    print("\n===== 流式执行示例 =====")
    print(f"任务: {task}")
    print("执行过程中的状态变化:")
    
    # 流式执行并跟踪状态变化
    for event in planning_graph.stream(initial_state):
        status = event.state.get("status", "未开始")
        step_index = event.state.get("current_step_index", 0)
        plan_length = len(event.state.get("plan", [])) if event.state.get("plan") else 0
        
        print(f"节点: {event.node} | 状态: {status} | 进度: {step_index}/{plan_length}")

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行示例
    2. 需要展示系统在不同任务上的表现
    3. 需要展示不同的运行模式
    
    HOW - 实现方式:
    1. 运行常规示例展示完整结果
    2. 运行流式示例展示执行过程
    3. 使用不同的任务示例系统的通用性
    
    WHAT - 功能作用:
    作为程序入口点，展示决策规划系统的完整功能和不同使用方式
    """
    print("===== LangGraph 决策与规划系统学习示例 =====\n")
    
    try:
        # 示例1: 完整规划示例
        print("\n示例1: 组织Python编程工作坊")
        run_planning_example("为初学者组织一场Python编程工作坊")
        
        # 示例2: 流式执行示例
        print("\n示例2: 组织社区清洁活动")
        demonstrate_stream_execution("组织一个小型社区清洁活动")
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 使用LangGraph构建决策规划系统")
        print("2. 实现目标分解和计划生成")
        print("3. 管理计划执行和状态追踪")
        print("4. 评估执行结果并提供改进建议")
        print("5. 使用条件路由实现动态流程控制")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 