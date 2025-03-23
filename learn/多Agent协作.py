#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 多Agent协作系统
=========================
本示例讲解LangGraph中的多Agent协作模式:
1. Agent定义与职责划分
2. 消息传递机制
3. 协作决策流程
4. 结果整合与输出

WHY - 设计思路:
1. 复杂任务通常需要不同专业领域的协作才能高质量完成
2. 单一Agent容易产生幻觉并受到能力限制
3. 多Agent系统可以实现责任分离和专业分工
4. 不同角色Agent可以各自专注于特定任务，提高整体效率和质量
5. 模块化设计有助于系统的扩展和维护

HOW - 实现方式:
1. 基于职责划分定义不同的Agent角色
2. 设计Agent间的状态传递机制
3. 利用LangGraph的图结构定义协作流程
4. 配置节点间的边关系控制信息流
5. 实现最终结果的整合输出

WHAT - 功能作用:
通过本示例，你将学习如何在LangGraph中构建多Agent协作系统，
使不同专业Agent协同工作，完成复杂任务。这种协作模式在内容创作、
复杂决策、研究分析等场景非常有价值。

学习目标:
- 理解多Agent系统的设计思路
- 掌握不同角色Agent的定义方法
- 学习Agent间的消息传递机制
- 了解协作决策流程的构建
"""

import os
import json
import time
from typing import TypedDict, Dict, List, Any, Optional, Union
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

# ===========================================================
# 第1部分: 状态定义
# ===========================================================

class TeamState(TypedDict):
    """团队协作状态定义
    
    WHY - 设计思路:
    1. 多Agent系统需要一个共享状态容器传递信息
    2. 不同Agent需要访问各自关心的状态部分
    3. 需要记录完整的协作过程和中间结果
    4. 状态设计应支持灵活的协作流程和条件跳转
    
    HOW - 实现方式:
    1. 使用TypedDict提供类型安全和代码提示
    2. 设计通用字段存储任务信息和元数据
    3. 为每个Agent角色设计专门的字段存储其输出
    4. 设计消息字段记录协作过程中的交流
    
    WHAT - 功能作用:
    提供一个结构化的状态容器，支持不同Agent之间的信息传递，
    记录协作过程中的中间结果和最终输出，为整个多Agent协作
    系统提供数据基础
    """
    task: str  # 当前任务描述
    query: str  # 用户查询
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # 消息历史
    metadata: Dict[str, Any]  # 元数据
    
    # 各Agent的工作区与输出
    research: Optional[Dict[str, Any]]  # 研究员Agent的研究结果
    content: Optional[str]  # 写作Agent的内容
    review: Optional[Dict[str, Any]]  # 编辑Agent的审核意见
    final_content: Optional[str]  # 最终输出内容

# ===========================================================
# 第2部分: Agent定义
# ===========================================================

# 初始化LLM
def get_llm():
    """获取LLM实例
    
    WHY - 设计思路:
    1. 需要一个可复用的LLM获取函数
    2. 便于统一配置和更换底层模型
    
    HOW - 实现方式:
    1. 使用langchain_ollama提供本地LLM能力
    2. 配置合适的参数确保输出质量
    
    WHAT - 功能作用:
    提供一个配置好的LLM实例，供各Agent使用，
    确保所有Agent使用相同的底层模型配置
    """
    return OllamaLLM(
        model="qwen:0.5b",  # 可替换为其他可用模型
        temperature=0.7,
    )

def researcher_agent(state: TeamState) -> Dict:
    """研究员Agent: 负责收集和分析信息
    
    WHY - 设计思路:
    1. 复杂任务需要先进行相关信息的收集与分析
    2. 研究过程需要结构化的思考和组织
    3. 后续Agent需要基于研究结果进行工作
    
    HOW - 实现方式:
    1. 从状态中提取任务和查询信息
    2. 使用LLM生成研究计划并执行
    3. 收集多角度的信息并进行初步分析
    4. 将结果整理为结构化的研究报告
    
    WHAT - 功能作用:
    作为多Agent系统的第一环节，负责信息收集和初步分析，
    为后续环节提供必要的知识基础，确保整个协作过程
    基于充分的信息进行
    
    Args:
        state: 当前团队状态
        
    Returns:
        Dict: 包含研究结果的状态更新
    """
    # 模拟研究员收集信息的过程
    print(f"研究员正在研究: {state['query']}")
    
    llm = get_llm()
    
    # 构建研究提示
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业研究员，负责收集和分析信息。提供全面、准确、多角度的研究结果。"),
        ("user", "请对以下问题进行深入研究，提供关键信息、背景知识和多个观点:\n\n{query}\n\n"
                "要求:\n1. 提供3-5个相关关键事实\n2. 分析至少2个不同视角\n"
                "3. 列出可能的挑战或限制\n\n以JSON格式输出，包含fields: facts, perspectives, challenges")
    ])
    
    # 生成研究结果
    research_response = llm.invoke(
        research_prompt.format(query=state["query"])
    )
    
    # 模拟处理时间
    time.sleep(1.5)
    
    # 尝试解析JSON结果
    try:
        research_result = json.loads(research_response.content)
    except:
        # 如果LLM没有返回有效JSON，创建一个结构化结果
        research_result = {
            "facts": [
                f"研究主题: {state['query']}",
                "这是一个由研究员Agent生成的分析内容",
                "包含了相关的背景信息和多角度分析"
            ],
            "perspectives": [
                {"name": "视角一", "description": "这是对问题的第一种理解视角"},
                {"name": "视角二", "description": "这是提供的另一种不同视角"}
            ],
            "challenges": [
                "可能面临的主要挑战",
                "需要注意的限制因素"
            ]
        }
    
    # 更新状态
    return {
        "research": research_result,
        "messages": state["messages"] + [
            AIMessage(content=f"研究员已完成研究，找到了{len(research_result.get('facts', []))}个关键事实和{len(research_result.get('perspectives', []))}种不同视角。")
        ]
    }

def writer_agent(state: TeamState) -> Dict:
    """写作Agent: 负责根据研究结果创作内容
    
    WHY - 设计思路:
    1. 需要将研究结果转化为连贯、有价值的内容
    2. 写作需要关注目标受众和表达方式
    3. 内容创作需要平衡信息全面性和可读性
    
    HOW - 实现方式:
    1. 从状态中获取研究员的研究结果
    2. 分析研究内容并构建写作框架
    3. 使用LLM根据研究结果生成连贯内容
    4. 确保内容涵盖重要观点并保持逻辑流畅
    
    WHAT - 功能作用:
    作为协作流程的第二环节，将研究成果转化为优质内容，
    处理和组织原始信息，创造出连贯、有价值的输出结果
    
    Args:
        state: 当前团队状态
        
    Returns:
        Dict: 包含写作结果的状态更新
    """
    # 检查研究结果是否存在
    if not state.get("research"):
        return {
            "content": "无法开始写作，缺少研究结果",
            "messages": state["messages"] + [
                AIMessage(content="写作Agent: 无法开始写作，缺少必要的研究资料。")
            ]
        }
    
    # 模拟写作过程
    print("写作Agent正在创作内容...")
    
    research = state["research"]
    llm = get_llm()
    
    # 构建写作提示
    writing_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业内容创作者，负责将研究结果转化为优质内容。注重逻辑性、可读性和信息价值。"),
        ("user", "请根据以下研究资料创作一篇完整、连贯的内容:\n\n"
                "主题: {query}\n\n"
                "研究资料:\n事实: {facts}\n不同视角: {perspectives}\n挑战: {challenges}\n\n"
                "要求:\n1. 内容丰富且逻辑清晰\n2. 包含引人入胜的开头和有力的结论\n"
                "3. 平衡展示不同视角\n4. 文章长度适中，结构合理")
    ])
    
    # 准备提示参数
    prompt_params = {
        "query": state["query"],
        "facts": "\n- ".join([""] + research.get("facts", ["研究资料不完整"])),
        "perspectives": "\n- ".join([""] + [f"{p.get('name', 'unnamed')}: {p.get('description', 'no description')}" 
                                         for p in research.get("perspectives", [{"name": "默认视角", "description": "无详细说明"}])]),
        "challenges": "\n- ".join([""] + research.get("challenges", ["未识别出明确挑战"]))
    }
    
    # 生成内容
    writing_response = llm.invoke(
        writing_prompt.format(**prompt_params)
    )
    
    # 模拟处理时间
    time.sleep(2)
    
    # 提取创作内容
    content = writing_response.content
    
    # 更新状态
    return {
        "content": content,
        "messages": state["messages"] + [
            AIMessage(content=f"写作Agent已完成内容创作，文章约有{len(content.split())}个词。")
        ]
    }

def editor_agent(state: TeamState) -> Dict:
    """编辑Agent: 负责审核和改进内容
    
    WHY - 设计思路:
    1. 内容创作后需要专业审核确保质量
    2. 编辑需要从读者角度评估内容价值
    3. 需要检查事实准确性、结构流畅性和表达清晰度
    4. 最终输出需要经过精炼和优化
    
    HOW - 实现方式:
    1. 从状态中获取写作Agent创作的内容
    2. 检查内容的事实准确性、逻辑性和可读性
    3. 生成详细的审核意见和改进建议
    4. 根据审核意见修改并完善内容
    5. 输出最终版本的高质量内容
    
    WHAT - 功能作用:
    作为协作流程的最后环节，保证输出质量，对内容进行专业审核，
    提出改进建议并完成最终润色，确保交付给用户的是最佳结果
    
    Args:
        state: 当前团队状态
        
    Returns:
        Dict: 包含审核结果和最终内容的状态更新
    """
    # 检查内容是否存在
    if not state.get("content"):
        return {
            "final_content": "无法开始编辑，缺少写作内容",
            "messages": state["messages"] + [
                AIMessage(content="编辑Agent: 无法开始编辑工作，缺少需要审核的内容。")
            ]
        }
    
    # 模拟编辑审核过程
    print("编辑Agent正在审核和改进内容...")
    
    content = state["content"]
    llm = get_llm()
    
    # 构建审核提示
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位资深编辑，负责审核和改进内容。注重事实准确性、逻辑流畅性和语言表达。"),
        ("user", "请审核以下内容，提供评价和改进建议:\n\n"
                "{content}\n\n"
                "请提供:\n1. 总体评价(1-10分)\n2. 优点和亮点\n3. 需改进的地方\n"
                "4. 具体修改建议\n\n以JSON格式输出，字段包括: rating, strengths, improvements, suggestions")
    ])
    
    # 生成审核意见
    review_response = llm.invoke(
        review_prompt.format(content=content)
    )
    
    # 模拟处理时间
    time.sleep(1.5)
    
    # 尝试解析JSON结果
    try:
        review_result = json.loads(review_response.content)
    except:
        # 如果LLM没有返回有效JSON，创建一个结构化结果
        review_result = {
            "rating": 7,
            "strengths": ["内容丰富", "逻辑清晰", "观点全面"],
            "improvements": ["可增加具体例子", "部分表达可以更精炼"],
            "suggestions": ["加强开头吸引力", "调整结构以突出重点"]
        }
    
    # 构建优化提示
    improve_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位优秀的内容完善专家，负责根据编辑反馈优化内容。"),
        ("user", "请根据以下编辑反馈，优化内容:\n\n"
                "原始内容:\n{content}\n\n"
                "编辑评分: {rating}/10\n"
                "优点: {strengths}\n"
                "需改进: {improvements}\n"
                "修改建议: {suggestions}\n\n"
                "请提供优化后的完整内容，保留原文优点的同时解决指出的问题。")
    ])
    
    # 准备提示参数
    prompt_params = {
        "content": content,
        "rating": review_result.get("rating", "无评分"),
        "strengths": "\n- ".join([""] + review_result.get("strengths", ["未指出具体优点"])),
        "improvements": "\n- ".join([""] + review_result.get("improvements", ["未提出改进建议"])),
        "suggestions": "\n- ".join([""] + review_result.get("suggestions", ["无具体修改建议"]))
    }
    
    # 生成优化内容
    improved_response = llm.invoke(
        improve_prompt.format(**prompt_params)
    )
    
    # 模拟处理时间
    time.sleep(1.5)
    
    # 提取优化后的内容
    final_content = improved_response.content
    
    # 更新状态
    return {
        "review": review_result,
        "final_content": final_content,
        "messages": state["messages"] + [
            AIMessage(content=f"编辑Agent已完成审核和改进。评分: {review_result.get('rating', 'N/A')}/10，最终内容已准备就绪。")
        ]
    }

# ===========================================================
# 第3部分: 辅助函数
# ===========================================================

def initialize_state(query: str) -> TeamState:
    """初始化团队状态
    
    WHY - 设计思路:
    1. 需要为多Agent协作系统提供一个干净的初始状态
    2. 初始状态需要包含任务信息和必要的元数据
    3. 需要提供清晰的系统指令引导协作过程
    
    HOW - 实现方式:
    1. 创建包含所有必要字段的TeamState字典
    2. 设置初始系统消息定义团队目标和协作机制
    3. 记录创建时间和任务信息
    
    WHAT - 功能作用:
    为多Agent协作系统提供一个结构化的起点，确保各个Agent
    拥有必要的上下文信息和任务目标，便于后续协作流程的进行
    
    Args:
        query: 用户查询/任务描述
        
    Returns:
        TeamState: 初始化的团队状态
    """
    current_time = datetime.now()
    
    return {
        "task": "协作完成内容创作与优化",
        "query": query,
        "messages": [
            SystemMessage(content="多Agent协作系统启动。研究员、写作员和编辑将依次工作，共同完成内容创作任务。"),
            HumanMessage(content=f"请针对以下主题进行研究并创作高质量内容: {query}")
        ],
        "metadata": {
            "created_at": current_time.isoformat(),
            "team": "研究-写作-编辑团队",
        },
        "research": None,
        "content": None,
        "review": None,
        "final_content": None
    }

def print_final_result(state: TeamState):
    """打印最终结果
    
    WHY - 设计思路:
    1. 需要一个清晰的方式展示多Agent协作的最终成果
    2. 展示应包含完整的协作过程和各个环节的贡献
    3. 用户关注的是最终结果和整体协作质量
    
    HOW - 实现方式:
    1. 提取状态中的最终内容和中间结果
    2. 格式化展示研究结果、创作内容和审核过程
    3. 突出显示最终完成的内容
    
    WHAT - 功能作用:
    提供一个直观的多Agent协作结果展示，便于查看协作成果
    和理解不同Agent的贡献，体现多Agent协作的价值
    
    Args:
        state: 当前团队状态
    """
    print("\n" + "="*50)
    print("多Agent协作结果".center(50))
    print("="*50)
    
    print(f"\n📋 任务: {state['query']}")
    
    # 打印研究结果摘要
    if state.get("research"):
        print("\n🔍 研究员发现:")
        facts = state["research"].get("facts", [])
        for i, fact in enumerate(facts[:3]):
            print(f"  {i+1}. {fact}")
        if len(facts) > 3:
            print(f"  ...等{len(facts)}个事实")
            
        perspectives = state["research"].get("perspectives", [])
        print(f"  🔄 分析了{len(perspectives)}种不同视角")
    
    # 打印编辑评价摘要
    if state.get("review"):
        print("\n✏️ 编辑评价:")
        print(f"  评分: {state['review'].get('rating', 'N/A')}/10")
        print("  优点:", ", ".join(state["review"].get("strengths", ["无详细评价"])[:2]))
        print("  改进:", ", ".join(state["review"].get("improvements", ["无改进建议"])[:2]))
    
    # 打印最终内容
    if state.get("final_content"):
        print("\n📄 最终内容:")
        content = state["final_content"]
        
        # 如果内容太长，只显示部分
        max_length = 500
        if len(content) > max_length:
            content = content[:max_length] + "...\n[内容过长，已截断]"
        
        # 打印内容，添加适当缩进
        for line in content.split("\n"):
            print(f"  {line}")
    else:
        print("\n❌ 未生成最终内容")
    
    print("\n" + "="*50)
    print("协作流程已完成".center(50))
    print("="*50 + "\n")

# ===========================================================
# 第4部分: 图构建
# ===========================================================

def create_team_graph() -> StateGraph:
    """创建团队协作图
    
    WHY - 设计思路:
    1. 需要一个结构化的协作流程管理多Agent工作
    2. 图结构可以明确定义Agent间的依赖和协作顺序
    3. 需要支持线性协作流程或条件分支处理
    
    HOW - 实现方式:
    1. 创建基于TeamState的StateGraph
    2. 添加各个Agent作为图的节点
    3. 设置节点间的边定义协作流程
    4. 最后一个Agent的输出作为整个系统的输出
    
    WHAT - 功能作用:
    提供一个结构化的协作框架，管理不同Agent之间的工作流程，
    确保信息按预定路径流动，各Agent按正确顺序协作完成任务
    
    Returns:
        StateGraph: 编译好的团队协作图
    """
    # 创建图实例
    workflow = StateGraph(TeamState)
    
    # 添加节点 - 每个节点代表一位专家Agent
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("editor", editor_agent)
    
    # 设置协作流程 - 线性流程：研究员 -> 写作员 -> 编辑
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", END)
    
    # 设置入口点
    workflow.set_entry_point("researcher")
    
    # 编译图
    return workflow.compile()

# ===========================================================
# 第5部分: 执行和演示
# ===========================================================

def run_team_collaboration_example():
    """运行团队协作示例
    
    WHY - 设计思路:
    1. 需要通过具体示例展示多Agent协作的完整流程
    2. 示例应涵盖初始化、执行和结果展示的全过程
    3. 需要选择适合展示协作价值的任务类型
    
    HOW - 实现方式:
    1. 创建团队协作图实例
    2. 初始化状态，设置示例任务
    3. 执行协作流程，展示中间步骤
    4. 最后展示完整的协作成果
    
    WHAT - 功能作用:
    提供一个完整的多Agent协作系统示例，帮助理解协作流程和各Agent职责，
    展示如何构建和运行一个完整的多Agent系统
    """
    print("\n===== 多Agent协作系统示例 =====")
    
    # 创建团队协作图
    team_graph = create_team_graph()
    
    # 示例任务选择
    example_tasks = [
        "解释人工智能在现代教育中的应用和未来发展趋势",
        "比较传统能源与可再生能源的优缺点及其对环境的影响",
        "分析远程工作对公司文化和员工生产力的影响",
        "探讨社交媒体对青少年心理健康的正面和负面影响"
    ]
    
    # 选择一个任务
    selected_task = example_tasks[0]
    print(f"\n选定的协作任务: {selected_task}")
    
    # 初始化状态
    state = initialize_state(selected_task)
    
    print("\n开始多Agent协作流程...")
    print("各专家Agent将依次工作，完成研究、写作和编辑任务")
    
    # 执行协作流程
    final_state = team_graph.invoke(state)
    
    # 打印最终结果
    print_final_result(final_state)
    
    return final_state

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行所有协作示例
    2. 需要适当的错误处理确保示例运行稳定
    3. 需要提供清晰的开始和结束提示
    
    HOW - 实现方式:
    1. 使用try-except包装主要执行逻辑，捕获运行异常
    2. 提供明确的开始和结束提示
    3. 调用具体示例函数执行多Agent协作演示
    
    WHAT - 功能作用:
    提供程序入口点，执行多Agent协作示例，确保示例运行平稳，
    增强用户体验和代码可靠性
    """
    print("===== LangGraph 多Agent协作系统示例 =====\n")
    
    try:
        # 运行团队协作示例
        run_team_collaboration_example()
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 设计和实现多Agent协作系统")
        print("2. 定义不同Agent角色及其职责")
        print("3. 配置Agent间的消息传递机制")
        print("4. 构建协作决策流程")
        print("5. 整合和展示协作成果")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 