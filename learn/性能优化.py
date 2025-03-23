"""
LangGraph 性能优化与扩展示例
======================
本示例展示如何优化LangGraph应用性能，包括:
1. 并行执行 - 使用异步处理同时执行多个任务
2. 缓存策略 - 使用LRU缓存避免重复计算
3. 分布式部署 - 使用Ray进行分布式计算

WHY - 设计思路:
1. LangGraph应用在处理复杂任务时可能面临性能瓶颈
2. 需要优化计算资源利用，提升响应速度
3. 需要处理可能的重复计算问题
4. 大规模应用需要分布式计算支持
5. 需要客观对比不同优化策略的效果

HOW - 实现方式:
1. 利用asyncio实现并行任务处理
2. 使用Python内置lru_cache实现缓存
3. 集成Ray框架实现分布式计算
4. 设计智能路由机制自动选择最佳处理策略
5. 对比不同处理方式的性能差异

WHAT - 功能作用:
通过本示例，你将学习如何优化LangGraph应用的性能，
掌握并行处理、缓存优化和分布式部署的实现方法，
理解如何针对不同场景选择合适的优化策略，
以及如何评估不同优化方法的效果。

学习目标:
- 掌握LangGraph中的异步并行处理
- 了解如何实现和使用缓存策略
- 学习Ray分布式框架与LangGraph的集成
- 理解性能优化的评估和比较方法
"""

import os
import time
import json
import asyncio
from functools import lru_cache
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import langgraph.checkpoint.memory as memory_checkpoint

# 如果有Ray，则导入Ray相关库
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("未安装Ray库，分布式示例将被禁用。可以通过pip install ray安装。")

# 使用Ollama作为本地LLM
from langchain_community.llms import Ollama

# 初始化LLM
llm = Ollama(model="qwen:0.5b")

# =================================================================
# 第1部分: 基础组件 - 状态定义与工具函数
# =================================================================

class ProcessingState(TypedDict):
    """处理状态定义
    
    WHY - 设计思路:
    1. 需要定义统一的状态结构来支持不同的处理方式
    2. 需要存储任务列表、处理结果和性能统计信息
    3. 需要支持消息传递和最终响应的生成
    
    HOW - 实现方式:
    1. 使用TypedDict定义类型安全的状态结构
    2. 包含消息历史、任务列表、处理结果等字段
    3. 添加性能计时字段用于性能对比
    
    WHAT - 功能作用:
    提供一个统一的状态接口，使不同的处理节点可以共享
    数据并追踪处理性能
    """
    messages: List[HumanMessage | AIMessage]
    tasks: List[Dict[str, Any]]
    task_results: List[Dict[str, Any]]
    final_response: Optional[str]
    timing_info: Dict[str, float]

# =================================================================
# 第2部分: 并行执行实现
# =================================================================

async def process_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """模拟一个耗时任务的处理
    
    WHY - 设计思路:
    1. 需要模拟现实中的耗时操作
    2. 不同任务可能有不同的处理时间
    3. 需要返回结构化的处理结果
    
    HOW - 实现方式:
    1. 使用任务的type属性决定处理时间
    2. 使用asyncio.sleep模拟处理延迟
    3. 返回包含任务信息和结果的字典
    
    WHAT - 功能作用:
    模拟一个异步处理任务，根据任务类型有不同的执行时间，
    用于测试并行和串行处理的性能差异
    
    Args:
        task: 包含任务信息的字典
        
    Returns:
        Dict[str, Any]: 任务处理结果
    """
    # 模拟不同任务有不同的处理时间
    task_type = task.get("type", "default")
    if task_type == "heavy":
        delay = 2.0
    elif task_type == "medium":
        delay = 1.0
    else:
        delay = 0.5
        
    print(f"处理任务: {task['name']} (类型: {task_type}, 延迟: {delay}秒)")
    await asyncio.sleep(delay)  # 模拟处理时间
    
    # 返回处理结果
    return {
        "task_id": task.get("id"),
        "task_name": task.get("name"),
        "result": f"完成任务 '{task.get('name')}' 的处理",
        "processed_at": datetime.now().isoformat()
    }

async def parallel_processing(state: ProcessingState) -> Dict[str, Any]:
    """并行处理多个任务
    
    WHY - 设计思路:
    1. 需要同时处理多个独立任务以提高效率
    2. 需要准确计算并行处理的性能提升
    3. 需要保持状态的不变性
    
    HOW - 实现方式:
    1. 使用asyncio.gather并行执行多个异步任务
    2. 记录开始和结束时间以计算性能指标
    3. 返回处理结果和时间信息
    
    WHAT - 功能作用:
    作为LangGraph图的节点，同时处理多个任务，利用异步IO
    提高处理效率，并记录性能数据
    
    Args:
        state: 当前处理状态
        
    Returns:
        Dict[str, Any]: 包含处理结果和性能数据的状态更新
    """
    start_time = time.time()
    tasks = state["tasks"]
    
    print(f"开始并行处理 {len(tasks)} 个任务...")
    
    # 创建异步任务
    async_tasks = [process_task(task) for task in tasks]
    
    # 并行执行所有任务
    results = await asyncio.gather(*async_tasks)
    
    end_time = time.time()
    
    # 更新时间信息
    timing_info = state.get("timing_info", {})
    timing_info["parallel_processing"] = end_time - start_time
    
    print(f"并行处理完成，耗时: {timing_info['parallel_processing']:.2f} 秒")
    
    return {
        "task_results": results,
        "timing_info": timing_info
    }

async def sequential_processing(state: ProcessingState) -> Dict[str, Any]:
    """串行处理多个任务（用于对比）
    
    WHY - 设计思路:
    1. 需要提供一个基准线来对比并行处理的性能提升
    2. 需要使用相同的任务处理逻辑保证公平比较
    3. 需要记录准确的时间数据
    
    HOW - 实现方式:
    1. 使用for循环顺序执行每个任务
    2. 记录开始和结束时间计算总耗时
    3. 返回相同格式的结果以便于对比
    
    WHAT - 功能作用:
    作为对比基准，串行处理相同的任务集，展示与并行处理的
    性能差异，帮助理解并行优化的效果
    
    Args:
        state: 当前处理状态
        
    Returns:
        Dict[str, Any]: 包含处理结果和性能数据的状态更新
    """
    start_time = time.time()
    tasks = state["tasks"]
    
    print(f"开始串行处理 {len(tasks)} 个任务...")
    
    results = []
    for task in tasks:
        # 一个一个处理任务
        result = await process_task(task)
        results.append(result)
    
    end_time = time.time()
    
    # 更新时间信息
    timing_info = state.get("timing_info", {})
    timing_info["sequential_processing"] = end_time - start_time
    
    print(f"串行处理完成，耗时: {timing_info['sequential_processing']:.2f} 秒")
    
    return {
        "task_results": results,
        "timing_info": timing_info
    }

# =================================================================
# 第3部分: 缓存策略实现
# =================================================================

@lru_cache(maxsize=100)
def cached_expensive_operation(input_data: str) -> str:
    """使用缓存优化昂贵操作
    
    WHY - 设计思路:
    1. 某些操作可能被重复调用但结果相同
    2. 缓存可以避免重复计算，提高响应速度
    3. 需要限制缓存大小防止内存溢出
    
    HOW - 实现方式:
    1. 使用Python内置的lru_cache装饰器
    2. 设置最大缓存项数为100
    3. 模拟耗时操作展示缓存效果
    
    WHAT - 功能作用:
    展示缓存优化技术，对重复输入只计算一次，大幅提高
    处理相同或重复任务的效率
    
    Args:
        input_data: 输入数据字符串
        
    Returns:
        str: 处理结果
    """
    print(f"执行昂贵操作，输入: {input_data}")
    
    # 模拟昂贵的操作
    time.sleep(1.0)  # 模拟耗时操作
    
    # 返回处理结果
    return f"处理结果: {input_data.upper()}"

def cached_node(state: ProcessingState) -> Dict[str, Any]:
    """使用缓存的节点
    
    WHY - 设计思路:
    1. 需要在LangGraph节点中利用缓存机制
    2. 需要处理多个可能重复的任务
    3. 需要记录缓存带来的性能提升
    
    HOW - 实现方式:
    1. 从任务中提取输入数据
    2. 调用使用lru_cache装饰的函数
    3. 记录处理时间并返回结果
    
    WHAT - 功能作用:
    作为LangGraph图的节点，处理多个任务时利用缓存
    机制提高性能，特别是对于重复或相似任务
    
    Args:
        state: 当前处理状态
        
    Returns:
        Dict[str, Any]: 包含处理结果和性能数据的状态更新
    """
    start_time = time.time()
    
    # 从任务中提取输入数据
    results = []
    for task in state["tasks"]:
        input_data = task.get("input", "")
        
        # 使用缓存的操作
        result = cached_expensive_operation(input_data)
        
        results.append({
            "task_id": task.get("id"),
            "input": input_data,
            "result": result
        })
    
    end_time = time.time()
    
    # 更新时间信息
    timing_info = state.get("timing_info", {})
    timing_info["cached_processing"] = end_time - start_time
    
    print(f"缓存处理完成，耗时: {timing_info['cached_processing']:.2f} 秒")
    
    return {
        "task_results": results,
        "timing_info": timing_info
    }

# =================================================================
# 第4部分: 分布式部署实现
# =================================================================

if HAS_RAY:
    # 初始化Ray
    if not ray.is_initialized():
        ray.init()
    
    @ray.remote
    def distributed_task_processor(task):
        """分布式处理单个任务
        
        WHY - 设计思路:
        1. 需要能在分布式环境中执行的任务处理函数
        2. 需要支持Ray的远程执行机制
        3. 任务处理逻辑应与其他方法保持一致
        
        HOW - 实现方式:
        1. 使用ray.remote装饰器标记为可分布式执行
        2. 使用time.sleep模拟处理延迟
        3. 返回与其他方法格式一致的结果
        
        WHAT - 功能作用:
        作为Ray分布式框架的远程任务，可以被分配到
        集群中的不同节点执行，实现真正的并行计算
        
        Args:
            task: 包含任务信息的字典
            
        Returns:
            Dict[str, Any]: 任务处理结果
        """
        print(f"[Ray分布式] 处理任务: {task['name']}")
        time.sleep(task.get("delay", 1.0))  # 模拟处理时间
        
        return {
            "task_id": task.get("id"),
            "task_name": task.get("name"),
            "result": f"[分布式] 完成任务 '{task.get('name')}' 的处理",
            "processed_at": datetime.now().isoformat()
        }
    
    def distributed_processing(state: ProcessingState) -> Dict[str, Any]:
        """使用Ray分布式处理任务
        
        WHY - 设计思路:
        1. 需要利用多核心甚至多机器的计算资源
        2. 需要处理计算密集型任务集合
        3. 需要记录分布式处理的性能数据
        
        HOW - 实现方式:
        1. 使用Ray的远程函数创建分布式任务
        2. 使用ray.get等待所有任务完成
        3. 记录性能数据并返回结果
        
        WHAT - 功能作用:
        作为LangGraph图的节点，利用Ray分布式框架执行
        任务处理，可以跨越多个CPU核心甚至多台机器，
        实现更高级别的并行
        
        Args:
            state: 当前处理状态
            
        Returns:
            Dict[str, Any]: 包含处理结果和性能数据的状态更新
        """
        start_time = time.time()
        tasks = state["tasks"]
        
        print(f"开始Ray分布式处理 {len(tasks)} 个任务...")
        
        # 创建Ray任务
        futures = [distributed_task_processor.remote(task) for task in tasks]
        
        # 等待所有任务完成
        results = ray.get(futures)
        
        end_time = time.time()
        
        # 更新时间信息
        timing_info = state.get("timing_info", {})
        timing_info["distributed_processing"] = end_time - start_time
        
        print(f"分布式处理完成，耗时: {timing_info['distributed_processing']:.2f} 秒")
        
        return {
            "task_results": results,
            "timing_info": timing_info
        }
else:
    # 未安装Ray时的替代函数
    def distributed_processing(state: ProcessingState) -> Dict[str, Any]:
        """未安装Ray时的替代处理函数
        
        WHY - 设计思路:
        1. 需要在没有Ray时提供备选方案
        2. 需要确保代码在各种环境中都能运行
        
        HOW - 实现方式:
        1. 检测Ray是否可用并提供替代实现
        2. 添加明确的警告信息
        
        WHAT - 功能作用:
        在未安装Ray库的环境中提供友好的降级处理，
        确保代码可以在不同环境中运行
        
        Args:
            state: 当前处理状态
            
        Returns:
            Dict[str, Any]: 最小更新的状态
        """
        print("未安装Ray，无法使用分布式处理")
        timing_info = state.get("timing_info", {})
        timing_info["distributed_processing"] = 0
        return state

# =================================================================
# 第5部分: 结果处理与总结
# =================================================================

def summarize_results(state: ProcessingState) -> Dict[str, Any]:
    """汇总处理结果并生成最终回应
    
    WHY - 设计思路:
    1. 需要对不同处理方法的结果进行汇总
    2. 需要生成人类可读的性能对比信息
    3. 需要提供智能化的结果分析
    
    HOW - 实现方式:
    1. 提取所有处理结果和性能数据
    2. 使用LLM生成易于理解的摘要
    3. 返回最终响应
    
    WHAT - 功能作用:
    作为处理流程的最后一步，汇总所有信息，使用LLM分析
    处理结果和性能数据，生成有意义的总结报告
    
    Args:
        state: 当前处理状态
        
    Returns:
        Dict[str, Any]: 包含最终响应的状态更新
    """
    results = state.get("task_results", [])
    timing_info = state.get("timing_info", {})
    
    # 计算总时间
    total_time = sum(timing_info.values())
    
    # 使用LLM生成摘要
    tasks_summary = "\n".join([f"- 任务 {r.get('task_name', 'unknown')}: {r.get('result', '')}" for r in results[:3]])
    if len(results) > 3:
        tasks_summary += f"\n- ... 共 {len(results)} 个任务结果"
    
    prompt = f"""
    根据以下任务处理结果生成简短摘要:
    
    {tasks_summary}
    
    处理性能统计:
    {json.dumps(timing_info, indent=2, ensure_ascii=False)}
    总处理时间: {total_time:.2f} 秒
    """
    
    response = llm.invoke(prompt)
    
    return {
        "final_response": response,
    }

def route_processing_method(state: ProcessingState) -> str:
    """根据任务特性选择处理方法
    
    WHY - 设计思路:
    1. 不同任务集合适合不同的处理方法
    2. 需要智能选择最佳处理策略
    3. 需要考虑环境限制(如是否有Ray)
    
    HOW - 实现方式:
    1. 分析任务数量和类型
    2. 根据任务特性和环境选择合适的处理方法
    3. 返回对应的路由决策
    
    WHAT - 功能作用:
    作为LangGraph的智能路由节点，分析任务特性自动
    选择最合适的处理方法，优化整体性能
    
    Args:
        state: 当前处理状态
        
    Returns:
        str: 路由决策，指示下一步使用哪种处理方法
    """
    tasks = state.get("tasks", [])
    
    # 根据任务特性选择最佳处理方法
    if not tasks:
        return "no_tasks"
    
    # 如果任务数量大，使用分布式处理
    if len(tasks) >= 5 and HAS_RAY:
        return "distributed"
    # 如果任务数量中等，使用并行处理
    elif len(tasks) >= 3:
        return "parallel"
    # 如果任务是同质的，使用缓存处理
    elif all(task.get("type") == "cacheable" for task in tasks):
        return "cached"
    # 否则使用串行处理
    else:
        return "sequential"

# =================================================================
# 第6部分: 图结构构建
# =================================================================

def build_processing_graph():
    """构建处理图
    
    WHY - 设计思路:
    1. 需要将不同处理节点组织为一个完整的工作流
    2. 需要实现智能路由选择最佳处理方法
    3. 需要支持不同环境配置(有无Ray)
    
    HOW - 实现方式:
    1. 创建基于ProcessingState的StateGraph
    2. 添加各种处理节点和汇总节点
    3. 添加条件边实现智能路由
    4. 设置结果处理流程
    
    WHAT - 功能作用:
    构建一个完整的LangGraph处理图，集成并行、串行、
    缓存和分布式处理节点，通过智能路由选择最佳处理方式
    
    Returns:
        StateGraph: 编译后的处理图
    """
    # 创建图
    workflow = StateGraph(ProcessingState)
    
    # 添加节点
    workflow.add_node("parallel_processing", parallel_processing)
    workflow.add_node("sequential_processing", sequential_processing)
    workflow.add_node("cached_processing", cached_node)
    workflow.add_node("summarize", summarize_results)
    
    if HAS_RAY:
        workflow.add_node("distributed_processing", distributed_processing)
    
    # 添加路由逻辑
    workflow.add_conditional_edges(
        "root",  # 起始节点
        route_processing_method,
        {
            "parallel": "parallel_processing",
            "sequential": "sequential_processing",
            "cached": "cached_processing",
            "distributed": "distributed_processing" if HAS_RAY else "parallel_processing",
            "no_tasks": "summarize"
        }
    )
    
    # 添加结果处理流程
    workflow.add_edge("parallel_processing", "summarize")
    workflow.add_edge("sequential_processing", "summarize")
    workflow.add_edge("cached_processing", "summarize")
    
    if HAS_RAY:
        workflow.add_edge("distributed_processing", "summarize")
    
    # 添加终止边
    workflow.add_edge("summarize", END)
    
    # 编译图
    return workflow.compile(checkpointer=memory_checkpoint.MemoryCheckpointer())

# =================================================================
# 第7部分: 演示与比较
# =================================================================

async def run_performance_comparison():
    """运行性能比较演示
    
    WHY - 设计思路:
    1. 需要演示和比较不同处理方法的性能
    2. 需要使用相同的测试数据确保公平比较
    3. 需要清晰展示每种方法的优缺点
    
    HOW - 实现方式:
    1. 构建处理图和测试任务集
    2. 强制使用不同处理方法进行测试
    3. 展示和比较处理结果和性能数据
    
    WHAT - 功能作用:
    提供一个完整的演示，直观展示并行、串行、缓存和
    分布式处理的性能差异，帮助理解各种优化方法的适用场景
    
    Returns:
        None
    """
    # 构建图
    graph = build_processing_graph()
    
    # 创建测试任务
    tasks = [
        {"id": 1, "name": "数据分析任务", "type": "heavy", "input": "data_analysis"},
        {"id": 2, "name": "图像处理", "type": "heavy", "input": "image_processing"},
        {"id": 3, "name": "文本摘要", "type": "medium", "input": "text_summarization"},
        {"id": 4, "name": "情感分析", "type": "light", "input": "sentiment_analysis"},
        {"id": 5, "name": "翻译处理", "type": "medium", "input": "translation"},
    ]
    
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content="请并行处理多个任务并比较性能")],
        "tasks": tasks,
        "task_results": [],
        "final_response": None,
        "timing_info": {}
    }
    
    print("\n===================== 异步并行处理 =====================")
    # 强制使用并行处理
    parallel_tasks = tasks.copy()
    result = await graph.ainvoke(
        {**initial_state, "tasks": parallel_tasks},
        {"configurable": {"thread_id": "parallel_demo"}}
    )
    print(f"\n并行处理结果摘要:\n{result.get('final_response', '')}")
    
    print("\n===================== 串行处理 =====================")
    # 强制使用串行处理
    sequential_tasks = [{"id": i, "name": f"串行任务-{i}", "type": "light"} for i in range(1, 6)]
    result = await graph.ainvoke(
        {**initial_state, "tasks": sequential_tasks},
        {"configurable": {"thread_id": "sequential_demo"}}
    )
    print(f"\n串行处理结果摘要:\n{result.get('final_response', '')}")
    
    print("\n===================== 缓存处理 =====================")
    # 使用可缓存的相同任务
    cacheable_tasks = [
        {"id": i, "name": f"缓存任务", "type": "cacheable", "input": "cache_test"} 
        for i in range(1, 6)
    ]
    result = await graph.ainvoke(
        {**initial_state, "tasks": cacheable_tasks},
        {"configurable": {"thread_id": "cache_demo"}}
    )
    print(f"\n缓存处理结果摘要:\n{result.get('final_response', '')}")
    
    if HAS_RAY:
        print("\n===================== 分布式处理 =====================")
        # 使用分布式处理
        distributed_tasks = [
            {"id": i, "name": f"分布式任务-{i}", "type": "heavy", "delay": 1.5} 
            for i in range(1, 8)
        ]
        result = await graph.ainvoke(
            {**initial_state, "tasks": distributed_tasks},
            {"configurable": {"thread_id": "distributed_demo"}}
        )
        print(f"\n分布式处理结果摘要:\n{result.get('final_response', '')}")

# =================================================================
# 第8部分: 主程序入口
# =================================================================

if __name__ == "__main__":
    """主程序入口
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行演示
    2. 需要显示环境信息和运行状态
    
    HOW - 实现方式:
    1. 检测环境（Ray是否可用）
    2. 运行异步演示函数
    
    WHAT - 功能作用:
    作为程序入口点，显示环境信息并启动性能比较演示
    """
    print("启动LangGraph性能优化示例...")
    print(f"Ray分布式支持: {'已启用' if HAS_RAY else '未启用'}")
    
    # 运行异步主函数
    asyncio.run(run_performance_comparison()) 