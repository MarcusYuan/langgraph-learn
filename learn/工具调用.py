#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LangGraph 工具调用示例

WHY - 设计思路:
1. LLM应用通常需要访问外部工具来扩展能力(如搜索、API调用、文件操作等)
2. LangGraph框架提供了构建复杂工具调用流程的能力，但需要示例展示最佳实践
3. 工具调用过程需要清晰的状态管理、工具选择逻辑和错误处理机制

HOW - 实现方式:
1. 使用TypedDict定义清晰的工具状态结构(ToolState)
2. 实现多种工具函数，覆盖搜索、天气、文件操作、计算等常见需求
3. 构建节点函数链，包括查询分析、工具选择、工具执行和回复生成
4. 使用条件边和路由函数实现动态工具选择和执行
5. 构建完整的工作流图，将各个节点和边连接为一个完整系统

WHAT - 功能作用:
本文件提供了一个完整的LangGraph工具调用示例，展示如何:
1. 设计工具状态和工具接口
2. 构建工具调用工作流图
3. 实现各类工具函数和处理节点
4. 处理用户查询并执行适当的工具
5. 生成整合了工具结果的回复

使用方法:
直接运行此文件，将展示多个预设示例，并提供交互式界面测试自定义查询
"""

from typing import TypedDict, List, Dict, Any, Optional, Union, Callable, Tuple
import os
import json
import time
import datetime
import requests
from pathlib import Path
import inspect
from dateutil import parser

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, Graph
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables import RunnablePassthrough

# ===========================================================
# 第1部分: 状态定义
# ===========================================================

class ToolState(TypedDict):
    """工具调用状态定义
    
    WHY - 设计思路:
    1. 工具调用需要一个结构化的状态容器，用于在节点间传递数据
    2. 需要跟踪用户查询、选择的工具、工具执行结果等信息
    3. 错误处理需要专门的状态字段，以便在出错时能够优雅地处理
    4. 不同类型的工具结果(搜索、API、文件)需要不同的存储结构
    
    HOW - 实现方式:
    1. 使用TypedDict提供类型安全和代码提示
    2. 定义消息历史字段存储对话上下文
    3. 设计专用字段存储不同类型的工具结果
    4. 使用Optional标记可能为空的字段
    
    WHAT - 功能作用:
    提供一个类型安全的状态结构，包含工具调用全流程所需的所有数据字段，
    使得工具调用的状态传递清晰可追踪，便于调试和维护
    """
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # 消息历史
    current_query: Optional[str]  # 当前查询
    tools_to_use: Optional[List[str]]  # 需要使用的工具列表
    search_results: Optional[List[Dict[str, Any]]]  # 搜索结果
    api_results: Optional[Dict[str, Any]]  # API调用结果
    file_content: Optional[str]  # 文件内容
    error: Optional[str]  # 错误信息

def initialize_state(messages: List[Dict] = None) -> ToolState:
    """初始化工具调用状态
    
    WHY - 设计思路:
    1. 需要为每次对话设置一个干净的初始状态
    2. 状态需要包含系统指令和消息历史，建立基础上下文
    3. 状态结构应清晰，便于后续在节点函数间传递和更新
    
    HOW - 实现方式:
    1. 创建符合ToolState类型的字典结构
    2. 设置默认的系统指令，定义AI助手的角色和行为
    3. 允许通过参数传入自定义的消息历史，便于测试和演示
    4. 确保所有必要的状态字段都被初始化
    
    WHAT - 功能作用:
    创建一个干净、一致的初始状态，作为工具调用工作流的起点，
    保证每次对话都从一个可预测的状态开始，避免状态污染
    
    Args:
        messages: 可选的初始消息列表，默认为None
        
    Returns:
        初始化的ToolState字典
    """
    # 设置默认系统消息
    sys_message = {
        "role": "system",
        "content": """你是一个有帮助的AI助手，可以使用多种工具来回答用户的问题。
可用的工具包括:
1. search_web: 用于在网络上搜索信息
2. get_weather: 用于获取天气信息
3. read_file: 用于读取文件内容
4. write_file: 用于将内容写入文件
5. get_latest_news: 用于获取最新新闻
6. calculate: 用于执行数学计算
7. convert_date: 用于日期格式转换

根据用户的查询，选择最合适的工具来提供帮助。"""
    }
    
    # 初始化消息历史
    if messages is None:
        messages = [sys_message]
    else:
        # 确保系统消息在最前面
        if not any(msg.get("role") == "system" for msg in messages):
            messages = [sys_message] + messages
    
    # 返回初始状态
    return {
        "messages": messages,
        "selected_tools": [],
        "results": {},
        "error": None
    }

# ===========================================================
# 第2部分: LLM配置
# ===========================================================

# 尝试使用Ollama本地模型，如果不可用，使用模拟LLM
try:
    llm = Ollama(model="llama3", temperature=0.7)
    print("成功连接到Ollama模型")
except:
    print("警告: 无法连接到Ollama模型，使用模拟LLM")
    
    # 创建一个模拟LLM用于演示
    class MockLLM:
        def invoke(self, prompt, **kwargs):
            print(f"模拟LLM接收到提示: {prompt[:50]}...")
            tools_mentioned = "搜索" in prompt or "查询" in prompt or "查找" in prompt
            weather_mentioned = "天气" in prompt
            news_mentioned = "新闻" in prompt
            file_mentioned = "文件" in prompt or "写入" in prompt or "读取" in prompt
            
            if tools_mentioned and weather_mentioned:
                return "我需要使用天气API工具来回答这个问题。应该查询当前的天气情况。"
            elif tools_mentioned and news_mentioned:
                return "我需要使用搜索工具来查找最新的新闻信息。"
            elif tools_mentioned and file_mentioned:
                return "我需要使用文件操作工具来处理用户的请求。"
            else:
                return "我可以直接回答这个问题，不需要使用特定工具。"
    
    llm = MockLLM()

# ===========================================================
# 第3部分: 工具定义
# ===========================================================

# 1. 搜索工具 - 模拟网络搜索
@tool
def search_web(query: str) -> str:
    """搜索互联网以获取信息
    
    WHY - 设计思路:
    1. 用户经常需要获取最新或专业信息，而LLM的知识可能有限或过时
    2. 搜索功能允许AI访问互联网获取最新信息，极大扩展了回答能力
    3. 需要有标准化的接口来处理搜索查询和返回结果
    
    HOW - 实现方式:
    1. 使用@tool装饰器定义标准化工具接口
    2. 接收用户查询文本作为输入参数
    3. 模拟搜索延迟，增加真实感
    4. 根据关键词匹配返回相应的模拟搜索结果
    
    WHAT - 功能作用:
    提供一个模拟的网络搜索功能，根据用户查询返回相关信息，
    在实际应用中可替换为真实的搜索API(如Google、Bing或专有搜索服务)
    
    Args:
        query: 搜索查询
        
    Returns:
        搜索结果的摘要
    """
    print(f"执行网络搜索: {query}")
    # 模拟搜索延迟
    time.sleep(1)
    
    # 根据不同查询返回模拟结果
    if "天气" in query:
        return "搜索结果: 今天大部分地区晴朗，温度在20-25°C之间，有微风。"
    elif "新闻" in query:
        return "搜索结果: 1.全球科技大会即将召开 2.新能源汽车销量创新高 3.人工智能技术取得新突破"
    elif "langgraph" in query.lower():
        return "搜索结果: LangGraph是一个用于构建和运行LLM驱动的多代理工作流的框架，它使用有向图结构来控制流程。"
    else:
        return f"搜索结果: 关于'{query}'的一些相关信息。这是模拟的搜索结果，仅用于演示。"

# 2. 天气API工具
@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息
    
    WHY - 设计思路:
    1. 天气查询是用户的高频需求，需要实时准确的数据
    2. LLM本身不具备获取实时天气的能力，需要外部API支持
    3. 天气数据需要根据地点参数化，以支持不同地区的查询
    
    HOW - 实现方式:
    1. 使用@tool装饰器创建标准化工具接口
    2. 接收位置名称作为输入参数
    3. 模拟API调用延迟，增强真实感
    4. 使用预定义的天气数据字典模拟不同城市的天气情况
    5. 格式化输出包含时间、温度、天气状况和湿度
    
    WHAT - 功能作用:
    提供一个天气查询功能，根据用户指定的位置返回相应的天气信息，
    在实际应用中可替换为真实的天气API服务(如OpenWeatherMap、天气网API等)
    
    Args:
        location: 地点名称，如'北京'、'上海'等
        
    Returns:
        天气信息
    """
    print(f"查询天气API: {location}")
    # 模拟API调用延迟
    time.sleep(1.5)
    
    # 返回模拟天气数据
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 为不同城市返回不同天气
    weather_data = {
        "北京": {"temperature": "23°C", "condition": "晴朗", "humidity": "45%"},
        "上海": {"temperature": "26°C", "condition": "多云", "humidity": "60%"},
        "广州": {"temperature": "30°C", "condition": "局部阵雨", "humidity": "75%"},
        "深圳": {"temperature": "29°C", "condition": "晴间多云", "humidity": "65%"}
    }
    
    if location in weather_data:
        data = weather_data[location]
        return f"{current_time} {location}天气: 温度{data['temperature']}, {data['condition']}, 湿度{data['humidity']}"
    else:
        return f"{current_time} {location}天气: 温度25°C, 晴朗, 湿度50% (模拟数据)"

# 3. 文件读取工具
@tool
def read_file(file_path: str) -> str:
    """读取文件内容
    
    WHY - 设计思路:
    1. 用户经常需要获取文件内容进行处理或分析
    2. AI助手需要访问本地文件系统，以便引用或基于文件内容回答问题
    3. 需要读取不同类型的文件(文本、代码、日志等)，并处理可能的读取错误
    
    HOW - 实现方式:
    1. 使用@tool装饰器定义标准化文件读取接口
    2. 接收文件路径作为参数
    3. 使用Python的文件操作函数打开并读取文件
    4. 采用try-except处理可能的文件不存在或权限错误
    5. 检测文件大小，对过大的文件进行截断处理
    
    WHAT - 功能作用:
    提供一个安全的文件读取功能，使AI助手能够访问本地文件内容，
    支持回答基于文件的问题，比如代码审查、文档分析等任务
    
    Args:
        file_path: 要读取的文件路径
        
    Returns:
        文件内容或错误信息
    """
    print(f"读取文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 如果文件太大，只返回前1000个字符
            if len(content) > 1000:
                content = content[:1000] + "\n...(文件过长，已截断)..."
            
            return f"文件内容:\n{content}"
    except FileNotFoundError:
        return f"错误: 找不到文件 '{file_path}'"
    except PermissionError:
        return f"错误: 没有权限读取文件 '{file_path}'"
    except Exception as e:
        return f"读取文件时出错: {str(e)}"

# 4. 文件写入工具
@tool
def write_file(file_path: str, content: str) -> str:
    """写入内容到文件
    
    WHY - 设计思路:
    1. 用户可能需要AI助手帮助创建或修改文件
    2. 文件写入操作需要安全控制，避免覆盖重要文件
    3. 需要处理可能的权限问题和路径错误
    
    HOW - 实现方式:
    1. 使用@tool装饰器定义标准化文件写入接口
    2. 接收文件路径和要写入的内容作为参数
    3. 使用安全检查确保不会写入到系统关键目录
    4. 使用Python的文件操作函数创建或覆盖文件
    5. 采用try-except处理可能的权限错误或路径问题
    
    WHAT - 功能作用:
    提供一个受控的文件写入功能，使AI助手能够创建或修改本地文件，
    支持代码生成、内容编辑、配置文件创建等任务
    
    Args:
        file_path: 要写入的文件路径
        content: 要写入的内容
        
    Returns:
        成功或错误信息
    """
    print(f"写入文件: {file_path}")
    
    # 安全检查，避免写入敏感目录
    sensitive_paths = ['/etc/', '/bin/', '/sbin/', '/usr/', '/boot/', '/root/']
    if any(file_path.startswith(path) for path in sensitive_paths):
        return f"错误: 安全限制，不允许写入系统目录 '{file_path}'"
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"成功: 内容已写入到文件 '{file_path}'"
    except PermissionError:
        return f"错误: 没有权限写入文件 '{file_path}'"
    except Exception as e:
        return f"写入文件时出错: {str(e)}"

# 5. 新闻API工具
@tool
def get_latest_news(category: str = "general") -> str:
    """获取最新新闻
    
    WHY - 设计思路:
    1. 用户经常需要了解最新事件和新闻动态
    2. LLM的训练数据存在时效性限制，需要外部API获取实时新闻
    3. 新闻需要按类别组织，满足用户对特定领域新闻的需求
    
    HOW - 实现方式:
    1. 使用@tool装饰器定义标准化新闻获取接口
    2. 接收新闻类别作为可选参数，默认为"general"
    3. 模拟API调用延迟以增强真实感
    4. 按类别提供预定义的模拟新闻数据
    5. 处理无效类别输入，提供可用类别信息
    
    WHAT - 功能作用:
    提供一个新闻获取功能，根据用户指定的类别返回相关的最新新闻，
    在实际应用中可替换为真实的新闻API服务(如NewsAPI、Google News API等)
    
    Args:
        category: 新闻类别，可选值包括"general"、"technology"、"sports"等
        
    Returns:
        特定类别的最新新闻列表或错误信息
    """
    print(f"获取{category}类别的最新新闻")
    # 模拟API调用延迟
    time.sleep(1.2)
    
    # 返回模拟新闻数据
    news_data = {
        "general": [
            "联合国召开气候变化紧急会议，多国承诺减排",
            "全球经济复苏迹象显现，多国央行维持利率不变",
            "新冠疫苗覆盖率突破70%，全球健康指标改善"
        ],
        "technology": [
            "最新AI模型在多项基准测试中打破记录",
            "科技巨头发布新一代量子计算机，性能提升100倍",
            "可降解电子元件问世，有望解决电子垃圾问题"
        ],
        "sports": [
            "夏季奥运会闭幕，美国位居奖牌榜首位",
            "国际足联宣布世界杯扩军计划，参赛队伍将增至48支",
            "网球名将宣布退役，职业生涯获20个大满贯冠军"
        ]
    }
    
    # 返回对应类别的新闻
    if category.lower() in news_data:
        news_list = news_data[category.lower()]
        return f"最新{category}新闻:\n" + "\n".join([f"- {item}" for item in news_list])
    else:
        return f"没有找到{category}类别的新闻，可用类别: general, technology, sports"

# 6. 计算器工具
@tool
def calculate(expression: str) -> str:
    """执行数学计算
    
    WHY - 设计思路:
    1. 用户经常需要进行数学计算，这是AI助手的基础功能之一
    2. 计算过程需要准确、安全，并能处理各类数学表达式
    3. 需要提供清晰的错误处理机制，应对错误或不安全的输入
    
    HOW - 实现方式:
    1. 使用@tool装饰器定义标准工具接口
    2. 接收数学表达式字符串作为输入
    3. 使用eval()函数执行计算，但需进行安全检查
    4. 使用try-except捕获可能的计算错误，提供友好的错误信息
    5. 格式化返回结果，确保可读性
    
    WHAT - 功能作用:
    提供一个安全的计算器功能，能够处理基础数学运算，
    返回计算结果或错误信息，使AI助手能够准确回答计算类问题
    
    Args:
        expression: 数学表达式字符串，如"2+3*4"
        
    Returns:
        计算结果或错误信息
    """
    print(f"执行计算: {expression}")
    try:
        # 简单安全检查
        if any(keyword in expression for keyword in ['import', 'eval', 'exec', 'open', '__']):
            return "计算错误: 包含不安全的表达式"
        
        # 使用更安全的ast.literal_eval替代eval也是一个选择
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 7. 日期转换工具
@tool
def convert_date(date_str: str, format_str: str = "YYYY-MM-DD") -> str:
    """日期格式转换
    
    WHY - 设计思路:
    1. 日期处理是常见的数据转换需求，不同场景需要不同的日期格式
    2. 用户经常需要将日期从一种格式转换为另一种格式
    3. 需要支持多种输入格式和输出格式，增强通用性
    
    HOW - 实现方式:
    1. 使用@tool装饰器定义标准化工具接口
    2. 接收日期字符串和目标格式参数
    3. 使用dateutil库解析各种格式的日期输入
    4. 根据目标格式字符串确定输出格式
    5. 使用异常处理确保即使输入无效也能得到友好的反馈
    
    WHAT - 功能作用:
    提供一个灵活的日期转换工具，能识别多种日期格式并按照指定格式输出，
    帮助用户处理日期相关的查询和任务
    
    Args:
        date_str: 输入的日期字符串
        format_str: 目标日期格式，如"YYYY-MM-DD"
        
    Returns:
        格式化后的日期字符串或错误信息
    """
    print(f"转换日期: {date_str} 到格式 {format_str}")
    try:
        # 使用dateutil解析各种格式的日期
        parsed_date = parser.parse(date_str)
        
        # 将format_str转换为datetime.strftime兼容的格式
        format_map = {
            "YYYY": "%Y",
            "YY": "%y",
            "MM": "%m",
            "DD": "%d",
            "HH": "%H",
            "mm": "%M",
            "ss": "%S"
        }
        
        for k, v in format_map.items():
            format_str = format_str.replace(k, v)
        
        return f"日期转换结果: {parsed_date.strftime(format_str)}"
    except Exception as e:
        return f"日期转换错误: {str(e)}"

# 收集所有工具
available_tools = [
    search_web,
    get_weather,
    read_file,
    write_file,
    get_latest_news,
    calculate,
    convert_date
]

# ===========================================================
# 第4部分: 节点函数定义
# ===========================================================

def analyze_user_query(state: ToolState) -> ToolState:
    """分析用户查询并识别意图
    
    WHY - 设计思路:
    1. 需要理解用户的实际需求和查询意图，以选择合适的工具进行响应
    2. 原始用户查询可能含糊不清，需要提取关键信息并明确查询目标
    3. 查询分析是工具调用流程的第一步，为后续工具选择提供依据
    
    HOW - 实现方式:
    1. 从state中获取最近的用户消息作为查询内容
    2. 使用LLM生成分析结果，提取用户查询的关键信息和意图
    3. 使用结构化提示词引导LLM分析查询类型、需求细节和可能的工具需求
    4. 将分析结果更新到state字典中，作为下一步决策的依据
    
    WHAT - 功能作用:
    理解并明确用户的查询意图，为后续的工具选择和执行提供清晰的指导，
    提高工具调用的精确性和相关性
    
    Args:
        state: 当前工具状态，包含消息历史等信息
        
    Returns:
        更新后的工具状态，包含查询分析结果
    """
    print("分析用户查询...")
    
    # 获取最近的用户消息
    last_message = state["messages"][-1]["content"]
    
    # 使用LLM分析用户查询
    prompt = f"""
    分析以下用户查询，识别其意图和可能需要的工具：
    
    用户查询: {last_message}
    
    请提供:
    1. 查询类型(搜索、获取信息、计算、文件操作等)
    2. 查询中的关键信息(如搜索词、地点、文件名等)
    3. 可能需要的工具
    
    以JSON格式返回结果。
    """
    
    try:
        analysis_result = lm_model(prompt)
        # 更新state，添加查询分析结果
        new_state = state.copy()
        new_state["query_analysis"] = analysis_result
        return new_state
    except Exception as e:
        # 如有错误，保持原状态并添加错误信息
        new_state = state.copy()
        new_state["error"] = f"查询分析错误: {str(e)}"
        return new_state

def select_tools(state: ToolState) -> ToolState:
    """根据用户查询选择合适的工具
    
    WHY - 设计思路:
    1. 需要根据用户查询类型智能选择最合适的工具或工具组合
    2. 不同查询需要不同的工具组合，有时需要多个工具协同工作
    3. 工具选择需要基于查询分析结果，避免不必要的工具调用
    
    HOW - 实现方式:
    1. 从state中获取查询分析结果和可用工具列表
    2. 使用LLM基于查询分析、可用工具和工具功能描述做出决策
    3. 使用结构化提示词指导LLM返回工具名称列表和调用顺序
    4. 解析LLM的返回结果，提取所需的工具名称
    5. 将选定的工具信息更新到state字典
    
    WHAT - 功能作用:
    智能选择处理用户查询所需的工具，确保工具选择的精确性和相关性，
    为后续的工具执行环节提供明确指导
    
    Args:
        state: 当前工具状态，包含查询分析结果
        
    Returns:
        更新后的工具状态，包含选定的工具信息
    """
    print("选择合适的工具...")
    
    # 如果前一步出现错误，跳过工具选择
    if "error" in state and state["error"]:
        return state
    
    # 获取查询分析和最近的用户消息
    query_analysis = state.get("query_analysis", "")
    last_message = state["messages"][-1]["content"]
    
    # 生成工具描述列表
    tool_descriptions = []
    for tool in available_tools:
        doc = tool.__doc__ or ""
        signature = str(inspect.signature(tool))
        tool_descriptions.append(f"工具名: {tool.__name__}{signature}\n描述: {doc.strip()}")
    
    # 使用LLM选择工具
    prompt = f"""
    基于用户查询和查询分析，选择需要调用的工具。
    
    用户查询: {last_message}
    
    查询分析: {query_analysis}
    
    可用工具:
    {'\n\n'.join(tool_descriptions)}
    
    请选择最合适的工具来回答用户查询。如果不需要工具，请返回空列表。
    以JSON格式返回工具名称列表，格式为: ["工具名1", "工具名2", ...]
    """
    
    try:
        tools_decision = lm_model(prompt)
        
        # 尝试解析JSON结果
        try:
            tools_to_use = json.loads(tools_decision)
            # 确保结果是列表类型
            if not isinstance(tools_to_use, list):
                tools_to_use = []
        except json.JSONDecodeError:
            # 如果不是有效的JSON，尝试直接从文本中提取工具名
            tools_to_use = []
            # 检查每个工具名是否出现在文本中
            for tool in available_tools:
                if tool.__name__ in tools_decision:
                    tools_to_use.append(tool.__name__)
                    
            # 如果仍然失败，手动解析
            if not tools_to_use:
                for tool_name in ["search_web", "get_weather", "read_file", "write_file", "get_latest_news", "calculate", "convert_date"]:
                    if tool_name in tools_decision:
                        tools_to_use.append(tool_name)
        
        # 更新state
        new_state = state.copy()
        new_state["selected_tools"] = tools_to_use
        return new_state
    except Exception as e:
        # 如有错误，保持原状态并添加错误信息
        new_state = state.copy()
        new_state["error"] = f"工具选择错误: {str(e)}"
        return new_state

def execute_search(state: ToolState) -> ToolState:
    """执行搜索操作
    
    WHY - 设计思路:
    1. 搜索是常见的信息获取需求，需要专门的节点函数处理
    2. 需要从用户查询中提取精确的搜索关键词，确保搜索准确性
    3. 搜索结果需要结构化存储，方便后续回答生成
    
    HOW - 实现方式:
    1. 检查state中是否选择了search_web工具
    2. 从用户最近消息中提取搜索关键词
    3. 使用LLM生成精确的搜索查询，避免冗余词汇
    4. 调用search_web工具执行实际搜索操作
    5. 将搜索结果添加到state的results字典中
    
    WHAT - 功能作用:
    执行网络搜索操作，获取用户所需的外部信息，
    扩展AI助手的知识范围，提供最新或专业的信息
    
    Args:
        state: 当前工具状态，包含工具选择信息
        
    Returns:
        更新后的工具状态，包含搜索结果
    """
    print("执行搜索...")
    
    # 检查是否需要搜索工具
    selected_tools = state.get("selected_tools", [])
    if "search_web" not in selected_tools:
        return state  # 如果不需要搜索，直接返回原状态
    
    # 获取最近的用户消息
    last_message = state["messages"][-1]["content"]
    
    # 使用LLM提取搜索关键词
    prompt = f"""
    从以下用户查询中提取搜索关键词，生成一个简洁的搜索查询:
    
    用户查询: {last_message}
    
    只返回搜索查询，不要包含其他文字。
    """
    
    try:
        search_query = lm_model(prompt).strip()
        
        # 执行搜索
        search_result = search_web(search_query)
        
        # 更新state
        new_state = state.copy()
        if "results" not in new_state:
            new_state["results"] = {}
        new_state["results"]["search"] = search_result
        return new_state
    except Exception as e:
        # 如有错误，保持原状态并添加错误信息
        new_state = state.copy()
        new_state["error"] = f"搜索执行错误: {str(e)}"
        return new_state

def execute_weather_api(state: ToolState) -> ToolState:
    """执行天气查询操作
    
    WHY - 设计思路:
    1. 天气查询是常见的用户需求，需要专门的节点函数处理
    2. 需要从用户查询中提取准确的地点信息，确保查询精确性
    3. 天气信息需要结构化存储，方便后续回答生成
    
    HOW - 实现方式:
    1. 检查state中是否选择了get_weather工具
    2. 从用户最近消息中提取地点信息
    3. 使用LLM从复杂查询中提取准确的地点名称
    4. 调用get_weather工具执行实际天气查询
    5. 将天气结果添加到state的results字典中
    
    WHAT - 功能作用:
    执行天气查询操作，获取用户指定地点的最新天气信息，
    为用户提供实时、准确的天气数据
    
    Args:
        state: 当前工具状态，包含工具选择信息
        
    Returns:
        更新后的工具状态，包含天气查询结果
    """
    print("执行天气查询...")
    
    # 检查是否需要天气工具
    selected_tools = state.get("selected_tools", [])
    if "get_weather" not in selected_tools:
        return state  # 如果不需要天气查询，直接返回原状态
    
    # 获取最近的用户消息
    last_message = state["messages"][-1]["content"]
    
    # 使用LLM提取地点信息
    prompt = f"""
    从以下用户查询中提取需要查询天气的地点名称:
    
    用户查询: {last_message}
    
    只返回地点名称，不要包含其他文字。如果查询中没有明确的地点，返回"北京"作为默认值。
    """
    
    try:
        location = lm_model(prompt).strip()
        
        # 执行天气查询
        weather_result = get_weather(location)
        
        # 更新state
        new_state = state.copy()
        if "results" not in new_state:
            new_state["results"] = {}
        new_state["results"]["weather"] = weather_result
        return new_state
    except Exception as e:
        # 如有错误，保持原状态并添加错误信息
        new_state = state.copy()
        new_state["error"] = f"天气查询错误: {str(e)}"
        return new_state

def execute_file_operation(state: ToolState) -> ToolState:
    """执行文件操作
    
    WHY - 设计思路:
    1. 文件读写是常见的本地操作需求，需要专门的节点函数处理
    2. 需要从用户查询中提取文件路径和操作类型（读/写）
    3. 文件操作涉及安全风险，需要严格的参数检查和错误处理
    
    HOW - 实现方式:
    1. 检查state中是否选择了read_file或write_file工具
    2. 使用LLM解析用户消息，提取文件路径和操作类型
    3. 对于写操作，还需要提取要写入的内容
    4. 根据操作类型调用相应的工具函数
    5. 将操作结果添加到state的results字典中
    
    WHAT - 功能作用:
    执行文件读写操作，使AI助手能够访问和修改本地文件，
    支持文档处理、代码生成、配置文件管理等任务
    
    Args:
        state: 当前工具状态，包含工具选择信息
        
    Returns:
        更新后的工具状态，包含文件操作结果
    """
    print("执行文件操作...")
    
    # 检查是否需要文件操作工具
    selected_tools = state.get("selected_tools", [])
    file_op_needed = "read_file" in selected_tools or "write_file" in selected_tools
    
    if not file_op_needed:
        return state  # 如果不需要文件操作，直接返回原状态
    
    # 获取最近的用户消息
    last_message = state["messages"][-1]["content"]
    
    # 提取文件路径和操作类型
    if "read_file" in selected_tools:
        # 提取读取的文件路径
        prompt = f"""
        从以下用户查询中提取需要读取的文件路径:
        
        用户查询: {last_message}
        
        只返回文件路径，不要包含其他文字。
        """
        
        try:
            file_path = lm_model(prompt).strip()
            
            # 执行文件读取
            read_result = read_file(file_path)
            
            # 更新state
            new_state = state.copy()
            if "results" not in new_state:
                new_state["results"] = {}
            new_state["results"]["file_read"] = read_result
            return new_state
        except Exception as e:
            # 如有错误，保持原状态并添加错误信息
            new_state = state.copy()
            new_state["error"] = f"文件读取错误: {str(e)}"
            return new_state
            
    elif "write_file" in selected_tools:
        # 提取写入的文件路径和内容
        prompt = f"""
        从以下用户查询中提取需要写入的文件路径和内容:
        
        用户查询: {last_message}
        
        以JSON格式返回，包含两个字段:
        1. file_path: 文件路径
        2. content: 要写入的内容
        """
        
        try:
            write_info = lm_model(prompt)
            
            # 解析返回的JSON
            try:
                write_data = json.loads(write_info)
                file_path = write_data.get("file_path", "")
                content = write_data.get("content", "")
            except:
                # 如果不是有效的JSON，尝试手动解析
                import re
                file_path_match = re.search(r'file_path["\s:]+([^,"]+)', write_info)
                content_match = re.search(r'content["\s:]+(.+?)(?=\n\}|\}|$)', write_info, re.DOTALL)
                
                file_path = file_path_match.group(1).strip() if file_path_match else ""
                content = content_match.group(1).strip() if content_match else ""
            
                # 去除可能的引号
                if file_path.startswith('"') and file_path.endswith('"'):
                    file_path = file_path[1:-1]
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
            
            if not file_path:
                return {**state, "error": "无法提取文件路径"}
                
            # 执行文件写入
            write_result = write_file(file_path, content)
            
            # 更新state
            new_state = state.copy()
            if "results" not in new_state:
                new_state["results"] = {}
            new_state["results"]["file_write"] = write_result
            return new_state
        except Exception as e:
            # 如有错误，保持原状态并添加错误信息
            new_state = state.copy()
            new_state["error"] = f"文件写入错误: {str(e)}"
            return new_state
    
    return state  # 默认返回原状态

def execute_news_api(state: ToolState) -> ToolState:
    """获取最新新闻
    
    WHY - 设计思路:
    1. 新闻查询是常见的信息获取需求，需要专门的节点函数处理
    2. 需要从用户查询中提取新闻类别，确保获取相关新闻
    3. 新闻信息需要结构化存储，方便后续回答生成
    
    HOW - 实现方式:
    1. 检查state中是否选择了get_latest_news工具
    2. 从用户消息中提取感兴趣的新闻类别
    3. 使用LLM识别用户可能感兴趣的新闻类别
    4. 调用get_latest_news工具获取特定类别的新闻
    5. 将新闻结果添加到state的results字典中
    
    WHAT - 功能作用:
    执行新闻获取操作，获取最新的新闻信息，使AI助手能够提供时事更新，
    满足用户了解最新资讯的需求
    
    Args:
        state: 当前工具状态，包含工具选择信息
        
    Returns:
        更新后的工具状态，包含新闻查询结果
    """
    print("获取最新新闻...")
    
    # 检查是否需要新闻工具
    selected_tools = state.get("selected_tools", [])
    if "get_latest_news" not in selected_tools:
        return state  # 如果不需要新闻，直接返回原状态
    
    # 获取最近的用户消息
    last_message = state["messages"][-1]["content"]
    
    # 使用LLM提取新闻类别
    prompt = f"""
    从以下用户查询中提取感兴趣的新闻类别:
    
    用户查询: {last_message}
    
    可选的类别有: general, technology, sports
    只返回类别名称，不要包含其他文字。如果查询中没有明确的类别，返回"general"作为默认值。
    """
    
    try:
        category = lm_model(prompt).strip().lower()
        
        # 确保类别有效
        if category not in ["general", "technology", "sports"]:
            category = "general"  # 使用默认类别
        
        # 执行新闻查询
        news_result = get_latest_news(category)
        
        # 更新state
        new_state = state.copy()
        if "results" not in new_state:
            new_state["results"] = {}
        new_state["results"]["news"] = news_result
        return new_state
    except Exception as e:
        # 如有错误，保持原状态并添加错误信息
        new_state = state.copy()
        new_state["error"] = f"新闻查询错误: {str(e)}"
        return new_state

def generate_response(state: ToolState) -> ToolState:
    """生成最终回复
    
    WHY - 设计思路:
    1. 需要将工具调用结果整合为连贯、有用的回复
    2. 回复应基于原始查询和所有收集的信息，保持上下文一致性
    3. 回复的语气和风格应保持一致，符合AI助手的角色定位
    
    HOW - 实现方式:
    1. 收集state中的所有结果信息(搜索、天气、文件、新闻等)
    2. 将查询和结果提供给LLM，生成自然、连贯的回答
    3. 使用结构化提示词引导LLM生成适当格式的回复
    4. 将生成的回复添加到state的消息历史中
    
    WHAT - 功能作用:
    生成对用户查询的最终回复，整合所有工具调用结果，
    以自然、有帮助的方式向用户呈现信息
    
    Args:
        state: 当前工具状态，包含所有工具执行结果
        
    Returns:
        更新后的工具状态，包含生成的回复
    """
    print("生成回复...")
    
    # 如果有错误，处理错误情况
    if "error" in state and state["error"]:
        error_message = state["error"]
        new_state = state.copy()
        new_state["messages"].append({
            "role": "assistant",
            "content": f"很抱歉，处理您的请求时遇到了问题: {error_message}"
        })
        return new_state
    
    # 获取最近的用户消息和工具结果
    last_message = state["messages"][-1]["content"]
    results = state.get("results", {})
    
    # 整理工具结果
    result_summary = []
    for key, value in results.items():
        result_summary.append(f"{key}: {value}")
    
    # 使用LLM生成回复
    prompt = f"""
    根据用户查询和工具结果生成回复:
    
    用户查询: {last_message}
    
    工具结果:
    {', '.join(result_summary) if result_summary else "没有工具结果"}
    
    请生成一个自然、友好的回复，包含工具提供的信息。
    """
    
    try:
        response = lm_model(prompt)
        
        # 更新state
        new_state = state.copy()
        new_state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return new_state
    except Exception as e:
        # 如有错误，返回错误信息
        new_state = state.copy()
        new_state["messages"].append({
            "role": "assistant",
            "content": f"很抱歉，生成回复时遇到了问题: {str(e)}"
        })
        return new_state

def handle_error(state: ToolState) -> ToolState:
    """处理错误情况
    
    WHY - 设计思路:
    1. 错误处理是健壮系统的关键组成部分，需要优雅地应对各种异常
    2. 用户需要理解错误原因，并得到有用的提示来解决问题
    3. 错误信息需要清晰、有帮助，避免技术性的堆栈跟踪
    
    HOW - 实现方式:
    1. 检查state中的error字段，确定是否存在错误
    2. 针对不同类型的错误生成友好的错误消息
    3. 将错误信息添加到消息历史中，以便用户了解问题
    4. 提供可能的解决方案或替代操作建议
    
    WHAT - 功能作用:
    处理工作流程中可能出现的各种错误，确保系统的稳定性和用户体验，
    即使在出错的情况下也能提供有用的反馈和指导
    
    Args:
        state: 当前工具状态，可能包含错误信息
        
    Returns:
        更新后的工具状态，包含错误处理结果
    """
    print("处理错误...")
    
    # 检查是否有错误信息
    if "error" not in state or not state["error"]:
        return state  # 如果没有错误，直接返回原状态
    
    error_message = state["error"]
    
    # 根据错误类型生成友好的错误消息
    user_friendly_error = error_message
    
    if "找不到文件" in error_message:
        user_friendly_error = "我无法找到您指定的文件。请检查文件路径是否正确，或者尝试提供完整的文件路径。"
    elif "没有权限" in error_message:
        user_friendly_error = "我没有权限访问您指定的文件或目录。请尝试其他位置或联系系统管理员。"
    elif "查询分析错误" in error_message:
        user_friendly_error = "我在理解您的查询时遇到了问题。请尝试重新表述您的问题，或者提供更多的细节。"
    elif "工具选择错误" in error_message:
        user_friendly_error = "我在选择合适的工具时遇到了问题。请尝试更明确地表达您的需求。"
    
    # 更新state，添加错误消息
    new_state = state.copy()
    new_state["messages"].append({
        "role": "assistant",
        "content": f"抱歉，出现了一个问题: {user_friendly_error}"
    })
    
    # 清除错误状态，防止重复处理
    if "error" in new_state:
        del new_state["error"]
    
    return new_state

# ===========================================================
# 第5部分: 图结构与路由
# ===========================================================

def route_tool(state: ToolState) -> str:
    """路由到合适的工具处理节点
    
    WHY - 设计思路:
    1. 需要动态决定下一个处理节点，实现灵活的工具调用流程
    2. 基于选定的工具类型，将执行流路由到专门的处理节点
    3. 路由决策是工具调用图灵活性的关键，使不同查询能走不同路径
    4. 处理无工具或错误情况，提供默认路由路径
    
    HOW - 实现方式:
    1. 检查state中的selected_tools字段，识别选定的工具类型
    2. 根据存在的工具类型返回对应的节点名称
    3. 使用优先级机制处理多工具选择情况，确保每个查询有明确路由
    4. 处理无工具或错误情况，提供默认路由路径
    
    WHAT - 功能作用:
    根据选定的工具类型动态选择下一个处理节点，实现灵活的流程控制，
    使工具调用图能够适应各种用户查询和处理需求
    
    Args:
        state: 当前工具状态，包含工具选择信息
        
    Returns:
        下一个处理节点的名称
    """
    # 检查是否有错误
    if "error" in state and state["error"]:
        return "handle_error"
    
    # 检查选择的工具
    tools = state.get("selected_tools", [])
    
    # 根据工具类型路由到不同节点
    if "search_web" in tools:
        return "search"
    elif "get_weather" in tools:
        return "weather"
    elif "read_file" in tools or "write_file" in tools:
        return "file_op"
    elif "get_latest_news" in tools:
        return "news"
    elif "calculate" in tools:
        return "calculate"
    elif "convert_date" in tools:
        return "convert_date"
    else:
        # 如果没有识别出工具或不需要工具，直接生成回复
        return "respond"

def create_tool_workflow() -> Graph:
    """创建工具调用工作流图
    
    WHY - 设计思路:
    1. 需要构建一个灵活的工作流程来处理不同类型的工具调用
    2. 工作流需要支持条件分支，以便根据工具类型选择不同的处理路径
    3. 图结构使复杂的工具调用过程模块化、可维护和可扩展
    
    HOW - 实现方式:
    1. 使用LangGraph的Graph对象作为工作流容器
    2. 添加所有必要的处理节点，每个节点负责特定的功能
    3. 建立节点之间的边，定义执行流程
    4. 使用条件边和路由函数实现动态决策
    5. 确保所有可能的路径都有明确的定义
    
    WHAT - 功能作用:
    创建一个完整的工具调用工作流图，支持用户查询分析、工具选择、
    动态路由和结果生成，实现智能、灵活的工具调用系统
    
    Returns:
        构建好的LangGraph工作流图
    """
    # 创建工作流图
    workflow = Graph()
    
    # 添加节点
    workflow.add_node("analyze", analyze_user_query)  # 分析用户查询
    workflow.add_node("select", select_tools)  # 选择工具
    workflow.add_node("search", execute_search)  # 搜索
    workflow.add_node("weather", execute_weather_api)  # 天气API
    workflow.add_node("file_op", execute_file_operation)  # 文件操作
    workflow.add_node("news", execute_news_api)  # 新闻API
    workflow.add_node("calculate", calculate)  # 计算器
    workflow.add_node("convert_date", convert_date)  # 日期转换
    workflow.add_node("respond", generate_response)  # 生成回复
    workflow.add_node("handle_error", handle_error)  # 错误处理
    
    # 添加边
    workflow.add_edge("analyze", "select")
    
    # 添加条件边，根据工具选择路由
    workflow.add_conditional_edges(
        "select",
        route_tool,
        {
            "search": "search",
            "weather": "weather",
            "file_op": "file_op",
            "news": "news",
            "calculate": "calculate",
            "convert_date": "convert_date",
            "respond": "respond",
            "handle_error": "handle_error"
        }
    )
    
    # 从各工具节点到回复节点的边
    workflow.add_edge("search", "respond")
    workflow.add_edge("weather", "respond")
    workflow.add_edge("file_op", "respond")
    workflow.add_edge("news", "respond")
    workflow.add_edge("calculate", "respond")
    workflow.add_edge("convert_date", "respond")
    workflow.add_edge("handle_error", END)
    workflow.add_edge("respond", END)
    
    # 编译工作流
    return workflow.compile()

# ===========================================================
# 第6部分: 组合工具链
# ===========================================================

def create_agent_with_tools():
    """创建带工具的代理
    
    使用LangChain的ReAct代理框架
    
    返回:
        Agent执行器
    """
    # 使用LangChain自带的代理框架
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有用的AI助手，可以使用多种工具来回答用户问题。
当你需要获取外部信息时，请使用提供的工具。
请仔细思考哪个工具最适合解决当前问题。
"""),
        ("user", "{input}"),
        ("user", """如果你需要使用工具，请遵循以下格式：

思考: 分析问题，思考需要使用哪个工具
行动: 使用工具名称和输入参数
观察: 工具的输出结果
... (可以有多轮思考-行动-观察)
回答: 最终回答，整合所有信息

如果你不需要使用工具，可以直接回答用户问题。""")
    ])
    
    # 创建ReAct代理
    agent = create_react_agent(llm, available_tools, prompt)
    
    # 创建代理执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=available_tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# ===========================================================
# 第7部分: 运行示例
# ===========================================================

def run_graph_example():
    """运行图结构工具调用示例
    
    基于图结构的工具调用流程
    """
    print("\n===== 基于图结构的工具调用示例 =====")
    
    # 创建图
    graph = create_tool_workflow()
    
    # 示例查询
    test_queries = [
        "北京今天的天气怎么样？",
        "帮我搜索LangGraph框架的信息",
        "在demo_file.txt中写入当前时间",
        "有哪些科技领域的最新新闻？"
    ]
    
    # 运行示例
    for i, query in enumerate(test_queries):
        print(f"\n----- 示例 {i+1}: '{query}' -----")
        
        # 初始化状态
        state = initialize_state()
        state["messages"].append(HumanMessage(content=query))
        
        # 执行图
        final_state = graph.invoke(state)
        
        # 打印结果
        print("\n结果:")
        if final_state["messages"]:
            print(f"AI: {final_state['messages'][-1].content}")
        
        if i < len(test_queries) - 1:
            input("\n按Enter继续下一个示例...")

def run_agent_example():
    """运行代理工具调用示例
    
    基于LangChain Agent的工具调用流程
    """
    print("\n===== 基于Agent的工具调用示例 =====")
    
    # 创建代理
    agent_executor = create_agent_with_tools()
    
    # 示例查询
    test_queries = [
        "北京和上海今天的天气对比如何？",
        "帮我总结一下最近的体育新闻",
        "搜索有关人工智能的最新研究进展"
    ]
    
    # 运行示例
    for i, query in enumerate(test_queries):
        print(f"\n----- 示例 {i+1}: '{query}' -----")
        
        # 执行代理
        try:
            result = agent_executor.invoke({"input": query})
            print("\n结果:")
            print(f"AI: {result['output']}")
        except Exception as e:
            print(f"\n执行出错: {str(e)}")
        
        if i < len(test_queries) - 1:
            input("\n按Enter继续下一个示例...")

# ===========================================================
# 第8部分: 主函数
# ===========================================================

def run_tool_workflow(user_query: str) -> Dict:
    """运行工具调用工作流
    
    WHY - 设计思路:
    1. 需要提供简单的接口来处理用户查询和执行整个工具调用流程
    2. 工作流执行需要管理状态传递和最终结果的提取
    3. 用户交互需要标准化处理，确保输入正确传递给工作流
    
    HOW - 实现方式:
    1. 创建工作流图的实例
    2. 初始化状态，包含用户查询
    3. 执行工作流，让查询流经所有必要的节点
    4. 从最终状态中提取回复和其他相关信息
    5. 返回格式化的结果
    
    WHAT - 功能作用:
    提供一个完整的工具调用执行流程，将用户查询转化为适当的工具调用和回复，
    封装复杂的图执行细节，提供简单的调用接口
    
    Args:
        user_query: 用户的查询文本
        
    Returns:
        包含执行结果的字典
    """
    print(f"处理查询: {user_query}")
    
    # 创建工作流
    tool_workflow = create_tool_workflow()
    
    # 创建初始状态
    initial_state = initialize_state()
    
    # 添加用户查询到消息历史
    initial_state["messages"].append({
        "role": "user",
        "content": user_query
    })
    
    # 执行工作流
    final_state = tool_workflow.invoke(initial_state)
    
    # 获取工作流执行的结果
    messages = final_state.get("messages", [])
    last_response = None
    
    # 找到最后一条助手回复
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_response = msg.get("content")
            break
    
    # 返回执行结果
    return {
        "query": user_query,
        "response": last_response,
        "tools_used": final_state.get("selected_tools", []),
        "execution_log": final_state.get("results", {})
    }

def run_example() -> None:
    """运行示例查询
    
    WHY - 设计思路:
    1. 需要提供预设的查询示例，展示不同类型的工具调用
    2. 示例可以帮助用户理解系统能力和正确的查询方式
    3. 测试不同的查询类型，确保工具调用流程正常工作
    
    HOW - 实现方式:
    1. 定义各种工具类型的示例查询
    2. 依次执行每个示例，并打印结果
    3. 提供明显的分隔符和说明，增强可读性
    4. 展示原始查询、使用的工具和得到的回复
    
    WHAT - 功能作用:
    提供自动化的示例执行，展示工具调用系统的各种功能，
    作为演示和测试的便捷方式
    
    Returns:
        None
    """
    print("\n============= 工具调用示例 =============")
    
    # 定义示例查询
    example_queries = [
        "北京今天的天气怎么样？",
        "搜索关于最新人工智能研究的信息",
        "帮我计算23 * 45 + 67的结果",
        "创建一个名为example.txt的文件，内容是'这是一个测试文件'",
        "阅读文件example.txt的内容",
        "今天有什么科技新闻？",
        "把今天的日期转换成MM/DD/YYYY格式"
    ]
    
    # 运行每个示例
    for i, query in enumerate(example_queries):
        print(f"\n示例 {i+1}: {query}")
        print("-" * 50)
        
        # 执行工作流
        result = run_tool_workflow(query)
        
        # 打印结果
        print(f"使用的工具: {', '.join(result['tools_used']) if result['tools_used'] else '无'}")
        print(f"回复: {result['response']}")
    
    print("\n============= 示例结束 =============")

def main() -> None:
    """主函数
    
    WHY - 设计思路:
    1. 需要一个统一的入口点来运行和展示工具调用系统
    2. 用户可能希望尝试自定义查询，而不仅是预设示例
    3. 程序需要稳定运行，能优雅地处理错误和退出
    
    HOW - 实现方式:
    1. 使用异常处理包装主要执行逻辑，确保错误不会导致崩溃
    2. 首先运行预设示例，展示系统功能
    3. 然后提供交互式模式，接受用户输入的查询
    4. 实现简单的命令处理，如退出程序
    
    WHAT - 功能作用:
    提供程序的主入口点，组织整体执行流程，
    支持自动示例运行和交互式查询，提升用户体验
    
    Returns:
        None
    """
    print("===== LangGraph 工具调用示例 =====")
    
    try:
        # 运行示例
        run_example()
        
        # 交互式模式
        print("\n现在您可以尝试自己的查询 (输入'exit'退出):")
        while True:
            user_input = input("\n请输入您的查询: ")
            if user_input.lower() in ['exit', 'quit', '退出']:
                break
                
            if not user_input.strip():
                continue
                
            try:
                # 处理用户输入
                result = run_tool_workflow(user_input)
                print(f"\n使用的工具: {', '.join(result['tools_used']) if result['tools_used'] else '无'}")
                print(f"回复: {result['response']}")
            except Exception as e:
                print(f"处理查询时出错: {str(e)}")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        print("\n===== 程序结束 =====")

# 如果是主程序，则执行main()
if __name__ == "__main__":
    main() 