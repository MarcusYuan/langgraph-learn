"""
LangGraph 智能客服系统
===================
本示例实现一个基于LangGraph的智能客服系统，包含:
1. 问题分类与路由 - 将用户查询分类并路由到合适的处理节点
2. 知识库集成 - 使用向量数据库检索相关知识
3. 人机协作流程 - 复杂问题自动转人工处理

WHY - 设计思路:
1. 客服系统需要高效处理不同类型的用户查询
2. 需要准确匹配用户问题与知识库内容
3. 需要识别复杂问题并转交人工处理
4. 需要维护会话上下文保持连贯对话
5. 系统应具备可扩展性以添加新的专业领域

HOW - 实现方式:
1. 使用LLM对用户查询进行分类
2. 使用条件边实现智能路由
3. 集成向量数据库进行相关知识检索
4. 设计复杂度评分机制判断是否需要人工介入
5. 使用状态管理维护对话历史

WHAT - 功能作用:
通过本示例，你将学习如何构建一个完整的智能客服系统，
了解问题分类、知识检索、人机协作的实现方法，
以及如何使用LangGraph构建复杂的条件流程。

学习目标:
- 掌握LangGraph中的问题分类与路由机制
- 了解向量数据库与LangGraph的集成方法
- 学习如何实现人机交互的无缝切换
- 理解客服系统中的状态管理策略
"""

import os
import time
import json
import copy
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import datetime
import random
import uuid

# LangGraph相关导入
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 知识库相关导入
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 使用Ollama作为本地LLM
from langchain_ollama import ChatOllama, OllamaEmbeddings

# =================================================================
# 第1部分: 基础组件 - 状态定义与LLM初始化
# =================================================================

class CustomerServiceState(TypedDict):
    """客服系统状态定义
    
    WHY - 设计思路:
    1. 需要存储完整的对话历史
    2. 需要记录当前查询的分类和处理状态
    3. 需要存储知识库检索结果
    4. 需要维护会话元数据
    
    HOW - 实现方式:
    1. 使用TypedDict定义类型安全的状态结构
    2. 包含消息历史、查询类型、检索结果等关键字段
    3. 添加人工处理状态标志
    
    WHAT - 功能作用:
    为整个客服系统提供统一的状态管理接口，
    存储对话过程中的各类信息
    """
    messages: List[Dict[str, Any]]  # 消息历史
    query_type: Optional[str]  # 当前查询类型: 技术支持/账单问题/产品咨询/其他
    retrieved_docs: List[Dict[str, Any]]  # 检索到的知识文档
    complexity_score: float  # 问题复杂度评分(0-1)
    referred_to_human: bool  # 是否已转人工
    metadata: Dict[str, Any]  # 会话元数据

def initialize_state() -> CustomerServiceState:
    """初始化客服状态
    
    WHY - 设计思路:
    1. 需要为每个用户会话提供一致的初始状态
    2. 需要包含适当的系统提示信息
    3. 需要设置默认元数据
    
    HOW - 实现方式:
    1. 创建包含系统欢迎消息的状态
    2. 初始化所有必要字段为默认值
    3. 生成唯一的会话ID和时间戳
    
    WHAT - 功能作用:
    为新的客服会话提供初始状态，确保所有会话
    从相同的起点开始，便于状态管理和追踪
    
    Returns:
        CustomerServiceState: 初始化的状态
    """
    session_id = f"session-{int(time.time())}"
    
    return {
        "messages": [
            {
                "role": "system", 
                "content": "你是一个专业的客服助手，负责回答用户问题并在必要时将用户转接给人工客服。请提供有帮助、准确、友好的回答。"
            },
            {
                "role": "assistant", 
                "content": "您好！我是智能客服助手。请问有什么可以帮助您的？"
            }
        ],
        "query_type": None,
        "retrieved_docs": [],
        "complexity_score": 0.0,
        "referred_to_human": False,
        "metadata": {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
    }

def get_llm():
    """获取语言模型实例
    
    WHY - 设计思路:
    1. 需要集中管理LLM的实例创建
    2. 需要提供灵活的模型选择
    
    HOW - 实现方式:
    1. 尝试加载本地模型
    2. 如果失败则使用远程API
    
    WHAT - 功能作用:
    创建并返回语言模型实例，用于各种处理节点
    
    Returns:
        BaseChatModel: 语言模型实例
    """
    # 首先尝试使用本地模型
    try:
        from langchain_community.llms import Ollama
        return Ollama(model="llama3")
    except:
        # 回退到OpenAI
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(temperature=0.7)
        except:
            # 最后回退到LangGraph自带的模型
            from langgraph_sdk.llm import RunRequestOptionsChain
            return RunRequestOptionsChain(temperature=0.7)

# =================================================================
# 第2部分: 知识库设置 - 向量数据库与检索机制
# =================================================================

# 示例知识库数据
KNOWLEDGE_BASE = [
    {
        "id": "tech-001",
        "category": "技术支持",
        "question": "如何重置我的账户密码？",
        "answer": "您可以通过以下步骤重置密码：1. 访问登录页面 2. 点击'忘记密码' 3. 输入您的注册邮箱 4. 按照邮件中的指引完成重置。如果您没有收到邮件，请检查垃圾邮件文件夹或联系客服。",
        "keywords": ["密码", "重置", "忘记", "登录问题"]
    },
    {
        "id": "tech-002",
        "category": "技术支持",
        "question": "应用无法启动怎么办？",
        "answer": "如果应用无法启动，请尝试以下解决方法：1. 重启设备 2. 检查应用是否需要更新 3. 卸载并重新安装应用 4. 确认设备存储空间是否充足 5. 检查设备系统是否需要更新。如果问题依然存在，请提供设备型号和系统版本，我们的技术团队将进一步协助您。",
        "keywords": ["应用", "启动", "崩溃", "无法打开"]
    },
    {
        "id": "billing-001",
        "category": "账单问题",
        "question": "如何查看我的月度账单？",
        "answer": "查看月度账单的步骤：1. 登录您的账户 2. 点击右上角的'账户'图标 3. 选择'账单与支付' 4. 您可以看到所有历史账单。您还可以在此页面下载PDF版本的账单或设置账单通知偏好。",
        "keywords": ["账单", "月度", "查看", "下载"]
    },
    {
        "id": "billing-002",
        "category": "账单问题",
        "question": "为什么我被多收费了？",
        "answer": "对于账单金额异常，可能有几种原因：1. 订阅计划自动续费 2. 您可能使用了额外的付费服务 3. 可能存在未结清的历史费用。请您登录账户查看详细的账单明细。如有疑问，请提供您的账户ID和具体的账单日期，我们将为您核查具体情况。",
        "keywords": ["多收费", "账单错误", "扣款", "收费问题"]
    },
    {
        "id": "product-001",
        "category": "产品咨询",
        "question": "你们的高级套餐包含哪些功能？",
        "answer": "我们的高级套餐包含以下特色功能：1. 无限存储空间 2. 优先客户支持 3. 高级数据分析工具 4. 自定义报表 5. 多用户协作。相比基础套餐，高级套餐更适合企业用户和有专业需求的个人用户。目前我们还提供14天免费试用，您可以在购买前体验所有高级功能。",
        "keywords": ["高级套餐", "功能", "价格", "比较"]
    },
    {
        "id": "product-002",
        "category": "产品咨询",
        "question": "你们的软件支持哪些平台？",
        "answer": "我们的软件目前支持以下平台：1. Windows 10及以上版本 2. macOS 10.14及以上版本 3. iOS 13及以上版本 4. Android 8.0及以上版本 5. 主流Web浏览器(Chrome, Firefox, Safari, Edge)。所有平台的数据可以云同步，确保您在不同设备上的使用体验一致。",
        "keywords": ["平台", "支持", "兼容性", "系统要求"]
    },
    {
        "id": "general-001",
        "category": "其他",
        "question": "如何联系人工客服？",
        "answer": "您可以通过以下方式联系人工客服：1. 电话热线：400-123-4567（工作时间：周一至周五 9:00-18:00）2. 电子邮件：support@example.com（24小时内回复）3. 在线聊天：点击网站右下角的"在线客服"按钮。如果是紧急问题，建议您使用电话热线直接联系我们。",
        "keywords": ["人工", "客服", "联系", "电话"]
    },
    {
        "id": "general-002",
        "category": "其他",
        "question": "你们的办公地址在哪里？",
        "answer": "我们的主要办公地址如下：总部：北京市海淀区科技园区8号楼5层。上海分部：上海市浦东新区张江高科技园区18号3层。广州分部：广州市天河区天河路385号4层。如您需要访问，建议提前预约，可发送邮件至visit@example.com进行预约安排。",
        "keywords": ["地址", "办公室", "位置", "参观"]
    }
]

def setup_knowledge_base():
    """设置向量知识库
    
    WHY - 设计思路:
    1. 需要将知识库内容转化为向量形式便于检索
    2. 需要支持语义相似度搜索
    3. 需要组织结构化的知识条目
    
    HOW - 实现方式:
    1. 使用文本分割器处理长文本
    2. 使用嵌入模型生成文本向量
    3. 创建FAISS向量存储支持相似度检索
    4. 添加元数据便于过滤和分类
    
    WHAT - 功能作用:
    创建一个向量化的知识库，支持语义搜索，
    为客服系统提供知识检索能力
    
    Returns:
        FAISS: 向量数据库实例
    """
    try:
        # 准备知识条目
        texts = []
        metadatas = []
        
        # 处理知识库数据
        for item in KNOWLEDGE_BASE:
            # 合并问题和答案为一个文档
            text = f"问题: {item['question']}\n回答: {item['answer']}"
            texts.append(text)
            
            # 添加元数据
            metadata = {
                "id": item["id"],
                "category": item["category"],
                "keywords": ", ".join(item["keywords"])
            }
            metadatas.append(metadata)
        
        # 创建嵌入模型
        try:
            # 优先使用Ollama嵌入模型
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            print("使用Ollama嵌入模型")
        except Exception as e:
            # 回退到HuggingFace嵌入模型
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            print(f"使用HuggingFace嵌入模型 (Ollama嵌入错误: {str(e)})")
        
        # 创建向量存储
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        print(f"知识库初始化完成，共 {len(texts)} 条知识条目")
        return vectorstore
    
    except Exception as e:
        print(f"知识库初始化失败: {str(e)}")
        # 如果向量存储创建失败，返回None
        return None

def retrieve_knowledge(query: str, category: Optional[str] = None, vectorstore=None) -> List[Dict[str, Any]]:
    """从知识库检索相关信息
    
    WHY - 设计思路:
    1. 需要基于用户查询检索相关知识
    2. 需要支持分类过滤提高相关性
    3. 需要处理向量存储不可用的情况
    
    HOW - 实现方式:
    1. 使用向量相似度搜索匹配查询
    2. 支持可选的分类过滤
    3. 包含回退机制处理异常情况
    
    WHAT - 功能作用:
    为用户查询检索最相关的知识条目，支持客服系统
    提供准确的回答
    
    Args:
        query: 用户查询文本
        category: 可选的过滤分类
        vectorstore: 向量数据库实例
        
    Returns:
        List[Dict[str, Any]]: 检索到的知识条目列表
    """
    # 如果向量存储不可用，使用关键词匹配回退
    if vectorstore is None:
        return fallback_retrieval(query, category)
    
    try:
        # 构建检索过滤器
        filter_dict = {}
        if category and category != "其他":
            filter_dict["category"] = category
        
        # 执行向量检索
        search_kwargs = {"k": 3}  # 检索前3个最相关结果
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            
        results = vectorstore.similarity_search_with_score(query, **search_kwargs)
        
        # 格式化结果
        docs = []
        for doc, score in results:
            docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return docs
    
    except Exception as e:
        print(f"知识检索失败: {str(e)}")
        return fallback_retrieval(query, category)

def fallback_retrieval(query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """关键词匹配的回退检索方法
    
    WHY - 设计思路:
    1. 需要在向量检索失败时提供备选方案
    2. 需要支持基本的关键词匹配
    
    HOW - 实现方式:
    1. 对查询文本和知识条目进行简单的关键词匹配
    2. 支持分类过滤
    3. 计算简单相似度得分
    
    WHAT - 功能作用:
    作为向量检索的备选方案，确保系统在向量数据库
    不可用时仍能提供基本的检索能力
    
    Args:
        query: 用户查询文本
        category: 可选的过滤分类
        
    Returns:
        List[Dict[str, Any]]: 检索到的知识条目列表
    """
    results = []
    
    # 将查询转为小写并分词
    query_words = set(query.lower().split())
    
    for item in KNOWLEDGE_BASE:
        # 如果指定了分类且不匹配，则跳过
        if category and category != "其他" and item["category"] != category:
            continue
        
        # 计算问题和关键词与查询的匹配度
        question_words = set(item["question"].lower().split())
        keyword_words = set(" ".join(item["keywords"]).lower().split())
        
        # 计算重叠词的数量作为简单的相似度分数
        question_overlap = len(query_words.intersection(question_words))
        keyword_overlap = len(query_words.intersection(keyword_words))
        
        # 总分数是问题匹配和关键词匹配的加权和
        score = question_overlap * 2 + keyword_overlap
        
        if score > 0:
            results.append({
                "content": f"问题: {item['question']}\n回答: {item['answer']}",
                "metadata": {
                    "id": item["id"],
                    "category": item["category"],
                    "keywords": ", ".join(item["keywords"])
                },
                "score": score
            })
    
    # 按匹配分数排序并返回前3个结果
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3]

# =================================================================
# 第3部分: 问题分类与路由 - 支持智能分流
# =================================================================

def classify_query_node(state: CustomerServiceState) -> Dict[str, Any]:
    """对用户查询进行分类
    
    WHY - 设计思路:
    1. 需要将用户查询分类到预定义的类别
    2. 不同类型的查询需要不同的处理方式
    3. 分类信息有助于提高知识检索相关性
    
    HOW - 实现方式:
    1. 提取最新的用户消息
    2. 使用LLM进行分类决策
    3. 更新状态中的查询类型
    
    WHAT - 功能作用:
    分析用户查询，将其分类为技术支持、账单问题、
    产品咨询或其他，为后续处理提供路由依据
    
    Args:
        state: 当前客服状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取最新的用户消息
    messages = state["messages"]
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    
    if not user_messages:
        # 如果没有用户消息，默认为"其他"
        new_state["query_type"] = "其他"
        return new_state
    
    latest_message = user_messages[-1]["content"]
    
    # 使用LLM进行分类
    llm = get_llm()
    
    # 分类提示
    classification_prompt = PromptTemplate.from_template(
        """
        将以下用户查询分类为以下类别之一：技术支持、账单问题、产品咨询、其他
        
        用户查询: {query}
        
        分类(只返回一个类别名称，不要包含任何解释):
        """
    )
    
    # 创建分类链
    classification_chain = classification_prompt | llm | StrOutputParser()
    
    # 执行分类
    query_type = classification_chain.invoke({"query": latest_message})
    
    # 标准化分类结果
    query_type = query_type.strip()
    valid_types = ["技术支持", "账单问题", "产品咨询", "其他"]
    
    if query_type not in valid_types:
        # 如果结果不在预定义类别中，默认为"其他"
        for valid_type in valid_types:
            if valid_type in query_type:
                query_type = valid_type
                break
        else:
            query_type = "其他"
    
    # 更新状态
    new_state["query_type"] = query_type
    print(f"查询已分类为: {query_type}")
    
    return new_state

def evaluate_complexity_node(state: CustomerServiceState) -> Dict[str, Any]:
    """评估问题的复杂度
    
    WHY - 设计思路:
    1. 需要识别复杂问题以便转交人工处理
    2. 需要量化问题复杂度以支持决策
    
    HOW - 实现方式:
    1. 提取最新的用户消息
    2. 使用启发式规则和LLM评估复杂度
    3. 生成0-1之间的复杂度评分
    
    WHAT - 功能作用:
    分析用户查询的复杂度，为人机协作提供决策依据，
    确保复杂问题得到适当处理
    
    Args:
        state: 当前客服状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 获取最新的用户消息
    messages = state["messages"]
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    
    if not user_messages:
        return {"complexity_score": 0.0}
    
    latest_message = user_messages[-1]["content"]
    
    # 简单启发式评分
    heuristic_score = 0.0
    
    # 1. 消息长度 - 较长的消息通常更复杂
    if len(latest_message) > 100:
        heuristic_score += 0.3
    elif len(latest_message) > 50:
        heuristic_score += 0.15
    
    # 2. 问题数量 - 多个问题标记更复杂
    question_marks = latest_message.count('?')
    if question_marks > 2:
        heuristic_score += 0.3
    elif question_marks > 0:
        heuristic_score += 0.1
    
    # 3. 复杂性关键词
    complexity_keywords = [
        "为什么", "怎么解决", "无法理解", "问题", "复杂", "难题",
        "错误", "故障", "bug", "失败", "不工作", "解释", "详细"
    ]
    
    for keyword in complexity_keywords:
        if keyword in latest_message.lower():
            heuristic_score += 0.05
            # 最多加0.25
            if heuristic_score > 0.75:
                break
    
    # 4. 对话历史长度 - 长对话可能表示问题难以解决
    if len(user_messages) > 5:
        heuristic_score += 0.15
    
    # 使用LLM进行评分（简化版，实际应用中可更详细）
    llm = get_llm()
    
    complexity_prompt = PromptTemplate.from_template(
        """
        评估以下用户查询的复杂度，返回0到10之间的分数。
        0表示非常简单，10表示极其复杂。考虑问题的技术性、具体性、需要的上下文等因素。
        
        用户查询: {query}
        
        复杂度评分(只返回数字，不要包含其他文字):
        """
    )
    
    # 创建评分链
    complexity_chain = complexity_prompt | llm | StrOutputParser()
    
    try:
        # 执行评分
        llm_score_text = complexity_chain.invoke({"query": latest_message})
        # 提取数字
        llm_score = 0.0
        for char in llm_score_text:
            if char.isdigit() or char == '.':
                llm_score = float(char) / 10.0
                break
    except Exception as e:
        print(f"复杂度评分错误: {str(e)}")
        llm_score = 0.5  # 默认中等复杂度
    
    # 综合评分 (启发式和LLM各占一半)
    final_score = (heuristic_score + llm_score) / 2.0
    # 确保在0-1范围内
    final_score = max(0.0, min(1.0, final_score))
    
    print(f"问题复杂度评分: {final_score:.2f}")
    
    return {"complexity_score": final_score}

def route_query(state: CustomerServiceState) -> str:
    """基于查询类型和复杂度路由到合适的处理节点
    
    WHY - 设计思路:
    1. 需要根据问题类型和复杂度选择处理路径
    2. 需要支持转人工决策
    
    HOW - 实现方式:
    1. 检查问题是否已转人工
    2. 分析复杂度评分是否超过阈值
    3. 根据查询类型选择处理路径
    
    WHAT - 功能作用:
    作为LangGraph的条件路由函数，决定下一步应该
    处理的节点，确保查询得到最合适的处理
    
    Args:
        state: 当前客服状态
        
    Returns:
        str: 下一个节点的名称
    """
    # 检查是否已经转人工
    if state["referred_to_human"]:
        return "human_agent"
    
    # 检查复杂度是否超过阈值
    if state["complexity_score"] > 0.7:
        return "human_handoff"
    
    # 根据查询类型路由
    query_type = state.get("query_type", "其他")
    
    if query_type == "技术支持":
        return "tech_support"
    elif query_type == "账单问题":
        return "billing"
    elif query_type == "产品咨询":
        return "product_info"
    else:
        return "general_assistant"

# =================================================================
# 第4部分: 处理节点实现 - 专业领域处理
# =================================================================

def tech_support_node(state: CustomerServiceState) -> Dict[str, Any]:
    """技术支持处理节点
    
    WHY - 设计思路:
    1. 需要处理技术相关问题
    2. 需要结合知识库提供准确回答
    
    HOW - 实现方式:
    1. 提取用户查询
    2. 从知识库检索相关技术文档
    3. 生成专业的技术支持回答
    
    WHAT - 功能作用:
    处理技术支持类问题，提供专业的问题解决方案
    
    Args:
        state: 当前客服状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取最新的用户消息
    messages = state["messages"]
    latest_user_message = [msg for msg in messages if msg["role"] == "user"][-1]["content"]
    
    # 获取向量存储
    vectorstore = setup_knowledge_base()
    
    # 检索相关文档
    retrieved_docs = retrieve_knowledge(
        query=latest_user_message,
        category="技术支持",
        vectorstore=vectorstore
    )
    
    # 保存检索到的文档
    new_state["retrieved_docs"] = retrieved_docs
    
    # 准备文档内容
    docs_content = ""
    for i, doc in enumerate(retrieved_docs, 1):
        docs_content += f"\n文档{i}: {doc['content']}\n"
    
    # 使用LLM生成回答
    llm = get_llm()
    
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名专业的技术支持客服。使用以下知识库中的信息回答用户的技术问题。
        如果知识库中没有相关信息，请坦诚地表示，并提供可能的解决方向或建议联系专业技术支持。
        回答应专业、准确、简洁，并以友好的语气。"""),
        ("human", "用户问题: {query}"),
        ("human", "相关知识库信息: {docs_content}"),
    ])
    
    chain = response_prompt | llm
    
    response = chain.invoke({
        "query": latest_user_message,
        "docs_content": docs_content
    })
    
    # 添加AI回复
    new_state["messages"] = [
        {"role": "assistant", "content": response.content}
    ]
    
    return new_state

def billing_node(state: CustomerServiceState) -> Dict[str, Any]:
    """账单问题处理节点
    
    WHY - 设计思路:
    1. 需要处理账单和付款相关问题
    2. 需要结合知识库提供准确的账单信息
    
    HOW - 实现方式:
    1. 提取用户查询
    2. 从知识库检索相关账单文档
    3. 生成专业的账单问题回答
    
    WHAT - 功能作用:
    处理账单相关问题，提供清晰的账单解释和指导
    
    Args:
        state: 当前客服状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取最新的用户消息
    messages = state["messages"]
    latest_user_message = [msg for msg in messages if msg["role"] == "user"][-1]["content"]
    
    # 获取向量存储
    vectorstore = setup_knowledge_base()
    
    # 检索相关文档
    retrieved_docs = retrieve_knowledge(
        query=latest_user_message,
        category="账单问题",
        vectorstore=vectorstore
    )
    
    # 保存检索到的文档
    new_state["retrieved_docs"] = retrieved_docs
    
    # 准备文档内容
    docs_content = ""
    for i, doc in enumerate(retrieved_docs, 1):
        docs_content += f"\n文档{i}: {doc['content']}\n"
    
    # 使用LLM生成回答
    llm = get_llm()
    
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名专业的账单客服。使用以下知识库中的信息回答用户的账单问题。
        如果涉及具体账户详情或无法从知识库中找到答案的问题，建议用户提供账号信息或联系账单专员。
        回答应清晰、准确，并以耐心、体贴的语气。"""),
        ("human", "用户问题: {query}"),
        ("human", "相关知识库信息: {docs_content}"),
    ])
    
    chain = response_prompt | llm
    
    response = chain.invoke({
        "query": latest_user_message,
        "docs_content": docs_content
    })
    
    # 添加AI回复
    new_state["messages"] = [
        {"role": "assistant", "content": response.content}
    ]
    
    return new_state

def product_info_node(state: CustomerServiceState) -> Dict[str, Any]:
    """产品咨询处理节点
    
    WHY - 设计思路:
    1. 需要处理产品功能和特性相关问题
    2. 需要提供准确的产品信息
    
    HOW - 实现方式:
    1. 提取用户查询
    2. 从知识库检索相关产品文档
    3. 生成专业的产品介绍回答
    
    WHAT - 功能作用:
    处理产品咨询问题，提供全面的产品信息和比较
    
    Args:
        state: 当前客服状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取最新的用户消息
    messages = state["messages"]
    latest_user_message = [msg for msg in messages if msg["role"] == "user"][-1]["content"]
    
    # 获取向量存储
    vectorstore = setup_knowledge_base()
    
    # 检索相关文档
    retrieved_docs = retrieve_knowledge(
        query=latest_user_message,
        category="产品咨询",
        vectorstore=vectorstore
    )
    
    # 保存检索到的文档
    new_state["retrieved_docs"] = retrieved_docs
    
    # 准备文档内容
    docs_content = ""
    for i, doc in enumerate(retrieved_docs, 1):
        docs_content += f"\n文档{i}: {doc['content']}\n"
    
    # 使用LLM生成回答
    llm = get_llm()
    
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名产品专家。使用以下知识库中的信息回答用户的产品问题。
        提供全面、准确的产品信息，突出产品优势，但不要过度营销或做出未经证实的承诺。
        如果知识库中没有相关信息，可以提供一般性的产品概述，并建议用户查看产品页面或联系销售团队获取详细信息。
        语气应专业、热情但不过度推销。"""),
        ("human", "用户问题: {query}"),
        ("human", "相关知识库信息: {docs_content}"),
    ])
    
    chain = response_prompt | llm
    
    response = chain.invoke({
        "query": latest_user_message,
        "docs_content": docs_content
    })
    
    # 添加AI回复
    new_state["messages"] = [
        {"role": "assistant", "content": response.content}
    ]
    
    return new_state

def general_assistant_node(state: CustomerServiceState) -> Dict[str, Any]:
    """通用问题处理节点
    
    WHY - 设计思路:
    1. 需要处理不属于特定类别的一般问题
    2. 需要提供全面且友好的回答
    
    HOW - 实现方式:
    1. 提取用户查询
    2. 从全部知识库检索相关文档
    3. 生成通用的助手回答
    
    WHAT - 功能作用:
    处理一般性问题，提供友好的通用回答
    
    Args:
        state: 当前客服状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 获取最新的用户消息
    messages = state["messages"]
    latest_user_message = [msg for msg in messages if msg["role"] == "user"][-1]["content"]
    
    # 获取向量存储
    vectorstore = setup_knowledge_base()
    
    # 检索相关文档 (不限制类别)
    retrieved_docs = retrieve_knowledge(
        query=latest_user_message,
        vectorstore=vectorstore
    )
    
    # 保存检索到的文档
    new_state["retrieved_docs"] = retrieved_docs
    
    # 准备文档内容
    docs_content = ""
    for i, doc in enumerate(retrieved_docs, 1):
        docs_content += f"\n文档{i}: {doc['content']}\n"
    
    # 使用LLM生成回答
    llm = get_llm()
    
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名友好的客服助手。使用以下知识库中的信息回答用户的问题。
        提供有帮助、友好的回答。如果知识库中没有相关信息，请诚实地告知用户，
        并提供可能有用的一般信息或建议联系相关部门。语气应亲切、专业。"""),
        ("human", "用户问题: {query}"),
        ("human", "相关知识库信息: {docs_content}"),
    ])
    
    chain = response_prompt | llm
    
    response = chain.invoke({
        "query": latest_user_message,
        "docs_content": docs_content
    })
    
    # 添加AI回复
    new_state["messages"] = [
        {"role": "assistant", "content": response.content}
    ]
    
    return new_state

# =================================================================
# 第5部分: 人机协作 - 人工客服处理
# =================================================================

def human_handoff_node(state: CustomerServiceState) -> Dict[str, Any]:
    """人工转接处理节点
    
    WHY - 设计思路:
    1. 需要处理系统判断为复杂的问题
    2. 需要提供平滑的人机交接体验
    
    HOW - 实现方式:
    1. 标记状态为已转人工
    2. 生成适当的转接提示信息
    3. 模拟人工接入
    
    WHAT - 功能作用:
    处理复杂问题的人工转接，确保用户获得连续的服务体验
    
    Args:
        state: 当前客服状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 标记为已转人工
    new_state["referred_to_human"] = True
    
    # 获取查询类型
    query_type = state.get("query_type", "一般问题")
    
    # 添加转接消息
    transfer_message = f"""
    您的问题似乎比较复杂，需要专业的{query_type}客服进一步协助。
    我正在将您转接到人工客服，请稍候...
    
    [系统通知] 您已被转接至人工客服队列，预计等待时间3-5分钟。
    """
    
    new_state["messages"] = [
        {"role": "assistant", "content": transfer_message}
    ]
    
    return new_state

def human_agent_node(state: CustomerServiceState) -> Dict[str, Any]:
    """人工客服处理节点
    
    WHY - 设计思路:
    1. 需要模拟人工客服的处理流程
    2. 需要提供更个性化的回答
    
    HOW - 实现方式:
    1. 模拟人工客服回复
    2. 保持人工处理状态
    
    WHAT - 功能作用:
    模拟人工客服处理复杂问题，提供更专业的解答
    
    Args:
        state: 当前客服状态
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 确保保持人工状态
    new_state["referred_to_human"] = True
    
    # 获取最新的用户消息
    messages = state["messages"]
    latest_user_message = [msg for msg in messages if msg["role"] == "user"][-1]["content"]
    
    # 获取查询类型
    query_type = state.get("query_type", "一般问题")
    
    # 模拟人工客服名称
    human_names = {
        "技术支持": ["张技术", "李工程师", "王技术支持"],
        "账单问题": ["刘会计", "陈财务", "赵账单专员"],
        "产品咨询": ["黄产品经理", "吴产品专家", "郑产品顾问"],
        "其他": ["孙客服", "周助理", "钱服务专员"]
    }
    
    category = query_type if query_type in human_names else "其他"
    human_name = random.choice(human_names[category])
    
    # 使用LLM生成人工回答
    llm = get_llm()
    
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一名专业的人工客服{human_name}，专攻{query_type}领域。
        请以人工客服的身份回答用户问题，表现出比AI更高的专业性、同理心和个性化。
        可以使用一些人工客服常用的表达方式，如"我理解您的困扰"、"让我为您查询一下"等。
        回答应详细、专业，体现出人工服务的价值。在回答开头表明你是人工客服{human_name}。"""),
        ("human", "用户问题: {query}"),
    ])
    
    chain = response_prompt | llm
    
    response = chain.invoke({
        "query": latest_user_message
    })
    
    # 添加人工客服回复
    new_state["messages"] = [
        {"role": "assistant", "content": response.content}
    ]
    
    return new_state

def user_input_node(state: CustomerServiceState, message: str) -> Dict[str, Any]:
    """处理用户输入的节点
    
    WHY - 设计思路:
    1. 需要将用户输入整合到状态中
    2. 需要重置部分状态以准备新的处理流程
    
    HOW - 实现方式:
    1. 添加用户消息到历史
    2. 更新时间戳
    3. 重置文档和分类信息
    
    WHAT - 功能作用:
    处理用户输入，更新状态中的消息历史和元数据
    
    Args:
        state: 当前客服状态
        message: 用户输入消息
        
    Returns:
        Dict[str, Any]: 更新后的状态部分
    """
    # 创建状态的副本
    new_state = {}
    
    # 添加用户消息
    new_state["messages"] = [
        {"role": "user", "content": message}
    ]
    
    # 更新元数据
    new_state["metadata"] = {
        "last_updated": datetime.now().isoformat()
    }
    
    # 重置检索文档
    new_state["retrieved_docs"] = []
    
    return new_state

# =================================================================
# 第6部分: 图结构构建
# =================================================================

def build_customer_service_graph():
    """构建客服系统图
    
    WHY - 设计思路:
    1. 需要将各功能节点组织为完整的工作流
    2. 需要实现基于分类的路由逻辑
    3. 需要支持人工转接的流程
    
    HOW - 实现方式:
    1. 创建基于CustomerServiceState的StateGraph
    2. 添加各种处理节点
    3. 实现基于查询类型的条件路由
    4. 设置人工转接逻辑
    
    WHAT - 功能作用:
    构建一个完整的智能客服系统图，整合问题分类、
    知识检索、专业回答和人工协作等功能
    
    Returns:
        StateGraph: 编译后的客服系统图
    """
    # 创建图
    workflow = StateGraph(CustomerServiceState)
    
    # 添加节点
    # 1. 输入和分析节点
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("evaluate_complexity", evaluate_complexity_node)
    
    # 2. 专业处理节点
    workflow.add_node("tech_support", tech_support_node)
    workflow.add_node("billing", billing_node)
    workflow.add_node("product_info", product_info_node)
    workflow.add_node("general_assistant", general_assistant_node)
    
    # 3. 人工协作节点
    workflow.add_node("human_handoff", human_handoff_node)
    workflow.add_node("human_agent", human_agent_node)
    
    # 设置入口点和流程
    # 先对查询进行分类
    workflow.add_edge("user_input", "classify_query")
    # 然后评估复杂度
    workflow.add_edge("classify_query", "evaluate_complexity")
    
    # 根据分类和复杂度路由到相应节点
    workflow.add_conditional_edges(
        "evaluate_complexity",
        route_query,
        {
            "tech_support": "tech_support",
            "billing": "billing",
            "product_info": "product_info",
            "general_assistant": "general_assistant",
            "human_handoff": "human_handoff",
            "human_agent": "human_agent"
        }
    )
    
    # 所有处理完成后结束会话
    for node in ["tech_support", "billing", "product_info", "general_assistant", "human_handoff", "human_agent"]:
        workflow.add_edge(node, END)
    
    # 自定义用户输入节点（接收外部消息）
    def process_user_message(state, input_dict):
        return user_input_node(state, input_dict.get("message", ""))
    
    # 添加用户输入节点
    workflow.add_node("user_input", process_user_message)
    
    # 设置入口点
    workflow.set_entry_point("user_input")
    
    # 编译图
    return workflow.compile(checkpointer=MemorySaver())

# =================================================================
# 第7部分: 主程序入口
# =================================================================

def run_example():
    """运行智能客服系统示例
    
    WHY - 设计思路:
    1. 需要演示系统的完整流程
    2. 需要提供样例对话和交互
    
    HOW - 实现方式:
    1. 构建客服系统图
    2. 初始化客服状态
    3. 发送示例问题
    4. 打印处理结果
    
    WHAT - 功能作用:
    展示智能客服系统的功能和效果，方便用户理解系统的运作方式
    """
    print("=" * 50)
    print("智能客服系统示例")
    print("=" * 50)
    
    # 构建客服图
    print("构建客服系统图...")
    graph = build_customer_service_graph()
    
    # 创建初始状态
    print("初始化客服状态...")
    state = {
        "messages": [],
        "query_type": None,
        "complexity_score": 0.0,
        "referred_to_human": False,
        "retrieved_docs": [],
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "session_id": str(uuid.uuid4()),
        }
    }
    
    # 示例问题
    sample_queries = [
        "我的软件无法启动，显示错误代码E-101，怎么解决？",
        "我想了解一下你们的高级会员套餐包含哪些功能？",
        "我上个月的账单有问题，好像多收费了，能帮我查一下吗？",
        "如何联系你们的技术支持团队？我有一个非常复杂的问题需要专业人员解答。"
    ]
    
    # 处理示例问题
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    for i, query in enumerate(sample_queries, 1):
        print("\n" + "=" * 30)
        print(f"示例 {i}: {query}")
        print("=" * 30)
        
        # 发送问题
        result = graph.invoke({"message": query}, config=config)
        
        # 打印结果
        print("\n用户: ", query)
        for message in result["messages"]:
            if message["role"] == "assistant":
                print("\n客服: ", message["content"])
        
        print("\n查询类型:", result.get("query_type", "未分类"))
        print("复杂度评分:", result.get("complexity_score", 0))
        if result.get("referred_to_human"):
            print("状态: 已转人工处理")
        
        print("-" * 50)
    
    print("\n演示完成!")

if __name__ == "__main__":
    from datetime import datetime
    import uuid
    import random
    from typing import Dict, List, Any, TypedDict, Optional, Tuple
    import os
    import time
    import json
    import copy
    
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("未安装OpenAI embeddings，将使用替代方案")
    
    try:
        import numpy as np
    except ImportError:
        print("未安装numpy，向量检索功能可能受限")
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("未安装LangChain Core，请使用pip install langchain_core安装")
        exit(1)
    
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.saver import MemorySaver
    except ImportError:
        print("未安装LangGraph，请使用pip install langgraph安装")
        exit(1)
    
    # 运行示例
    run_example() 