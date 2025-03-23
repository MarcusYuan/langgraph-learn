#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 记忆与持久化
===================
本示例讲解LangGraph中的记忆管理与持久化功能:
1. 短期记忆(会话内对话历史)
2. 长期记忆(向量存储)
3. 记忆检索与相关性排序

WHY - 设计思路:
1. AI对话系统需要记住过去的交互以保持上下文连贯
2. 单一对话历史无法满足跨会话的知识保留需求
3. 随着交互增多，需要高效存储和检索相关记忆
4. 不同类型的信息需要不同的记忆存储策略
5. 为了提供个性化体验，系统需要记住用户偏好和过往信息

HOW - 实现方式:
1. 实现基于状态的短期记忆(对话历史)
2. 使用向量数据库存储长期记忆
3. 结合嵌入模型进行语义相似性检索
4. 设计记忆管理策略(存储、检索、更新)
5. 实现记忆持久化机制，确保数据不丢失

WHAT - 功能作用:
通过本示例，你将学习如何在LangGraph中实现短期和长期记忆管理，
使AI系统能够记住过去的交互和知识，提供连贯且个性化的用户体验，
即使在会话间隔或系统重启后也能恢复相关上下文。

学习目标:
- 理解短期记忆与长期记忆的区别与应用场景
- 掌握向量数据库在记忆存储中的应用
- 学习如何实现基于相关性的记忆检索
- 了解记忆持久化策略与实现方法
"""

import os
import json
import time
import uuid
from typing import TypedDict, Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import copy

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_ollama import OllamaLLM

# ===========================================================
# 第1部分: 状态与记忆定义
# ===========================================================

class MemoryState(TypedDict):
    """记忆与持久化状态定义
    
    WHY - 设计思路:
    1. 需要同时支持短期记忆(对话历史)和长期记忆(持久化知识)
    2. 状态需要包含当前查询信息，用于记忆检索
    3. 需要存储检索到的相关记忆，供响应生成参考
    4. 需要跟踪记忆的来源和相关性，便于分析和调试
    5. 支持元数据记录，便于记忆管理和持久化
    
    HOW - 实现方式:
    1. 使用TypedDict提供类型安全和代码提示
    2. 设计messages字段存储短期记忆(当前对话历史)
    3. 设计retrieved_memories字段存储检索到的长期记忆
    4. 添加query字段用于记录当前查询，便于记忆检索
    5. 添加metadata字段存储会话元数据和持久化信息
    
    WHAT - 功能作用:
    提供一个完整的状态结构，同时支持短期和长期记忆管理，
    实现记忆检索、利用和持久化功能，为构建具有记忆能力的
    对话系统提供基础
    """
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # 短期记忆(对话历史)
    query: Optional[str]  # 当前查询，用于记忆检索
    retrieved_memories: List[Document]  # 检索到的相关长期记忆
    metadata: Dict[str, Any]  # 元数据(包含会话ID、时间戳等)

# ===========================================================
# 第2部分: 记忆管理工具
# ===========================================================

class MemoryManager:
    """记忆管理器
    
    WHY - 设计思路:
    1. 需要统一管理短期和长期记忆
    2. 需要提供持久化机制，确保记忆不丢失
    3. 需要高效的存储和检索策略
    4. 需要处理不同类型的记忆数据(文本、结构化数据等)
    5. 需要维护记忆的相关性和时效性
    
    HOW - 实现方式:
    1. 使用向量数据库存储长期记忆
    2. 使用嵌入模型计算文本语义表示
    3. 提供增删改查接口，统一记忆操作
    4. 实现基于相似度的记忆检索
    5. 支持记忆持久化到磁盘和从磁盘加载
    
    WHAT - 功能作用:
    提供一个完整的记忆管理解决方案，包括记忆存储、检索和持久化，
    为对话系统提供记忆能力，使系统能够记住过去的交互并利用相关知识
    """
    
    def __init__(self, persist_directory: str = "./memory_store"):
        """初始化记忆管理器
        
        Args:
            persist_directory: 记忆持久化存储目录
        """
        self.persist_directory = persist_directory
        
        # 初始化嵌入模型
        self.embedding_model = self._initialize_embedding_model()
        
        # 初始化向量存储
        self.vector_store = self._initialize_vector_store()
        
        # 记忆统计
        self.stats = {
            "total_memories": 0,
            "last_added": None,
            "last_retrieved": None
        }
    
    def _initialize_embedding_model(self) -> Embeddings:
        """初始化嵌入模型
        
        WHY - 设计思路:
        1. 需要将文本转换为向量以支持语义检索
        2. 模型需要在本地运行，避免API依赖
        3. 模型需要支持中文等多语言处理
        
        HOW - 实现方式:
        1. 使用HuggingFace提供的预训练嵌入模型
        2. 选择多语言支持的模型确保通用性
        3. 配置适当的模型参数平衡性能和效果
        
        WHAT - 功能作用:
        提供文本向量化能力，将文本转换为语义向量，
        支持基于语义相似度的记忆检索
        
        Returns:
            Embeddings: 初始化好的嵌入模型
        """
        # 使用HuggingFace的多语言嵌入模型
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _initialize_vector_store(self) -> Chroma:
        """初始化向量存储
        
        WHY - 设计思路:
        1. 需要高效存储和检索向量化后的记忆
        2. 需要支持持久化，保证记忆不丢失
        3. 需要支持相似度搜索，基于语义检索记忆
        
        HOW - 实现方式:
        1. 使用Chroma作为向量数据库
        2. 配置持久化目录，确保数据可保存和恢复
        3. 使用嵌入模型作为向量化工具
        
        WHAT - 功能作用:
        提供高效的向量存储和检索功能，作为长期记忆的存储后端，
        支持基于语义相似度的记忆检索
        
        Returns:
            Chroma: 初始化好的向量存储
        """
        # 确保存储目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 初始化Chroma向量存储
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name="long_term_memory"
        )
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加记忆到长期存储
        
        WHY - 设计思路:
        1. 需要将重要信息保存到长期记忆
        2. 需要保存记忆相关的元数据，便于后续检索和管理
        3. 需要生成唯一ID标识每条记忆
        
        HOW - 实现方式:
        1. 生成唯一记忆ID
        2. 添加时间戳等基本元数据
        3. 使用向量存储的add_texts方法存储文本和元数据
        4. 持久化保存确保数据不丢失
        
        WHAT - 功能作用:
        将文本信息添加到长期记忆存储中，使系统能够在未来检索和使用这些信息
        
        Args:
            text: 要存储的文本内容
            metadata: 关联的元数据
            
        Returns:
            str: 记忆ID
        """
        # 生成记忆ID
        memory_id = str(uuid.uuid4())
        
        # 准备元数据
        if metadata is None:
            metadata = {}
        
        # 添加基本元数据
        metadata.update({
            "memory_id": memory_id,
            "created_at": datetime.now().isoformat(),
            "source": "user_interaction"
        })
        
        # 添加到向量存储
        self.vector_store.add_texts(
            texts=[text],
            metadatas=[metadata]
        )
        
        # 更新统计信息
        self.stats["total_memories"] += 1
        self.stats["last_added"] = metadata["created_at"]
        
        # 持久化保存
        self.vector_store.persist()
        
        return memory_id
    
    def retrieve_memories(self, query: str, k: int = 3) -> List[Document]:
        """检索相关记忆
        
        WHY - 设计思路:
        1. 需要基于当前查询检索相关的历史信息
        2. 需要根据语义相似度而非关键词匹配
        3. 检索结果数量应可控，避免信息过载
        
        HOW - 实现方式:
        1. 使用向量存储的similarity_search方法
        2. 将查询文本向量化
        3. 基于向量相似度检索最相关的记忆
        4. 返回Document对象，包含内容和元数据
        
        WHAT - 功能作用:
        根据当前查询检索最相关的历史记忆，为回答生成提供上下文信息
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Document]: 检索到的记忆列表
        """
        # 从向量存储检索相关记忆
        results = self.vector_store.similarity_search(query, k=k)
        
        # 更新统计信息
        self.stats["last_retrieved"] = datetime.now().isoformat()
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息
        
        Returns:
            Dict[str, Any]: 记忆统计信息
        """
        # 更新总记忆数
        self.stats["total_memories"] = self.vector_store._collection.count()
        return self.stats

# 创建全局记忆管理器实例
memory_manager = MemoryManager()

# ===========================================================
# 第3部分: 状态管理函数
# ===========================================================

def initialize_state() -> MemoryState:
    """初始化记忆状态
    
    WHY - 设计思路:
    1. 需要为对话系统提供一个干净的初始状态
    2. 初始状态需要包含基本的元数据和系统提示
    3. 需要初始化空的记忆列表和查询字段
    
    HOW - 实现方式:
    1. 创建包含所有必要字段的MemoryState字典
    2. 添加初始系统消息设置对话基调
    3. 初始化空记忆列表和查询字段
    4. 设置基本元数据如会话ID和创建时间
    
    WHAT - 功能作用:
    提供对话系统的起点状态，初始化所有必要字段，
    为后续的记忆管理和对话流程奠定基础
    
    Returns:
        MemoryState: 初始化的状态
    """
    current_time = datetime.now()
    session_id = f"session-{int(time.time())}"
    
    return {
        "messages": [
            SystemMessage(content="这是一个具有短期和长期记忆能力的对话系统。系统能够记住当前对话中的信息(短期记忆)，也能记住并检索过去存储的重要信息(长期记忆)。")
        ],
        "query": None,
        "retrieved_memories": [],
        "metadata": {
            "session_id": session_id,
            "created_at": current_time.isoformat(),
            "last_updated": current_time.isoformat(),
        }
    }

def add_user_message(state: MemoryState, message: HumanMessage) -> MemoryState:
    """添加用户消息到状态
    
    WHY - 设计思路:
    1. 需要保持状态不变性，每次更新都返回新状态
    2. 用户消息是对话的驱动力，需要特别处理
    3. 用户消息需要保存到短期记忆(对话历史)
    4. 需要提取用户查询，用于后续记忆检索
    
    HOW - 实现方式:
    1. 创建状态的深拷贝，确保不修改原状态
    2. 向消息列表添加用户消息
    3. 将用户消息内容设置为当前查询
    4. 更新元数据时间戳
    
    WHAT - 功能作用:
    处理用户输入，更新短期记忆，并准备查询条件用于记忆检索
    
    Args:
        state: 当前状态
        message: 用户消息
        
    Returns:
        MemoryState: 更新后的新状态
    """
    # 创建状态的深拷贝，确保不可变性
    new_state = copy.deepcopy(state)
    
    # 添加用户消息到对话历史(短期记忆)
    new_state["messages"].append(message)
    
    # 设置当前查询
    new_state["query"] = message.content
    
    # 更新元数据
    new_state["metadata"]["last_updated"] = datetime.now().isoformat()
    
    return new_state

def add_ai_message(state: MemoryState, message: AIMessage) -> MemoryState:
    """添加AI消息到状态
    
    WHY - 设计思路:
    1. 需要保持状态不变性，返回新状态
    2. AI消息也是短期记忆的一部分，需要保存
    3. AI消息可能包含重要信息，需要考虑是否添加到长期记忆
    
    HOW - 实现方式:
    1. 创建状态的深拷贝，确保不修改原状态
    2. 向消息列表添加AI消息
    3. 更新元数据时间戳
    4. 清空查询和检索记忆，准备下一轮交互
    
    WHAT - 功能作用:
    处理AI响应，更新短期记忆，并重置查询状态
    
    Args:
        state: 当前状态
        message: AI消息
        
    Returns:
        MemoryState: 更新后的新状态
    """
    # 创建状态的深拷贝，确保不可变性
    new_state = copy.deepcopy(state)
    
    # 添加AI消息到对话历史(短期记忆)
    new_state["messages"].append(message)
    
    # 重置查询和检索记忆
    new_state["query"] = None
    new_state["retrieved_memories"] = []
    
    # 更新元数据
    new_state["metadata"]["last_updated"] = datetime.now().isoformat()
    
    return new_state

# ===========================================================
# 第4部分: 节点函数
# ===========================================================

def get_llm():
    """获取LLM实例
    
    WHY - 设计思路:
    1. 需要一个可复用的LLM获取函数
    2. 便于统一配置和更换底层模型
    
    HOW - 实现方式:
    1. 使用langchain_ollama提供本地LLM能力
    2. 配置合适的参数确保输出质量
    
    WHAT - 功能作用:
    提供一个配置好的LLM实例，供各节点使用，
    确保整个对话系统使用相同的底层模型配置
    
    Returns:
        OllamaLLM: LLM实例
    """
    return OllamaLLM(
        model="qwen:0.5b",  # 可替换为其他可用模型
        temperature=0.7,
    )

def retrieve_memories_node(state: MemoryState) -> MemoryState:
    """记忆检索节点
    
    WHY - 设计思路:
    1. 需要根据当前查询检索相关的长期记忆
    2. 检索结果需要添加到状态，供后续响应生成参考
    3. 只有在存在查询时才进行检索，避免不必要的操作
    
    HOW - 实现方式:
    1. 检查状态中是否有当前查询
    2. 使用记忆管理器检索相关记忆
    3. 将检索结果添加到状态中
    4. 保持状态不变性，返回新状态
    
    WHAT - 功能作用:
    根据当前用户查询检索相关的长期记忆，为AI响应提供更多上下文信息
    
    Args:
        state: 当前状态
        
    Returns:
        MemoryState: 更新后的新状态
    """
    # 创建状态的深拷贝，确保不可变性
    new_state = copy.deepcopy(state)
    
    # 检查是否有查询
    if not new_state["query"]:
        return new_state
    
    # 检索相关记忆
    query = new_state["query"]
    retrieved_docs = memory_manager.retrieve_memories(query, k=3)
    
    # 更新状态
    new_state["retrieved_memories"] = retrieved_docs
    
    print(f"已检索 {len(retrieved_docs)} 条相关记忆")
    
    return new_state

def generate_response_node(state: MemoryState) -> MemoryState:
    """响应生成节点
    
    WHY - 设计思路:
    1. 需要基于当前查询和检索到的记忆生成响应
    2. 响应应考虑短期记忆(对话历史)和长期记忆(检索结果)
    3. 生成的响应需要添加到对话历史
    
    HOW - 实现方式:
    1. 构建提示模板，包含对话历史和检索到的记忆
    2. 使用LLM生成响应
    3. 将响应添加到状态
    4. 考虑是否将重要信息添加到长期记忆
    
    WHAT - 功能作用:
    综合利用短期和长期记忆，生成连贯且信息丰富的响应
    
    Args:
        state: 当前状态
        
    Returns:
        MemoryState: 更新后的新状态
    """
    # 获取LLM
    llm = get_llm()
    
    # 构建提示模板
    template = """你是一个具有记忆能力的AI助手。

对话历史:
{chat_history}

"""
    
    # 如果有检索到的记忆，添加到提示中
    if state["retrieved_memories"]:
        template += """根据你的记忆，以下是与当前问题相关的信息:
{relevant_memories}

"""
    
    template += """用户当前问题: {query}

请根据所有信息提供有帮助的回答。如果记忆中的信息与问题相关，请合理地融入到回答中，但不要明确提及"根据我的记忆"等字样。"""
    
    # 格式化对话历史
    chat_history = "\n".join([
        f"{'用户' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in state["messages"][-5:]  # 只使用最近5条消息作为上下文
    ])
    
    # 格式化检索到的记忆
    relevant_memories = ""
    if state["retrieved_memories"]:
        relevant_memories = "\n\n".join([
            f"记忆 {i+1}:\n{doc.page_content}\n(来源时间: {doc.metadata.get('created_at', 'Unknown')})"
            for i, doc in enumerate(state["retrieved_memories"])
        ])
    
    # 构建提示
    prompt_inputs = {
        "chat_history": chat_history,
        "query": state["query"],
    }
    
    if state["retrieved_memories"]:
        prompt_inputs["relevant_memories"] = relevant_memories
    
    # 生成响应
    prompt = ChatPromptTemplate.from_template(template)
    response = llm.invoke(prompt.format(**prompt_inputs))
    
    # 将重要信息添加到长期记忆
    # 此处使用简单启发式规则：如果响应长度超过100字符，认为包含重要信息
    if len(response.content) > 100:
        memory_manager.add_memory(
            text=response.content,
            metadata={
                "source": "ai_response",
                "query": state["query"],
                "session_id": state["metadata"]["session_id"]
            }
        )
    
    # 添加响应到状态
    new_state = add_ai_message(state, response)
    
    return new_state

def store_user_memory_node(state: MemoryState) -> MemoryState:
    """存储用户信息到长期记忆节点
    
    WHY - 设计思路:
    1. 需要选择性地将用户输入存储到长期记忆
    2. 不是所有用户输入都值得长期保存
    3. 存储时需要保留相关上下文和元数据
    
    HOW - 实现方式:
    1. 获取最后一条用户消息
    2. 使用启发式规则判断是否值得保存
    3. 调用记忆管理器存储到长期记忆
    4. 保持状态不变性，返回不变的状态
    
    WHAT - 功能作用:
    将重要的用户信息保存到长期记忆，使系统能够在未来检索和使用这些信息
    
    Args:
        state: 当前状态
        
    Returns:
        MemoryState: 原状态(此节点不修改状态)
    """
    # 获取最后一条用户消息
    last_user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_message = msg
            break
    
    # 如果没有用户消息，直接返回原状态
    if not last_user_message:
        return state
    
    # 判断是否值得存储
    # 此处使用简单启发式规则：如果消息长度超过30字符，认为包含重要信息
    if len(last_user_message.content) > 30:
        # 存储前2条和后2条消息作为上下文
        context_messages = state["messages"][-5:]
        context = "\n".join([
            f"{'用户' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            for msg in context_messages
        ])
        
        # 添加到长期记忆
        memory_manager.add_memory(
            text=last_user_message.content,
            metadata={
                "source": "user_input",
                "context": context,
                "session_id": state["metadata"]["session_id"]
            }
        )
        
        print("已将用户输入添加到长期记忆")
    
    # 返回原状态，不做修改
    return state

def user_input_node(state: MemoryState) -> MemoryState:
    """用户输入处理节点
    
    WHY - 设计思路:
    1. 需要处理用户输入，包括常规问题和特殊命令
    2. 用户输入需要保存到短期记忆(对话历史)
    3. 特殊命令可能需要直接操作记忆系统
    
    HOW - 实现方式:
    1. 从命令行获取用户输入
    2. 检查是否是特殊命令
    3. 处理常规输入，添加到状态
    4. 返回更新后的状态
    
    WHAT - 功能作用:
    处理用户输入，更新短期记忆，处理特殊命令
    
    Args:
        state: 当前状态
        
    Returns:
        MemoryState: 更新后的新状态
    """
    # 获取用户输入
    user_input = input("\n用户输入> ")
    
    # 检查特殊命令
    if user_input.startswith("/"):
        # 处理特殊命令
        if user_input == "/stats":
            # 显示记忆统计信息
            stats = memory_manager.get_stats()
            print("\n=== 记忆统计信息 ===")
            print(f"总记忆数: {stats['total_memories']}")
            print(f"最后添加时间: {stats['last_added']}")
            print(f"最后检索时间: {stats['last_retrieved']}")
            print("=====================")
            
            # 返回原状态，不做修改
            return state
        
        elif user_input == "/clear":
            # 清空短期记忆(对话历史)，但保留系统消息
            new_state = copy.deepcopy(state)
            system_messages = [msg for msg in new_state["messages"] if isinstance(msg, SystemMessage)]
            new_state["messages"] = system_messages
            new_state["query"] = None
            new_state["retrieved_memories"] = []
            
            print("已清空当前对话历史")
            return new_state
    
    # 处理常规用户输入
    message = HumanMessage(content=user_input)
    new_state = add_user_message(state, message)
    
    return new_state

# ===========================================================
# 第5部分: 图构建
# ===========================================================

def create_memory_graph():
    """创建支持记忆功能的图
    
    WHY - 设计思路:
    1. 需要一个支持记忆管理的图结构
    2. 图结构需要包含记忆检索和存储节点
    3. 需要定义合适的节点执行顺序
    4. 确保用户输入被适当存储和利用
    5. 需要高效的记忆检索和响应生成流程
    
    HOW - 实现方式:
    1. 创建基于MemoryState的StateGraph
    2. 添加用户输入、记忆检索、响应生成等节点
    3. 添加记忆存储节点保存重要信息
    4. 定义节点间的边，形成完整的执行流程
    5. 设置入口点为用户输入节点
    
    WHAT - 功能作用:
    提供一个支持记忆管理的对话图结构，能够检索和利用长期记忆，
    保存重要信息到长期存储，为对话系统提供记忆能力
    
    Returns:
        StateGraph: 编译好的图实例
    """
    # 创建状态图
    workflow = StateGraph(MemoryState)
    
    # 添加节点
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("retrieve_memories", retrieve_memories_node)
    workflow.add_node("store_user_memory", store_user_memory_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # 设置边 - 定义执行流程
    workflow.add_edge("user_input", "retrieve_memories")
    workflow.add_edge("retrieve_memories", "store_user_memory")
    workflow.add_edge("store_user_memory", "generate_response")
    workflow.add_edge("generate_response", "user_input")
    
    # 设置入口点
    workflow.set_entry_point("user_input")
    
    # 编译图
    return workflow.compile()

# ===========================================================
# 第6部分: 示例运行
# ===========================================================

def run_memory_example():
    """运行记忆管理示例
    
    WHY - 设计思路:
    1. 需要展示记忆管理和持久化的完整流程
    2. 需要演示短期和长期记忆的不同作用
    3. 用户需要能够交互式体验记忆功能
    
    HOW - 实现方式:
    1. 创建支持记忆的图实例
    2. 初始化状态
    3. 开始交互式对话，展示记忆功能
    4. 提供说明和示例用法
    
    WHAT - 功能作用:
    提供记忆管理和持久化的交互式演示，展示系统如何记住和使用信息
    """
    print("\n===== LangGraph 记忆与持久化示例 =====")
    
    # 创建图实例
    graph = create_memory_graph()
    
    # 初始化状态
    state = initialize_state()
    
    print("\n本示例演示LangGraph中的记忆管理和持久化功能。")
    print("系统包含短期记忆(当前对话历史)和长期记忆(向量存储的历史信息)。")
    print("与系统交流时，它会记住重要信息并在未来回答中利用这些记忆。")
    print("\n可用特殊命令:")
    print("  /stats - 显示记忆统计信息")
    print("  /clear - 清空当前对话历史(短期记忆)")
    print("\n建议体验流程:")
    print("1. 先告诉系统一些个人信息(如喜好、职业等)")
    print("2. 聊一些其他话题")
    print("3. 过后再问与之前信息相关的问题，看系统是否记得")
    print("4. 使用/stats查看记忆统计")
    
    print("\n开始对话 (输入'退出'结束)...")
    
    # 交互式对话
    try:
        while True:
            # 进行一轮对话
            state = graph.invoke(state)
            
            # 检查是否退出
            if len(state["messages"]) > 1 and isinstance(state["messages"][-2], HumanMessage):
                if state["messages"][-2].content.lower() == "退出":
                    break
    
    except KeyboardInterrupt:
        print("\n用户中断，结束对话")
    
    print("\n===== 记忆与持久化示例结束 =====")

def main():
    """主函数 - 执行示例
    
    WHY - 设计思路:
    1. 需要一个统一的入口点运行记忆示例
    2. 需要适当的错误处理确保示例稳定运行
    3. 需要提供清晰的开始和结束提示
    
    HOW - 实现方式:
    1. 使用try-except包装主要执行逻辑
    2. 提供开始和结束提示
    3. 调用具体示例函数
    4. 总结关键学习点
    
    WHAT - 功能作用:
    作为程序入口点，执行记忆管理和持久化示例，
    确保示例执行的稳定性，增强用户学习体验
    """
    print("===== LangGraph 记忆与持久化学习示例 =====\n")
    
    try:
        # 运行记忆示例
        run_memory_example()
        
        print("\n===== 示例结束 =====")
        print("通过本示例，你学习了如何:")
        print("1. 设计支持短期和长期记忆的状态结构")
        print("2. 使用向量数据库存储和检索长期记忆")
        print("3. 实现基于相关性的记忆检索")
        print("4. 构建具有记忆能力的对话系统")
        print("5. 实现记忆持久化，确保数据不丢失")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 如果直接运行此脚本
if __name__ == "__main__":
    main() 