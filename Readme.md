# LangGraph React风格应用示例

本项目展示了如何使用LangGraph构建模拟React组件生命周期的应用程序。通过这些示例，您可以了解如何将React的核心概念（如状态管理、组件渲染和副作用处理）应用到LangGraph框架中。

## 项目结构

项目包含三个主要示例：

1. **基础计数器应用** (`react_langgraph_breakpoint.py`) - 带断点调试功能的简单计数器
2. **待办事项应用** (`react_langgraph_todo.py`) - 更复杂的待办事项管理器，展示了更完整的React模式
3. **流式计数器应用** (`react_langgraph_streaming.py`) - 展示了如何实现流式UI更新的计数器

## 核心概念图解

### 基本状态流转图

所有示例都遵循类似的状态流转模式，模拟React的渲染循环：

```mermaid
stateDiagram-v2
    [*] --> parse_input: 用户输入
    parse_input --> update_state: 解析命令
    update_state --> render: 更新状态
    note right of update_state: 断点位置
    render --> generate_response: 渲染UI
    generate_response --> [*]: 返回响应
    
    %% 状态定义
    state "ReactState" as state_def {
        messages: List[BaseMessage] -- 消息历史
        count: int -- 计数器值
        ui: str -- 渲染的UI文本
        user_input: Optional[str] -- 用户输入
    }
```

### 待办事项应用流程图

待办事项应用实现了更完整的React模式，包括副作用处理：

```mermaid
stateDiagram-v2
    [*] --> parse_input: 用户输入
    parse_input --> use_state: 解析命令
    use_state --> action_handler: 获取状态
    action_handler --> use_effect: 处理动作
    use_effect --> render: 处理副作用
    render --> generate_response: 渲染UI
    generate_response --> [*]: 返回响应
    
    %% 状态定义
    state "ReactComponentState" as state_def {
        messages: List[BaseMessage] -- 消息历史
        todos: List[TodoItem] -- 待办事项列表
        input_value: str -- 输入框值
        effects: List[Dict] -- 副作用列表
        ui: str -- 渲染的UI文本
        action: Optional[str] -- 当前动作
        action_params: Optional[Dict] -- 动作参数
    }
    
    %% 动作处理流程
    state action_handler {
        [*] --> check_action
        check_action --> add_todo: action == "add_todo"
        check_action --> toggle_todo: action == "toggle_todo"
        check_action --> delete_todo: action == "delete_todo"
        check_action --> set_input: action == "set_input"
        check_action --> clear_todos: action == "clear_todos"
        add_todo --> [*]
        toggle_todo --> [*]
        delete_todo --> [*]
        set_input --> [*]
        clear_todos --> [*]
    }
```

### 流式处理应用流程图

流式处理应用展示了如何实现异步流式UI更新：

```mermaid
stateDiagram-v2
    [*] --> parse_input: 用户输入
    parse_input --> update_state: 解析命令
    update_state --> render: 更新状态
    note right of update_state: 流式输出
    render --> generate_response: 渲染UI
    generate_response --> [*]: 返回响应
    
    %% 状态定义
    state "ReactStreamState" as state_def {
        messages: List[BaseMessage] -- 消息历史
        counter: int -- 计数器值
        ui: str -- 渲染的UI文本
        user_input: Optional[str] -- 用户输入
        processing: bool -- 处理状态标志
    }
    
    %% 流式处理详情
    state update_state {
        [*] --> start_processing
        start_processing --> increment: user_input == "increment"
        start_processing --> decrement: user_input == "decrement"
        start_processing --> reset: user_input == "reset"
        increment --> [*]: 流式更新UI
        decrement --> [*]: 流式更新UI
        reset --> [*]: 流式更新UI
    }
```

## 功能说明

### 1. 基础计数器应用 (带断点)

这个应用实现了一个简单的计数器，并展示了LangGraph的断点功能：

- **功能**：增加、减少和重置计数器
- **特色**：在`update_state`节点后设置断点，允许用户在执行过程中检查和修改状态
- **状态结构**：
  - `messages`: 消息历史
  - `count`: 计数器值
  - `ui`: 渲染的UI文本
  - `user_input`: 用户输入

### 2. 待办事项应用

这个应用实现了一个更复杂的待办事项管理器，展示了更完整的React模式：

- **功能**：添加、切换、删除和清空待办事项
- **特色**：实现了React的`useState`和`useEffect`钩子模式
- **状态结构**：
  - `messages`: 消息历史
  - `todos`: 待办事项列表
  - `input_value`: 输入框值
  - `effects`: 副作用列表
  - `ui`: 渲染的UI文本
  - `action`: 当前动作
  - `action_params`: 动作参数

### 3. 流式计数器应用

这个应用展示了如何实现流式UI更新：

- **功能**：增加、减少和重置计数器，带有动画效果
- **特色**：使用异步迭代器实现流式UI更新，模拟处理过程中的状态变化
- **状态结构**：
  - `messages`: 消息历史
  - `counter`: 计数器值
  - `ui`: 渲染的UI文本
  - `user_input`: 用户输入
  - `processing`: 处理状态标志

## 设计思想

### React模式在LangGraph中的应用

这些示例展示了如何在LangGraph中实现React的核心设计理念：

1. **状态隔离**：每个组件维护自己的状态，通过TypedDict定义状态结构

2. **单向数据流**：数据沿着图的边单向流动，从输入解析到状态更新，再到渲染和响应生成

3. **声明式UI**：UI是状态的函数，通过`render`函数将当前状态转换为UI表示

4. **副作用处理**：使用类似React的`useEffect`模式处理副作用，如日志记录和数据持久化

5. **组件生命周期**：通过图的节点和边模拟React组件的生命周期，从状态初始化到渲染和更新

### LangGraph与React模式对比图

```mermaid
graph LR
    subgraph "React组件生命周期"
        R_Init["初始化"] --> R_Render["渲染"] 
        R_Render --> R_Effect["副作用"]
        R_Effect --> R_Event["事件处理"]
        R_Event --> R_State["状态更新"]
        R_State --> R_Render
    end
    
    subgraph "LangGraph状态流转"
        L_Start["START"] --> L_Parse["parse_input"]
        L_Parse --> L_State["use_state/update_state"]
        L_State --> L_Action["action_handler"]
        L_Action --> L_Effect["use_effect"]
        L_Effect --> L_Render["render"]
        L_Render --> L_Response["generate_response"]
        L_Response --> L_End["END"]
    end
    
    %% 概念映射关系
    R_Init -.-> L_Start
    R_State -.-> L_State
    R_Event -.-> L_Action
    R_Effect -.-> L_Effect
    R_Render -.-> L_Render
```

### LangGraph特性的应用

这些示例还展示了LangGraph的一些关键特性：

1. **状态图**：使用`StateGraph`定义应用的状态转换逻辑

2. **断点调试**：在`react_langgraph_breakpoint.py`中展示了如何使用断点暂停执行并检查/修改状态

3. **流式处理**：在`react_langgraph_streaming.py`中展示了如何使用异步迭代器实现流式UI更新

4. **状态持久化**：使用`MemorySaver`实现状态的持久化和恢复

### 状态管理详细图

```mermaid
classDiagram
    class ReactState {
        +List~BaseMessage~ messages
        +int count
        +str ui
        +Optional~str~ user_input
    }
    
    class ReactComponentState {
        +List~BaseMessage~ messages
        +List~TodoItem~ todos
        +str input_value
        +List~Dict~ effects
        +str ui
        +Optional~str~ action
        +Optional~Dict~ action_params
    }
    
    class ReactStreamState {
        +List~BaseMessage~ messages
        +int counter
        +str ui
        +Optional~str~ user_input
        +bool processing
    }
    
    class TodoItem {
        +str id
        +str text
        +bool completed
    }
    
    ReactComponentState -- TodoItem : contains
```

### 断点调试流程图

```mermaid
sequenceDiagram
    participant User as 用户
    participant App as LangGraph应用
    participant Graph as StateGraph
    participant Checkpointer as MemorySaver
    
    User->>App: 输入命令
    App->>Graph: 调用图执行
    Graph->>Graph: parse_input
    Graph->>Graph: update_state
    Graph-->>Checkpointer: 保存状态(断点)
    Graph-->>App: 暂停执行
    App-->>User: 显示断点信息
    User->>App: 修改状态(可选)
    App->>Checkpointer: 更新状态
    App->>Graph: 继续执行
    Graph->>Graph: render
    Graph->>Graph: generate_response
    Graph-->>App: 返回最终状态
    App-->>User: 显示响应
```

### 流式处理详细图

```mermaid
sequenceDiagram
    participant User as 用户
    participant App as LangGraph应用
    participant Graph as StateGraph
    participant UI as 用户界面
    
    User->>App: 输入命令
    App->>Graph: 调用图执行(stream=True)
    Graph->>Graph: parse_input
    Graph->>Graph: update_state开始
    loop 流式更新
        Graph-->>UI: 发送UI更新块
        UI-->>User: 显示实时更新
    end
    Graph->>Graph: update_state完成
    Graph->>Graph: render
    Graph->>Graph: generate_response
    Graph-->>App: 返回最终状态
    App-->>User: 显示完整响应
```

## 总结

这些示例展示了如何将React的设计理念应用到LangGraph框架中，创建具有清晰状态管理和UI渲染逻辑的应用程序。通过模拟React的组件生命周期和状态管理模式，这些示例提供了一种结构化的方法来构建复杂的LangGraph应用。

无论您是想构建简单的计数器还是复杂的待办事项管理器，这些示例都提供了可扩展的模式，可以应用于各种LangGraph应用场景。#   l a n g g r a p h - l e a r n  
 