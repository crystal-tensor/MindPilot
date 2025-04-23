# MindPilot - 认知图谱平台

## 项目概述

MindPilot是一个类似Obsidian的认知图谱平台，通过接入大模型，实现知识图谱生成、思维链可视化、用户画像分析和决策模型支持。该平台分为三个主要功能阶段：

1. **知识图谱生成**：通过与大模型对话，自动生成知识图谱，展示概念之间的关联。
2. **思维链与认知图谱**：当用户提出决策型问题时，将大模型的思维链转化为思维导图，并结合用户画像生成认知图谱。
3. **决策模型**：基于认知图谱，使用决策模型给出帕累托最优解。

## 技术架构

- **前端**：HTML5, CSS3, JavaScript, D3.js (可视化)
- **后端**：Python Flask
- **数据处理**：NetworkX, Matplotlib, NumPy
- **AI集成**：可对接各种大型语言模型API

┌──────────┐         ┌────────────────┐      ┌────────────┐
│ Frontend │←WS/HTTP→│  API Gateway   │←gRPC→│  Auth Svc  │
└──────────┘         └────────────────┘      └────────────┘
       │REST                              ▲
       ▼                                  │
┌────────────┐ Kafka  ┌─────────────┐  GraphQL  ┌────────────┐
│ Ingestion  │──────→│  KG Service │←────────→│  MindMap   │
└────────────┘       └─────────────┘           └────────────┘
       │                                   ▲
       ▼                                   │
┌─────────────┐   gRPC   ┌────────────────┐ │Cypher
│ Cognition   │────────→│ Decision Engine │─┘
│ Fusion Svc  │         └────────────────┘
└─────────────┘
微服务：FastAPI + Uvicorn
异步事件：Kafka (topic: dialog.parsed, kg.updated)
存储：Neo4j (KG), PostgreSQL (user), MinIO (blob)
部署：Docker → Kubernetes (Helm charts); Istio service‑mesh for A/B

## 项目结构

```
/CognitionMap/
├── app.py              # Flask后端应用
├── index.html          # 主页面
├── requirements.txt    # 项目依赖
├── README.md           # 项目说明
├── static/             # 静态资源
│   ├── main.js         # 前端JavaScript
│   └── style.css       # 样式表
```

## 功能特点

### 1. 知识图谱生成
- 通过对话自动提取关键概念和关系
- 生成交互式知识图谱可视化
- 支持图谱节点拖拽和缩放

技术栈 | Technologies
名称	用途	主要用法摘要
LangChain + ChatOpenAI	调用 LLM，输出包含 function‑call 格式的实体‑关系 JSON。	Prompt‑Template → LLM → StructuredOutputParser
spaCy‑Transformer / Bert‑NER	兜底实体识别、防止 LLM 漏检。	Pipeline: nlp.add_pipe("transformer") → add_pipe("ner")
Relation Extraction（Bootstrapped‑RE, LLM‑RE）	找“实体‑关系‑实体”三元组。	零样本 LLM 提示：You are a triple extractor...
Neo4j ↔︎ LangChain KG Store	可持久化、有索引的图数据库。	KG = Neo4jGraph(...), KG.add_triples()

技术流程 | Implementation Flow
Webhook 接收单轮对话 →
LLM + Regex Filter：实体 & 关系抽取 →
冲突检测（Graph Indexing）→ 增量写入 Neo4j →
事件发布到 Kafka（供下游模块消费）。

### 2. 思维链可视化
- 将大模型思考过程转化为可视化思维链
- 展示决策推理的各个步骤和逻辑关系

技术
LLM CoT 捕获：使用 temperature≈0, format=verbose-rationale 强制输出推理链。
树/图解析器：langchain.output_parsers.RoutingParser 将层级标号 (1, 1.1, 1.1.1) 转为树。

可视化：
前端 ECharts Mindmap 或 GoJS
后端 NetworkX ➜ nx.tree_graph() ➜ JSON

实现步骤
Prompt：“请以思维链格式回答，并使用①②③层级”。
Parser 解析层级 → 数据结构 {node, parent}。
合并：与用户画像节点做 G.merge()（NetworkX）。
前端 WebSocket 推送实时 mind‑map JSON。

### 3. 用户画像分析
- 从历史对话中提取用户特征
- 构建用户认知模型和偏好分析

### 4. 认知图谱
- 融合用户画像和思维链
- 提供个性化的认知边界可视化

「KG + MindMap + 目标/资源/愿景 = 认知图谱」
关键技术
功能	技术	说明
资源/目标本体	OWL / SHACL	定义 Goal, Resource, Milestone 类
图融合	RDF Alignment, Graph‑Embedding Merge	用 GraphSAGE / TransE 计算节点相似度后对齐
规划节点排序	Topological Sort + 优先队列	考虑依赖与优先级
跨域查询	GraphQL Federation	单查询同时访问 KG & 画像存储

流程
目标输入表单/对话 → LLM 归一化 (“Obsidian‑style tags”)
资源抓取：从 KG 查询可用知识/联系/工具；
Graph Alignment：对齐后写入 Cognitive Graph Store；
输出 JSON → 渲染 Cytoscape.js 圈层式布局。

### 5. 决策模型
- 多因素决策分析
- 提供帕累托最优解
- 可视化决策权衡和建议

算法栈
模块	库/框架	关键 API
多目标进化	pymoo	problem = Problem(n_var, n_obj) → algorithm = NSGA2()
Nash 求解	nashpy / gambit	g = nash.Game(A, B); g.support_enumeration()
量子加速（可选）	Qiskit Aqua / QAOA	将 payoff → cost Hamiltonian，调用 QuantumInstance
可视化	Plotly Dash	交互式帕累托前沿散点图

运行流程
Cognitive Graph 生成候选策略集合 S；
定义双目标：(Central min risk, Local max vitality) → payoff matrices A, B；
NSGA‑II 求解初始帕累托集合；
局部 Nash 检验：剔除非稳定点；
可选 QAOA 微调最优策略；
REST 返回 {"frontier":[...], "recommended": strategy*}。

## 安装与使用

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动应用

```bash
python app.py
```

应用将在 http://localhost:5000 启动

## 使用指南

1. 打开浏览器访问 http://localhost:5000
2. 在对话框中输入问题或想法
3. 系统会自动生成回复并更新相应的可视化内容
4. 切换标签页查看不同类型的可视化（知识图谱、思维链、认知图谱、决策模型）
5. 可以创建新项目来组织不同主题的对话和分析

## 未来计划

- 支持多用户系统和认知图谱分享
- 增强决策模型的复杂性和准确性
- 添加更多可视化类型和交互方式
- 实现认知图谱的导出和导入功能
- 集成更多大模型API选项

## 贡献

欢迎提交问题和改进建议！