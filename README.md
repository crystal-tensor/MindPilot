# MindPilot - 认知图谱平台

## 项目概述

MindPilot是一个功能完整的认知图谱平台，通过接入大模型（如DeepSeek），实现智能对话、知识图谱生成、思维链可视化、用户画像分析和决策模型支持。该平台提供类似Obsidian的知识管理体验，但专注于AI驱动的认知分析。

## ✨ 核心功能

### 🤖 智能对话系统
- **多模型支持**：集成DeepSeek、OpenAI、Claude等主流AI模型
- **流式响应**：实时打字效果，提升交互体验
- **上下文管理**：保持对话连续性和历史记录

### 📊 可视化分析
- **知识图谱**：自动从对话中提取概念并生成关联图谱
- **思维链**：将AI推理过程可视化为思维导图
- **认知图谱**：结合用户画像的个性化认知分析
- **决策模型**：多因素分析与帕累托最优解可视化

### 💾 对话管理系统
- **本地保存**：支持对话保存到本地JSON文件
- **系统另存为**：原生系统对话框保存体验
- **对话加载**：一键加载历史对话到聊天界面
- **管理功能**：重命名、删除对话记录

### 🎨 用户界面
- **响应式设计**：Bootstrap 5 + 自定义CSS
- **多布局模式**：完整视图/仅聊天/仅可视化
- **可拖拽调整**：鼠标拖拽调整面板大小
- **实时更新**：动态图谱生成和界面刷新

## 🛠️ 技术架构

### 后端技术栈
- **Flask**：轻量级Web框架
- **NetworkX**：图论和网络分析
- **Matplotlib**：数据可视化
- **NumPy**：数值计算
- **OpenAI SDK**：AI模型接口

### 前端技术栈
- **HTML5/CSS3/JavaScript**：现代Web标准
- **Bootstrap 5**：响应式UI框架
- **File System Access API**：原生文件操作

### 项目结构
```
MindPilot/
├── app.py                 # Flask主应用
├── templates/
│   └── index.html         # 主页面模板
├── static/
│   ├── main.js           # 前端JavaScript逻辑
│   └── style.css         # 样式表
├── conversations/        # 对话存储目录
├── requirements.txt      # Python依赖
├── .gitignore           # Git忽略文件
└── README.md            # 项目文档
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/crystal-tensor/MindPilot.git
cd MindPilot

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥
1. 启动应用：`python app.py`
2. 打开浏览器访问：`http://127.0.0.1:5000`
3. 点击右上角"设置"按钮
4. 配置您的API密钥（支持DeepSeek、OpenAI等）

### 3. 开始使用
- **对话聊天**：在底部输入框中输入问题
- **查看图谱**：切换标签查看不同类型的可视化
- **保存对话**：点击"保存当前对话"或"另存为"
- **加载对话**：点击侧边栏中的对话记录

## 📱 功能演示

### 智能对话
- 支持中英文对话
- 自动生成AI回复
- 保持上下文连贯性

### 知识图谱生成
- 从对话中自动提取关键概念
- 生成节点关联图谱
- 基于所有历史对话的综合分析

### 对话管理
- 系统原生另存为对话框
- 一键加载历史对话
- 完整的CRUD操作（创建、读取、更新、删除）

### 界面布局
- 可拖拽调整聊天和可视化面板大小
- 三种布局模式：完整视图/仅聊天/仅可视化
- 响应式设计，适配不同屏幕

## 🎯 使用场景

- **知识管理**：构建个人知识图谱
- **学习研究**：可视化思维过程
- **决策分析**：多因素权衡和最优解分析
- **团队协作**：分享和讨论认知模型
- **创意思考**：探索概念间的潜在联系

## 🔧 高级配置

### 模型提供商配置
支持配置多种AI模型提供商：
- DeepSeek API
- OpenAI GPT系列
- Anthropic Claude
- Google Gemini
- 阿里通义千问
- 百度文心一言

### 可视化参数调整
- 图谱节点大小和颜色
- 边的粗细和透明度
- 布局算法参数
- 图片输出分辨率

## 🌟 特色亮点

1. **原生系统集成**：使用File System Access API实现真正的系统另存为对话框
2. **综合知识图谱**：不仅分析单次对话，更整合所有历史对话生成全局图谱
3. **流畅用户体验**：可拖拽界面、实时更新、智能布局
4. **完整的对话生命周期**：从创建到保存、加载、管理的全流程支持
5. **多维度可视化**：知识图谱、思维链、认知图谱、决策模型四种视角

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发建议
- 遵循Python PEP 8代码规范
- 前端代码使用ES6+语法
- 提交前请测试所有功能模块
- 更新相关文档

## 📄 许可证

MIT License - 详见LICENSE文件

## 🔗 相关链接

- [项目仓库](https://github.com/crystal-tensor/MindPilot)
- [DeepSeek API文档](https://platform.deepseek.com/api-docs/)
- [File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_Access_API)

---

**MindPilot** - 让思维可视化，让决策更智能 🧠✨