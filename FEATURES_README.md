# MindPilot 新功能说明

本文档详细介绍了MindPilot认知图谱平台新增的三个主要功能。

## 功能概览

### 1. 对话保存与管理功能
- ✅ 保存当前对话到本地文件
- ✅ 自定义对话名称
- ✅ 选择保存路径
- ✅ 加载已保存的对话
- ✅ 重命名对话
- ✅ 删除对话

### 2. 知识图谱生成功能
- ✅ 基于当前对话生成知识图谱
- ✅ 基于保存的对话生成知识图谱
- ✅ 关键词提取和共现分析
- ✅ 可视化节点和边的关系
- ✅ 支持重新生成图谱

### 3. 窗口布局调整功能
- ✅ 完整视图（对话+可视化）
- ✅ 仅对话视图
- ✅ 仅可视化视图
- ✅ 鼠标拖拽调整窗口大小
- ✅ 可调整分隔条

## 详细功能说明

### 对话保存与管理

#### 保存对话
1. 点击左侧边栏的"保存当前对话"按钮
2. 在弹出的模态框中输入对话名称
3. 选择保存路径（默认为`conversations`文件夹）
4. 点击"保存"按钮完成保存

#### 加载对话
1. 点击"加载对话"按钮刷新对话列表
2. 在对话列表中点击文件夹图标加载特定对话
3. 对话历史将显示在聊天窗口中

#### 管理对话
- **重命名**: 点击铅笔图标，输入新名称
- **生成图谱**: 点击图谱图标，基于该对话生成知识图谱
- **删除**: 点击垃圾桶图标删除对话（需确认）

### 知识图谱生成

#### 自动生成
- 每次发送消息后，系统会自动生成四种可视化图表：
  - 知识图谱
  - 思维链
  - 认知图谱
  - 决策模型

#### 手动生成
- 点击可视化区域右上角的"重新生成"按钮
- 可以基于保存的对话生成知识图谱

#### 图谱特性
- **节点大小**: 反映关键词出现频率
- **节点颜色**: 区分高频和低频关键词
- **边的粗细**: 表示关键词共现强度
- **布局算法**: 使用Spring布局优化显示效果

### 窗口布局调整

#### 布局模式
1. **完整视图**: 同时显示对话和可视化区域
2. **仅对话**: 隐藏可视化区域，专注于对话
3. **仅可视化**: 隐藏对话区域，专注于图表分析

#### 大小调整
- 在完整视图模式下，可以拖拽中间的分隔条调整两个区域的大小
- 每个区域都有最小高度限制（200px）
- 支持CSS resize属性进行微调

## API接口说明

### 对话管理API

#### 保存对话
```http
POST /api/save_conversation
Content-Type: application/json

{
    "user_id": "用户ID",
    "conversation_name": "对话名称",
    "save_path": "保存路径"
}
```

#### 加载对话列表
```http
GET /api/load_conversations?path=conversations
```

#### 加载特定对话
```http
GET /api/load_conversation/{conversation_id}?path=conversations
```

#### 重命名对话
```http
POST /api/rename_conversation
Content-Type: application/json

{
    "conversation_id": "对话ID",
    "new_name": "新名称",
    "path": "对话路径"
}
```

#### 删除对话
```http
DELETE /api/delete_conversation
Content-Type: application/json

{
    "conversation_id": "对话ID",
    "path": "对话路径"
}
```

### 知识图谱API

#### 生成知识图谱
```http
POST /api/generate_knowledge_graph
Content-Type: application/json

{
    "user_id": "用户ID",
    "conversation_id": "对话ID（可选）",
    "path": "对话路径（可选）"
}
```

## 文件结构

```
MindPilot/
├── app.py                 # 主应用文件
├── templates/
│   └── index.html        # 前端模板
├── static/
│   ├── main.js           # 主要JavaScript逻辑
│   └── style.css         # 样式文件
├── conversations/        # 默认对话保存目录
├── test_features.py      # 功能测试脚本
└── FEATURES_README.md    # 本文档
```

## 使用示例

### 1. 基本对话流程
```javascript
// 发送消息
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        message: '你好，请介绍一下人工智能',
        user_id: 'user_123',
        provider: 'deepseek',
        apiKey: 'your-api-key'
    })
});

// 获取响应和图表URL
const data = await response.json();
console.log(data.response);
console.log(data.knowledge_graph_url);
```

### 2. 保存和加载对话
```javascript
// 保存对话
await fetch('/api/save_conversation', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        user_id: 'user_123',
        conversation_name: 'AI讨论',
        save_path: 'my_conversations'
    })
});

// 加载对话列表
const conversations = await fetch('/api/load_conversations?path=my_conversations');
const data = await conversations.json();
```

### 3. 布局控制
```javascript
// 切换到仅对话模式
setLayout('chat-only');

// 切换到仅可视化模式
setLayout('visualization-only');

// 切换到完整视图
setLayout('both');
```

## 技术实现细节

### 后端技术栈
- **Flask**: Web框架
- **NetworkX**: 图结构处理
- **Matplotlib**: 图表生成
- **NumPy**: 数值计算
- **Collections**: 数据结构

### 前端技术栈
- **Bootstrap 5**: UI框架
- **Bootstrap Icons**: 图标库
- **Vanilla JavaScript**: 交互逻辑
- **CSS Grid/Flexbox**: 布局系统

### 数据存储
- **JSON文件**: 对话数据存储
- **PNG图片**: 可视化图表
- **LocalStorage**: 用户设置

## 测试说明

运行测试脚本验证功能：

```bash
# 启动应用
python app.py

# 在另一个终端运行测试
python test_features.py
```

测试脚本会验证：
- 聊天API功能
- 对话保存功能
- 对话加载功能
- 知识图谱生成功能
- 对话重命名功能

## 故障排除

### 常见问题

1. **对话保存失败**
   - 检查保存路径是否有写入权限
   - 确认用户会话是否存在

2. **知识图谱生成失败**
   - 检查matplotlib是否正确安装
   - 确认中文字体支持

3. **布局调整不生效**
   - 检查CSS样式是否正确加载
   - 确认JavaScript事件监听器是否绑定

### 调试技巧

1. 打开浏览器开发者工具查看控制台错误
2. 检查网络请求的状态码和响应
3. 查看Flask应用的日志输出
4. 使用测试脚本验证API功能

## 未来改进

### 计划中的功能
- [ ] 对话搜索功能
- [ ] 批量导出对话
- [ ] 图谱样式自定义
- [ ] 多用户支持
- [ ] 云端存储集成

### 性能优化
- [ ] 图谱生成缓存
- [ ] 异步图表渲染
- [ ] 大文件分页加载
- [ ] 内存使用优化

---

如有问题或建议，请联系开发团队。 