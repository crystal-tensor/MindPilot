# API 配置指南

## 🔑 API 密钥配置

为了使用MindPilot，您需要配置AI模型的API密钥。**请注意：您的API密钥只会保存在本地浏览器中，不会被上传到服务器或GitHub。**

## 🚀 快速配置

### 1. 获取API密钥

#### DeepSeek API (推荐)
1. 访问 [DeepSeek 平台](https://platform.deepseek.com/)
2. 注册账号并登录
3. 进入API密钥管理页面
4. 创建新的API密钥
5. 复制密钥（格式：`sk-xxxxxxxxxxxxxx`）

#### OpenAI API
1. 访问 [OpenAI 平台](https://platform.openai.com/)
2. 注册账号并登录
3. 进入API Keys页面
4. 创建新的API密钥

### 2. 在MindPilot中配置

1. 启动MindPilot应用
2. 点击右上角"设置"⚙️按钮
3. 在"模型"标签页中：
   - 选择模型提供方（DeepSeek/OpenAI等）
   - 输入您的API密钥
   - 配置模型名称和基础URL
4. 点击"测试连接"验证配置
5. 点击"保存"按钮

## 🔒 安全说明

- ✅ **本地存储**：API密钥仅保存在您的浏览器localStorage中
- ✅ **不会上传**：密钥不会被发送到GitHub或任何第三方服务器
- ✅ **数据隔离**：每个浏览器独立保存配置
- ⚠️ **注意事项**：清除浏览器数据会删除保存的配置

## 📊 支持的AI模型

| 提供商 | 模型示例 | 基础URL |
|--------|----------|----------|
| DeepSeek | deepseek-chat | https://api.deepseek.com/v1 |
| OpenAI | gpt-4, gpt-3.5-turbo | https://api.openai.com/v1 |
| Anthropic | claude-3-opus | https://api.anthropic.com/v1 |
| Google | gemini-pro | https://generativelanguage.googleapis.com/v1 |

## 🛠️ 故障排除

### 常见问题

1. **连接失败**
   - 检查API密钥格式是否正确
   - 确认网络连接正常
   - 验证API密钥是否有效

2. **模型不响应**
   - 检查模型名称是否正确
   - 确认API余额是否充足
   - 查看浏览器控制台错误信息

3. **设置丢失**
   - 确认没有清除浏览器数据
   - 尝试重新配置API密钥

### 获取帮助

如果遇到问题，请：
1. 查看浏览器开发者工具的控制台错误
2. 在GitHub仓库提交Issue
3. 参考API提供商的官方文档

---

**记住：永远不要在代码中硬编码API密钥！** 🔐 