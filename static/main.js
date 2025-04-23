/**
 * MindPilot - 认知图谱平台
 * 主要JavaScript功能实现
 */

// 全局变量
let currentUserId = 'user_' + Math.random().toString(36).substring(2, 9);
let currentTab = 'knowledge-graph';
let chatHistory = [];
let modelSettings = {}; // 存储模型设置

// DOM元素
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const tabs = document.querySelectorAll('.tab');
const visualizationPlaceholder = document.getElementById('visualization-placeholder');

// 设置相关元素
const modelProvider = document.getElementById('modelProvider');
const apiKey = document.getElementById('apiKey');
const toggleApiKey = document.getElementById('toggleApiKey');
const modelName = document.getElementById('modelName');
const baseUrl = document.getElementById('baseUrl');
const testConnection = document.getElementById('testConnection');
const connectionStatus = document.getElementById('connectionStatus');
const saveSettings = document.getElementById('saveSettings');
const contextWindow = document.getElementById('contextWindow');
const contextWindowValue = document.getElementById('contextWindowValue');
const temperature = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperatureValue');
const saveNewModel = document.getElementById('saveNewModel');

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    // 切换可视化标签
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            currentTab = this.getAttribute('data-tab');
            updateVisualization();
        });
    });
    
    // 发送消息事件
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 侧边栏项目点击事件
    document.querySelectorAll('.sidebar-item').forEach(item => {
        item.addEventListener('click', function() {
            document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // 加载保存的设置
    loadSettings();
    
    // 设置相关事件监听器
    initSettingsEvents();
});

/**
 * 发送消息到后端并处理响应
 */
async function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') return;
    
    // 添加用户消息到聊天界面
    addMessage(message, 'user');
    userInput.value = '';
    
    // 显示加载状态
    const loadingMessage = addMessage('正在思考...', 'bot');
    
    try {
        // 获取当前设置
        const settings = getModelSettings();
        
        // 检查API密钥
        if (!settings.apiKey) {
            loadingMessage.textContent = '请在设置中配置API密钥';
            loadingMessage.classList.add('error');
            return;
        }
        
        // 发送请求到后端API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${settings.apiKey}` // 将 API 密钥添加到 Authorization header
            },
            body: JSON.stringify({
                message: message,
                user_id: currentUserId,
                model: settings.modelName,
                temperature: parseFloat(settings.temperature),
                stream: settings.streamingResponse,
                provider: settings.provider
                // api_key 已移至 header
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || '网络响应不正常');
        }
        
        const data = await response.json();
        
        // 更新聊天界面
        loadingMessage.textContent = data.response;
        
        // 如果有警告信息，显示给用户
        if (data.warning) {
            const warningElement = document.createElement('div');
            warningElement.classList.add('message', 'warning');
            warningElement.textContent = data.warning;
            chatContainer.appendChild(warningElement);
        }
        
        // 更新可视化
        updateVisualizationWithData(data);
        
        // 保存到聊天历史
        if (settings.saveHistory) {
            chatHistory.push({ role: 'user', content: message });
            chatHistory.push({ role: 'assistant', content: data.response });
        }
        
    } catch (error) {
        console.error('Error:', error);
        loadingMessage.textContent = '抱歉，处理您的请求时出现了错误: ' + error.message;
        loadingMessage.classList.add('error');
        
        // 记录详细错误信息到控制台，便于调试
        console.log('请求详情:', {
            'API密钥是否提供': !!settings.apiKey,
            '模型名称': settings.modelName,
            '提供商': settings.provider,
            '请求时间': new Date().toISOString()
        });
    }
}

/**
 * 添加消息到聊天界面
 * @param {string} text - 消息文本
 * @param {string} sender - 发送者类型 ('user' 或 'bot')
 */
function addMessage(text, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.classList.add(sender === 'user' ? 'user' : 'assistant');
    messageElement.textContent = text;
    
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageElement;
}

/**
 * 初始化设置相关事件
 */
function initSettingsEvents() {
    // 切换API密钥可见性
    toggleApiKey.addEventListener('click', () => {
        if (apiKey.type === 'password') {
            apiKey.type = 'text';
            toggleApiKey.innerHTML = '<i class="bi bi-eye-slash"></i>';
        } else {
            apiKey.type = 'password';
            toggleApiKey.innerHTML = '<i class="bi bi-eye"></i>';
        }
    });
    
    // 模型提供方变更事件
    modelProvider.addEventListener('change', () => {
        // 根据选择的提供方更新默认值
        switch(modelProvider.value) {
            case 'deepseek':
                baseUrl.value = 'https://api.deepseek.com';
                modelName.value = 'deepseek-chat';
                break;
            case 'openai':
                baseUrl.value = 'https://api.openai.com/v1';
                modelName.value = 'gpt-4';
                break;
            case 'gemini':
                baseUrl.value = 'https://generativelanguage.googleapis.com/v1';
                modelName.value = 'gemini-pro';
                break;
            case 'claude':
                baseUrl.value = 'https://api.anthropic.com/v1';
                modelName.value = 'claude-3-opus';
                break;
            case 'tongyi':
                baseUrl.value = 'https://dashscope.aliyuncs.com/api/v1';
                modelName.value = 'qwen-max';
                break;
            case 'douban':
                baseUrl.value = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1';
                modelName.value = 'ernie-bot-4';
                break;
        }
    });
    
    // 测试连接
    testConnection.addEventListener('click', async () => {
        connectionStatus.innerHTML = '<span class="text-warning"><i class="bi bi-hourglass-split"></i> 测试中...</span>';
        
        try {
            // 获取当前设置
            const settings = {
                provider: modelProvider.value,
                apiKey: apiKey.value,
                modelName: modelName.value,
                baseUrl: baseUrl.value
            };
            
            // 发送测试请求
            const response = await fetch('/api/test_connection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${settings.apiKey}`
                },
                body: JSON.stringify(settings)
            });
            
            if (response.ok) {
                connectionStatus.innerHTML = '<span class="text-success"><i class="bi bi-check-circle"></i> 连接成功</span>';
            } else {
                const errorData = await response.json();
                connectionStatus.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle"></i> 连接失败: ${errorData.error || '未知错误'}</span>`;
            }
        } catch (error) {
            console.error('Connection test error:', error);
            connectionStatus.innerHTML = '<span class="text-danger"><i class="bi bi-exclamation-triangle"></i> 连接失败: 网络错误</span>';
        }
    });
    
    // 保存新模型
    saveNewModel.addEventListener('click', () => {
        const newModelName = document.getElementById('newModelName').value;
        const newModelProvider = document.getElementById('newModelProvider').value;
        const newModelUrl = document.getElementById('newModelUrl').value;
        const newModelKey = document.getElementById('newModelKey').value;
        
        if (newModelName && newModelProvider && newModelUrl) {
            // 添加到模型提供方下拉列表
            const option = document.createElement('option');
            option.value = newModelProvider.toLowerCase().replace(/\s+/g, '_');
            option.textContent = newModelProvider;
            modelProvider.appendChild(option);
            modelProvider.value = option.value;
            
            // 更新其他字段
            modelName.value = newModelName;
            baseUrl.value = newModelUrl;
            apiKey.value = newModelKey;
            
            // 关闭折叠面板
            const collapse = bootstrap.Collapse.getInstance(document.getElementById('newModelForm'));
            collapse.hide();
            
            // 清空表单
            document.getElementById('newModelName').value = '';
            document.getElementById('newModelProvider').value = '';
            document.getElementById('newModelUrl').value = '';
            document.getElementById('newModelKey').value = '';
            
            // 保存自定义模型到本地存储
            saveCustomModel({
                name: newModelName,
                provider: newModelProvider,
                value: option.value,
                url: newModelUrl
            });
        }
    });
    
    // 滑块值更新
    contextWindow.addEventListener('input', () => {
        if (contextWindow.value >= 95) {
            contextWindowValue.textContent = '不限制';
        } else {
            contextWindowValue.textContent = contextWindow.value;
        }
    });
    
    temperature.addEventListener('input', () => {
        temperatureValue.textContent = temperature.value;
    });
    
    // 保存设置
    saveSettings.addEventListener('click', () => {
        // 保存设置到本地存储
        const settings = {
            provider: modelProvider.value,
            apiKey: apiKey.value,
            modelName: modelName.value,
            baseUrl: baseUrl.value,
            temperature: temperature.value,
            contextWindow: contextWindow.value,
            streamingResponse: document.getElementById('streamingResponse').checked,
            saveHistory: document.getElementById('saveHistory').checked
        };
        
        localStorage.setItem('mindpilotSettings', JSON.stringify(settings));
        modelSettings = settings; // 更新全局设置
        
        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
        modal.hide();
        
        // 显示保存成功提示
        const toast = new bootstrap.Toast(document.getElementById('settingsToast'));
        toast.show();
    });
}

/**
 * 保存自定义模型到本地存储
 */
function saveCustomModel(model) {
    let customModels = JSON.parse(localStorage.getItem('customModels') || '[]');
    customModels.push(model);
    localStorage.setItem('customModels', JSON.stringify(customModels));
}

/**
 * 加载自定义模型
 */
function loadCustomModels() {
    const customModels = JSON.parse(localStorage.getItem('customModels') || '[]');
    
    customModels.forEach(model => {
        // 检查是否已存在
        if (!document.querySelector(`option[value="${model.value}"]`)) {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.provider;
            modelProvider.appendChild(option);
        }
    });
}

/**
 * 加载保存的设置
 */
function loadSettings() {
    const savedSettings = localStorage.getItem('mindpilotSettings');
    if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        
        modelProvider.value = settings.provider || 'deepseek';
        apiKey.value = settings.apiKey || 'sk-5c35391ff9f04c73a3ccafff36fed371';
        modelName.value = settings.modelName || 'DeepSeek-R1';
        baseUrl.value = settings.baseUrl || 'https://api.deepseek.com/v1';
        temperature.value = settings.temperature || 0.7;
        temperatureValue.textContent = settings.temperature || 0.7;
        contextWindow.value = settings.contextWindow || 50;
        
        if (settings.contextWindow >= 95) {
            contextWindowValue.textContent = '不限制';
        } else {
            contextWindowValue.textContent = settings.contextWindow || 50;
        }
        
        if (settings.streamingResponse !== undefined) {
            document.getElementById('streamingResponse').checked = settings.streamingResponse;
        }
        
        if (settings.saveHistory !== undefined) {
            document.getElementById('saveHistory').checked = settings.saveHistory;
        }
        
        // 更新全局设置
        modelSettings = settings;
    } else {
        // 默认设置
        modelSettings = {
            provider: 'deepseek',
            apiKey: 'sk-5c35391ff9f04c73a3ccafff36fed371',
            modelName: 'deepseek-chat',
            baseUrl: 'https://api.deepseek.com',
            temperature: 0.7,
            contextWindow: 50,
            streamingResponse: true,
            saveHistory: true
        };
    }
    
    // 加载自定义模型
    loadCustomModels();
}

/**
 * 获取当前模型设置
 */
function getModelSettings() {
    return modelSettings;
}

/**
 * 添加消息到聊天界面
 * @param {string} text - 消息文本
 * @param {string} sender - 发送者类型 ('user' 或 'bot')
 * @returns {HTMLElement} - 创建的消息元素
 */
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageDiv;
}

/**
 * 根据当前选中的标签更新可视化内容
 */
function updateVisualization() {
    // 如果没有数据，显示占位符
    if (chatHistory.length === 0) {
        visualizationPlaceholder.innerHTML = `
            <div class="text-center">
                <i class="bi bi-diagram-3 fs-1 mb-2"></i>
                <p>开始对话以生成可视化内容</p>
            </div>
        `;
        return;
    }
    
    // 根据当前标签显示相应的可视化
    switch (currentTab) {
        case 'knowledge-graph':
            visualizationPlaceholder.innerHTML = `<img src="/static/knowledge_graph_${currentUserId}.png?t=${new Date().getTime()}" class="img-fluid" alt="知识图谱">`;
            break;
        case 'thought-chain':
            visualizationPlaceholder.innerHTML = `<img src="/static/thought_chain_${currentUserId}.png?t=${new Date().getTime()}" class="img-fluid" alt="思维链">`;
            break;
        case 'cognitive-map':
            visualizationPlaceholder.innerHTML = `<img src="/static/cognitive_map_${currentUserId}.png?t=${new Date().getTime()}" class="img-fluid" alt="认知图谱">`;
            break;
        case 'decision-model':
            visualizationPlaceholder.innerHTML = `<img src="/static/decision_model_${currentUserId}.png?t=${new Date().getTime()}" class="img-fluid" alt="决策模型">`;
            break;
    }
}

/**
 * 使用API返回的数据更新可视化
 * @param {Object} data - API返回的数据
 */
function updateVisualizationWithData(data) {
    // 更新当前选中的可视化
    updateVisualization();
}

/**
 * 创建新项目
 */
function createNewProject() {
    const projectName = prompt('请输入新项目名称:');
    if (projectName && projectName.trim() !== '') {
        // 重置当前会话
        currentUserId = 'user_' + Math.random().toString(36).substring(2, 9);
        chatHistory = [];
        
        // 清空聊天界面
        chatContainer.innerHTML = '';
        addMessage(`欢迎使用新项目「${projectName}」！我可以帮助你生成知识图谱、分析思维链并提供决策支持。`, 'bot');
        
        // 重置可视化
        visualizationPlaceholder.innerHTML = `
            <div class="text-center">
                <i class="bi bi-diagram-3 fs-1 mb-2"></i>
                <p>开始对话以生成可视化内容</p>
            </div>
        `;
        
        // 添加新项目到侧边栏
        const sidebarSection = document.querySelector('.sidebar-section:first-child');
        const newProjectItem = document.createElement('div');
        newProjectItem.classList.add('sidebar-item');
        newProjectItem.innerHTML = `<i class="bi bi-diagram-2"></i> ${projectName}`;
        
        // 插入到"新建项目