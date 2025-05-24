/**
 * MindPilot - 认知图谱平台
 * 主要JavaScript功能实现
 */

// 全局变量
let currentUserId = window.currentUserId || 'user_' + Math.random().toString(36).substring(2, 9);
let currentTab = 'knowledge-graph';
let chatHistory = [];
let modelSettings = {}; // 存储模型设置
let conversations = []; // 存储对话列表
let currentConversationId = null; // 当前选中的对话ID
let conversationsPath = 'conversations'; // 对话保存路径

// DOM元素引用
let chatContainer, userInput, sendButton, tabs, visualizationContainer;
let layoutBothBtn, layoutChatOnlyBtn, layoutVisualizationOnlyBtn;
let saveConversationBtn, loadConversationsBtn, conversationsList, generateGraphBtn;
let modelProvider, apiKey, toggleApiKey, modelName, baseUrl, testConnection;
let connectionStatus, saveSettings, contextWindow, contextWindowValue;
let temperature, temperatureValue, saveNewModel;

// 主初始化函数
window.initMindPilot = function() {
    console.log('Initializing MindPilot...');
    
    // 初始化DOM引用
    initDOMReferences();
    
    // 初始化各个功能模块
    initLayoutControls();
    initConversationManagement();
    initResizableContainers();
    initVisualizationTabs();
    initChatFunctionality();
    initSettingsEvents();
    
    // 加载保存的设置
    loadSettings();
    
    // 加载对话列表
    loadConversationsList();
    
    // 另存为相关元素
    const selectSaveLocation = document.getElementById('selectSaveLocation');
    
    // 初始化另存为功能
    if (selectSaveLocation) {
        initSaveAsSelection(selectSaveLocation);
    }
    
    console.log('MindPilot initialized successfully');
};

/**
 * 初始化DOM元素引用
 */
function initDOMReferences() {
    // 基础聊天元素
    chatContainer = document.getElementById('chatContainer');
    userInput = document.getElementById('userInput');
    sendButton = document.getElementById('sendButton');
    tabs = document.querySelectorAll('.tab');
    visualizationContainer = document.getElementById('visualizationContainer');

    // 布局控制元素
    layoutBothBtn = document.getElementById('layoutBoth');
    layoutChatOnlyBtn = document.getElementById('layoutChatOnly');
    layoutVisualizationOnlyBtn = document.getElementById('layoutVisualizationOnly');

    // 对话管理元素
    saveConversationBtn = document.getElementById('saveConversationBtn');
    loadConversationsBtn = document.getElementById('loadConversationsBtn');
    conversationsList = document.getElementById('conversationsList');
    generateGraphBtn = document.getElementById('generateGraphBtn');

    // 设置相关元素
    modelProvider = document.getElementById('modelProvider');
    apiKey = document.getElementById('apiKey');
    toggleApiKey = document.getElementById('toggleApiKey');
    modelName = document.getElementById('modelName');
    baseUrl = document.getElementById('baseUrl');
    testConnection = document.getElementById('testConnection');
    connectionStatus = document.getElementById('connectionStatus');
    saveSettings = document.getElementById('saveSettings');
    contextWindow = document.getElementById('contextWindow');
    contextWindowValue = document.getElementById('contextWindowValue');
    temperature = document.getElementById('temperature');
    temperatureValue = document.getElementById('temperatureValue');
    saveNewModel = document.getElementById('saveNewModel');
}

/**
 * 初始化布局控制功能
 */
function initLayoutControls() {
    console.log('Initializing layout controls...');
    
    if (layoutBothBtn) {
        layoutBothBtn.addEventListener('click', () => setLayout('both'));
    }
    if (layoutChatOnlyBtn) {
        layoutChatOnlyBtn.addEventListener('click', () => setLayout('chat-only'));
    }
    if (layoutVisualizationOnlyBtn) {
        layoutVisualizationOnlyBtn.addEventListener('click', () => setLayout('visualization-only'));
    }
}

/**
 * 设置布局模式
 */
function setLayout(mode) {
    console.log('Setting layout to:', mode);
    
    const chatContainer = document.getElementById('chatContainer');
    const visualizationContainer = document.getElementById('visualizationContainer');
    const resizeHandle = document.getElementById('resizeHandle');
    const chatInput = document.querySelector('.chat-input');
    
    // 移除所有按钮的激活状态
    document.querySelectorAll('.layout-btn').forEach(btn => btn.classList.remove('active'));
    
    switch(mode) {
        case 'both':
            if (chatContainer) {
                chatContainer.classList.remove('hidden');
                chatContainer.style.height = '40%'; // 设置默认高度
            }
            if (visualizationContainer) {
                visualizationContainer.classList.remove('hidden');
                visualizationContainer.style.height = '60%'; // 设置默认高度
            }
            if (resizeHandle) resizeHandle.classList.remove('hidden');
            if (chatInput) chatInput.classList.remove('hidden');
            if (layoutBothBtn) layoutBothBtn.classList.add('active');
            break;
        case 'chat-only':
            if (chatContainer) {
                chatContainer.classList.remove('hidden');
                chatContainer.style.height = 'calc(100vh - 200px)'; // 占满可用空间
            }
            if (visualizationContainer) visualizationContainer.classList.add('hidden');
            if (resizeHandle) resizeHandle.classList.add('hidden');
            if (chatInput) chatInput.classList.remove('hidden');
            if (layoutChatOnlyBtn) layoutChatOnlyBtn.classList.add('active');
            break;
        case 'visualization-only':
            if (chatContainer) chatContainer.classList.add('hidden');
            if (visualizationContainer) {
                visualizationContainer.classList.remove('hidden');
                visualizationContainer.style.height = 'calc(100vh - 150px)'; // 占满可用空间
            }
            if (resizeHandle) resizeHandle.classList.add('hidden');
            if (chatInput) chatInput.classList.add('hidden');
            if (layoutVisualizationOnlyBtn) layoutVisualizationOnlyBtn.classList.add('active');
            break;
    }
}

/**
 * 初始化对话管理功能
 */
function initConversationManagement() {
    console.log('Initializing conversation management...');
    
    // 保存对话按钮
    if (saveConversationBtn) {
        saveConversationBtn.addEventListener('click', showSaveConversationModal);
    }
    
    // 加载对话按钮
    if (loadConversationsBtn) {
        loadConversationsBtn.addEventListener('click', loadConversationsList);
    }
    
    // 重新生成图谱按钮
    if (generateGraphBtn) {
        generateGraphBtn.addEventListener('click', generateCurrentGraph);
    }
    
    // 保存对话确认按钮
    const confirmSaveBtn = document.getElementById('confirmSaveConversation');
    if (confirmSaveBtn) {
        confirmSaveBtn.addEventListener('click', saveCurrentConversation);
    }
    
    // 重命名对话确认按钮
    const confirmRenameBtn = document.getElementById('confirmRenameConversation');
    if (confirmRenameBtn) {
        confirmRenameBtn.addEventListener('click', renameConversation);
    }
}

/**
 * 显示保存对话模态框
 */
function showSaveConversationModal() {
    console.log('Showing save conversation modal...');
    
    // 检查是否有对话历史
    if (!chatHistory || chatHistory.length === 0) {
        alert('当前没有对话内容可以保存');
        return;
    }
    
    const conversationNameInput = document.getElementById('conversationName');
    const savePathInput = document.getElementById('savePath');
    
    if (conversationNameInput) {
        conversationNameInput.value = `对话_${new Date().toLocaleString('zh-CN').replace(/[\/\s:]/g, '_')}`;
    }
    if (savePathInput) {
        savePathInput.value = conversationsPath;
    }
    
    if (window.saveConversationModal) {
        window.saveConversationModal.show();
    }
}

/**
 * 保存当前对话
 */
async function saveCurrentConversation() {
    console.log('Saving current conversation...');
    
    const conversationName = document.getElementById('conversationName').value.trim();
    const savePath = document.getElementById('savePath').value.trim() || 'conversations';
    
    if (!conversationName) {
        alert('请输入对话名称');
        return;
    }
    
    try {
        const response = await fetch('/api/save_conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: currentUserId,
                conversation_name: conversationName,
                save_path: savePath
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            if (window.saveConversationModal) {
                window.saveConversationModal.hide();
            }
            showNotification('对话保存成功！', 'success');
            loadConversationsList(); // 刷新对话列表
        } else {
            alert('保存失败：' + result.error);
        }
    } catch (error) {
        console.error('保存对话时出错：', error);
        alert('保存对话时出错：' + error.message);
    }
}

/**
 * 加载对话列表
 */
async function loadConversationsList() {
    console.log('Loading conversations list from path:', conversationsPath);
    
    try {
        const response = await fetch(`/api/load_conversations?path=${encodeURIComponent(conversationsPath)}`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            console.error('加载对话列表失败：', result.error);
            showNotification('加载对话列表失败: ' + result.error, 'error');
            return;
        }
        
        conversations = result.conversations || [];
        
        console.log(`从路径 ${result.path} 加载了 ${conversations.length} 个对话`);
        
        if (result.error_files && result.error_files > 0) {
            console.warn(`警告: ${result.error_files} 个文件读取失败`);
            showNotification(`加载完成，但有 ${result.error_files} 个文件读取失败`, 'warning');
        }
        
        displayConversationsList();
        
        console.log(`Successfully loaded ${conversations.length} conversations`);
        
    } catch (error) {
        console.error('加载对话列表时出错：', error);
        showNotification('加载对话列表失败: ' + error.message, 'error');
        
        // 在错误情况下显示空列表
        conversations = [];
        displayConversationsList();
    }
}

/**
 * 显示对话列表
 */
function displayConversationsList() {
    if (!conversationsList) return;
    
    conversationsList.innerHTML = '';
    
    if (conversations.length === 0) {
        conversationsList.innerHTML = '<div class="text-muted text-center p-3">暂无保存的对话</div>';
        return;
    }
    
    conversations.forEach(conversation => {
        const conversationItem = document.createElement('div');
        conversationItem.className = 'conversation-item';
        
        // 转义对话名称，防止特殊字符影响onclick
        const escapedName = conversation.name.replace(/'/g, "\\'");
        const escapedId = conversation.id;
        
        conversationItem.innerHTML = `
            <div onclick="loadConversationById('${escapedId}')" style="cursor: pointer; flex: 1;">
                <div class="fw-bold">${conversation.name}</div>
                <small class="text-muted">${new Date(conversation.created_at).toLocaleString('zh-CN')} | ${conversation.message_count} 条消息</small>
            </div>
            <div class="conversation-actions">
                <button class="btn btn-sm" onclick="loadConversationById('${escapedId}')" title="加载对话">
                    <i class="bi bi-folder-open"></i>
                </button>
                <button class="btn btn-sm" onclick="showRenameModal('${escapedId}', '${escapedName}')" title="重命名">
                    <i class="bi bi-pencil"></i>
                </button>
                <button class="btn btn-sm" onclick="generateGraphFromConversation('${escapedId}')" title="生成知识图谱">
                    <i class="bi bi-diagram-3"></i>
                </button>
                <button class="btn btn-sm text-danger" onclick="deleteConversationById('${escapedId}')" title="删除">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        `;
        
        conversationsList.appendChild(conversationItem);
    });
    
    console.log(`显示了 ${conversations.length} 个对话项`);
}

/**
 * 加载特定对话 - 新版本
 */
async function loadConversationById(conversationId) {
    console.log('Loading conversation by ID:', conversationId, 'from path:', conversationsPath);
    
    try {
        const response = await fetch(`/api/load_conversation/${conversationId}?path=${encodeURIComponent(conversationsPath)}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const conversation = await response.json();
        
        if (conversation.error) {
            throw new Error(conversation.error);
        }
        
        console.log('Loaded conversation:', conversation.name, 'with', conversation.chat_history?.length || 0, 'messages');
        
        // 清空当前聊天容器
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
        }
        
        // 清空并重新设置聊天历史
        chatHistory = [];
        
        // 验证对话历史存在
        if (!conversation.chat_history || !Array.isArray(conversation.chat_history)) {
            console.warn('对话历史格式不正确或为空');
            showNotification('对话历史为空或格式不正确', 'warning');
            
            // 显示默认欢迎消息
            if (messagesContainer) {
                messagesContainer.innerHTML = `
                    <div class="message assistant">
                        欢迎使用MindPilot认知图谱平台。我可以帮助您分析问题、做出决策，并生成可视化的认知图谱。请告诉我您想探讨的话题或需要解决的问题。
                    </div>
                `;
            }
            return;
        }
        
        // 显示加载的对话历史
        let loadedMessages = 0;
        conversation.chat_history.forEach((message, index) => {
            try {
                if (message && message.content && message.role) {
                    addMessage(message.content, message.role === 'user' ? 'user' : 'assistant');
                    chatHistory.push({
                        role: message.role,
                        content: message.content
                    });
                    loadedMessages++;
                } else {
                    console.warn('消息格式不正确:', index, message);
                }
            } catch (error) {
                console.error('处理消息时出错:', index, error);
            }
        });
        
        console.log(`成功加载 ${loadedMessages} 条消息`);
        
        // 设置当前对话ID
        currentConversationId = conversationId;
        
        // 更新对话列表的选中状态
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // 查找并高亮当前对话项
        const conversationButtons = document.querySelectorAll(`[onclick*="${conversationId}"]`);
        conversationButtons.forEach(button => {
            const conversationItem = button.closest('.conversation-item');
            if (conversationItem) {
                conversationItem.classList.add('active');
            }
        });
        
        showNotification(`对话"${conversation.name}"加载成功！已加载${loadedMessages}条消息`, 'success');
        
        // 滚动到最新消息
        setTimeout(() => {
            const messagesContainer2 = document.getElementById('chatMessages');
            if (messagesContainer2) {
                messagesContainer2.scrollTop = messagesContainer2.scrollHeight;
            }
        }, 100);
        
    } catch (error) {
        console.error('加载对话时出错：', error);
        showNotification('加载对话失败: ' + error.message, 'error');
    }
}

/**
 * 显示重命名对话模态框
 */
function showRenameModal(conversationId, currentName) {
    console.log('Showing rename modal for:', conversationId);
    
    const newNameInput = document.getElementById('newConversationName');
    if (newNameInput) {
        newNameInput.value = currentName;
        newNameInput.dataset.conversationId = conversationId;
    }
    
    if (window.renameConversationModal) {
        window.renameConversationModal.show();
    }
}

/**
 * 重命名对话
 */
async function renameConversation() {
    console.log('Renaming conversation...');
    
    const newNameInput = document.getElementById('newConversationName');
    const conversationId = newNameInput.dataset.conversationId;
    const newName = newNameInput.value.trim();
    
    if (!newName) {
        alert('请输入新的对话名称');
        return;
    }
    
    try {
        const response = await fetch('/api/rename_conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: conversationId,
                new_name: newName,
                path: conversationsPath
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            if (window.renameConversationModal) {
                window.renameConversationModal.hide();
            }
            showNotification('重命名成功！', 'success');
            loadConversationsList(); // 刷新对话列表
        } else {
            alert('重命名失败：' + result.error);
        }
    } catch (error) {
        console.error('重命名对话时出错：', error);
        alert('重命名对话时出错：' + error.message);
    }
}

/**
 * 删除对话 - 新版本
 */
async function deleteConversationById(conversationId) {
    console.log('Deleting conversation by ID:', conversationId);
    
    if (!confirm('确定要删除这个对话吗？此操作不可撤销。')) {
        return;
    }
    
    try {
        const response = await fetch('/api/delete_conversation', {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: conversationId,
                path: conversationsPath
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('删除成功！', 'success');
            loadConversationsList(); // 刷新对话列表
        } else {
            alert('删除失败：' + result.error);
        }
    } catch (error) {
        console.error('删除对话时出错：', error);
        alert('删除对话时出错：' + error.message);
    }
}

/**
 * 从对话生成知识图谱
 */
async function generateGraphFromConversation(conversationId) {
    console.log('Generating comprehensive graph including conversation:', conversationId);
    
    try {
        // 修改为生成基于所有对话的综合知识图谱，而不是特定对话
        const response = await fetch('/api/generate_knowledge_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: currentUserId
                // 不传conversation_id，让后端自动聚合所有对话内容
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // 切换到知识图谱标签
            setActiveTab('knowledge-graph');
            
            // 显示生成的图谱
            displayVisualization(result.graph_url);
            
            showNotification('综合知识图谱生成成功！', 'success');
        } else {
            alert('生成知识图谱失败：' + result.error);
        }
    } catch (error) {
        console.error('生成知识图谱时出错：', error);
        alert('生成知识图谱时出错：' + error.message);
    }
}

/**
 * 生成当前对话的知识图谱
 */
async function generateCurrentGraph() {
    console.log('Generating comprehensive knowledge graph...');
    
    try {
        // 生成基于所有对话内容的综合知识图谱
        const response = await fetch('/api/generate_knowledge_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: currentUserId
                // 不传特定对话ID，让后端聚合所有数据
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayVisualization(result.graph_url);
            showNotification('综合知识图谱重新生成成功！', 'success');
        } else {
            alert('生成知识图谱失败：' + result.error);
        }
    } catch (error) {
        console.error('生成知识图谱时出错：', error);
        alert('生成知识图谱时出错：' + error.message);
    }
}

/**
 * 初始化可调整大小的容器
 */
function initResizableContainers() {
    console.log('Initializing resizable containers...');
    
    const resizeHandle = document.getElementById('resizeHandle');
    
    if (!resizeHandle) {
        console.warn('Resize handle not found');
        return;
    }
    
    // 全局拖拽状态
    let isDragging = false;
    let startMouseY = 0;
    let startChatHeight = 0;
    let startVisualizationHeight = 0;
    let totalAvailableHeight = 0;
    
    // 初始化容器高度
    function initializeContainerHeights() {
        const chatContainer = document.getElementById('chatContainer');
        const visualizationContainer = document.getElementById('visualizationContainer');
        const content = document.querySelector('.content');
        
        if (!chatContainer || !visualizationContainer || !content) {
            console.warn('Required containers not found');
            return false;
        }
        
        // 计算可用总高度
        const contentHeight = content.clientHeight;
        const layoutControlsHeight = document.querySelector('.layout-controls')?.offsetHeight || 0;
        const chatInputHeight = document.querySelector('.chat-input')?.offsetHeight || 0;
        const resizeHandleHeight = resizeHandle.offsetHeight || 12;
        
        totalAvailableHeight = contentHeight - layoutControlsHeight - chatInputHeight - resizeHandleHeight - 40; // 40px for padding
        
        // 设置初始高度（40% 对话，60% 可视化）
        const initialChatHeight = Math.floor(totalAvailableHeight * 0.4);
        const initialVisualizationHeight = totalAvailableHeight - initialChatHeight;
        
        chatContainer.style.height = initialChatHeight + 'px';
        visualizationContainer.style.height = initialVisualizationHeight + 'px';
        
        console.log('Initialized heights:', {
            total: totalAvailableHeight,
            chat: initialChatHeight,
            visualization: initialVisualizationHeight
        });
        
        return true;
    }
    
    // 初始化高度
    setTimeout(initializeContainerHeights, 100);
    
    // 开始拖拽
    function startDrag(e) {
        const chatContainer = document.getElementById('chatContainer');
        const visualizationContainer = document.getElementById('visualizationContainer');
        
        // 检查容器是否可见
        if (!chatContainer || !visualizationContainer || 
            chatContainer.classList.contains('hidden') || 
            visualizationContainer.classList.contains('hidden')) {
            console.warn('Cannot resize: containers not visible');
            return;
        }
        
        isDragging = true;
        startMouseY = e.clientY;
        startChatHeight = chatContainer.offsetHeight;
        startVisualizationHeight = visualizationContainer.offsetHeight;
        
        console.log('Start dragging:', {
            mouseY: startMouseY,
            chatHeight: startChatHeight,
            visualizationHeight: startVisualizationHeight
        });
        
        // 添加视觉反馈
        resizeHandle.style.backgroundColor = 'var(--primary-color)';
        resizeHandle.style.transform = 'scaleY(1.2)';
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
        
        // 添加全局事件监听器
        document.addEventListener('mousemove', handleDrag, { passive: false });
        document.addEventListener('mouseup', stopDrag);
        document.addEventListener('mouseleave', stopDrag); // 防止鼠标离开页面时卡住
        
        e.preventDefault();
        e.stopPropagation();
    }
    
    // 处理拖拽
    function handleDrag(e) {
        if (!isDragging) return;
        
        const chatContainer = document.getElementById('chatContainer');
        const visualizationContainer = document.getElementById('visualizationContainer');
        
        if (!chatContainer || !visualizationContainer) return;
        
        const deltaY = e.clientY - startMouseY;
        
        // 计算新的高度
        let newChatHeight = startChatHeight + deltaY;
        let newVisualizationHeight = startVisualizationHeight - deltaY;
        
        // 应用最小和最大高度限制
        const minHeight = 150;
        const maxChatHeight = totalAvailableHeight - minHeight;
        const maxVisualizationHeight = totalAvailableHeight - minHeight;
        
        newChatHeight = Math.max(minHeight, Math.min(maxChatHeight, newChatHeight));
        newVisualizationHeight = totalAvailableHeight - newChatHeight;
        
        // 应用新高度
        chatContainer.style.height = newChatHeight + 'px';
        visualizationContainer.style.height = newVisualizationHeight + 'px';
        
        // 更新flex属性以固定高度
        chatContainer.style.flex = 'none';
        visualizationContainer.style.flex = 'none';
        
        e.preventDefault();
    }
    
    // 停止拖拽
    function stopDrag(e) {
        if (!isDragging) return;
        
        isDragging = false;
        
        console.log('Stop dragging');
        
        // 移除视觉反馈
        resizeHandle.style.backgroundColor = '';
        resizeHandle.style.transform = '';
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        
        // 移除事件监听器
        document.removeEventListener('mousemove', handleDrag);
        document.removeEventListener('mouseup', stopDrag);
        document.removeEventListener('mouseleave', stopDrag);
        
        // 保存当前高度到localStorage
        const chatContainer = document.getElementById('chatContainer');
        const visualizationContainer = document.getElementById('visualizationContainer');
        
        if (chatContainer && visualizationContainer) {
            const heights = {
                chat: chatContainer.offsetHeight,
                visualization: visualizationContainer.offsetHeight
            };
            localStorage.setItem('mindpilot_container_heights', JSON.stringify(heights));
            console.log('Saved heights:', heights);
        }
    }
    
    // 绑定鼠标事件
    resizeHandle.addEventListener('mousedown', startDrag);
    
    // 双击重置
    resizeHandle.addEventListener('dblclick', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        const chatContainer = document.getElementById('chatContainer');
        const visualizationContainer = document.getElementById('visualizationContainer');
        
        if (chatContainer && visualizationContainer && totalAvailableHeight > 0) {
            const defaultChatHeight = Math.floor(totalAvailableHeight * 0.4);
            const defaultVisualizationHeight = totalAvailableHeight - defaultChatHeight;
            
            chatContainer.style.height = defaultChatHeight + 'px';
            visualizationContainer.style.height = defaultVisualizationHeight + 'px';
            chatContainer.style.flex = 'none';
            visualizationContainer.style.flex = 'none';
            
            // 移除保存的高度
            localStorage.removeItem('mindpilot_container_heights');
            
            console.log('Reset to default heights');
        }
    });
    
    // 窗口大小改变时重新计算
    window.addEventListener('resize', () => {
        setTimeout(() => {
            initializeContainerHeights();
            
            // 恢复保存的高度比例
            const savedHeights = localStorage.getItem('mindpilot_container_heights');
            if (savedHeights) {
                try {
                    const heights = JSON.parse(savedHeights);
                    const chatContainer = document.getElementById('chatContainer');
                    const visualizationContainer = document.getElementById('visualizationContainer');
                    
                    if (chatContainer && visualizationContainer) {
                        const totalSaved = heights.chat + heights.visualization;
                        const chatRatio = heights.chat / totalSaved;
                        
                        const newChatHeight = Math.floor(totalAvailableHeight * chatRatio);
                        const newVisualizationHeight = totalAvailableHeight - newChatHeight;
                        
                        chatContainer.style.height = newChatHeight + 'px';
                        visualizationContainer.style.height = newVisualizationHeight + 'px';
                    }
                } catch (e) {
                    console.warn('Failed to restore saved heights:', e);
                }
            }
        }, 100);
    });
    
    // 恢复保存的高度
    const savedHeights = localStorage.getItem('mindpilot_container_heights');
    if (savedHeights) {
        setTimeout(() => {
            try {
                const heights = JSON.parse(savedHeights);
                const chatContainer = document.getElementById('chatContainer');
                const visualizationContainer = document.getElementById('visualizationContainer');
                
                if (chatContainer && visualizationContainer && totalAvailableHeight > 0) {
                    chatContainer.style.height = heights.chat + 'px';
                    visualizationContainer.style.height = heights.visualization + 'px';
                    chatContainer.style.flex = 'none';
                    visualizationContainer.style.flex = 'none';
                    
                    console.log('Restored saved heights:', heights);
                }
            } catch (e) {
                console.warn('Failed to restore saved heights:', e);
            }
        }, 200);
    }
}

/**
 * 初始化可视化标签切换
 */
function initVisualizationTabs() {
    console.log('Initializing visualization tabs...');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            currentTab = this.getAttribute('data-tab');
            updateVisualization();
        });
    });
}

/**
 * 设置活动标签
 */
function setActiveTab(tabName) {
    tabs.forEach(tab => {
        if (tab.getAttribute('data-tab') === tabName) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
    currentTab = tabName;
}

/**
 * 显示通知
 */
function showNotification(message, type = 'info') {
    console.log(`Notification [${type}]:`, message);
    
    // 创建通知元素
    const notification = document.createElement('div');
    let alertClass = 'alert-info';
    
    switch(type) {
        case 'success':
            alertClass = 'alert-success';
            break;
        case 'error':
            alertClass = 'alert-danger';
            break;
        case 'warning':
            alertClass = 'alert-warning';
            break;
        default:
            alertClass = 'alert-info';
    }
    
    notification.className = `alert ${alertClass} alert-dismissible fade show`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        max-width: 500px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;
    notification.innerHTML = `
        <strong>${type === 'error' ? '错误' : type === 'warning' ? '警告' : type === 'success' ? '成功' : '信息'}:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    document.body.appendChild(notification);
    
    // 自动删除时间根据类型调整
    const autoRemoveTime = type === 'error' ? 8000 : type === 'warning' ? 6000 : 3000;
    
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, autoRemoveTime);
}

/**
 * 初始化聊天功能
 */
function initChatFunctionality() {
    console.log('Initializing chat functionality...');
    
    // 发送消息事件
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    
    if (userInput) {
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    // 侧边栏项目点击事件
    document.querySelectorAll('.sidebar-item').forEach(item => {
        item.addEventListener('click', function() {
            document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

/**
 * 发送消息到后端并处理响应
 */
async function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') return;
    
    console.log('Sending message:', message);
    
    // 添加用户消息到聊天界面和历史记录
    addMessage(message, 'user');
    chatHistory.push({ role: 'user', content: message });
    
    userInput.value = '';
    
    // 显示加载状态
    const loadingMessage = addMessage('正在思考...', 'assistant');
    
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
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                user_id: currentUserId,
                model: settings.modelName,
                temperature: parseFloat(settings.temperature),
                stream: settings.streamingResponse,
                provider: settings.provider,
                apiKey: settings.apiKey,
                base_url: settings.baseUrl
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || '网络响应不正常');
        }
        
        const data = await response.json();
        
        // 更新聊天界面
        loadingMessage.textContent = data.response;
        loadingMessage.classList.remove('loading');
        
        // 添加助手回复到历史记录
        chatHistory.push({ role: 'assistant', content: data.response });
        
        // 如果有警告信息，显示给用户
        if (data.warning) {
            showNotification(data.warning, 'warning');
        }
        
        // 更新可视化
        updateVisualizationWithData(data);
        
    } catch (error) {
        console.error('Error:', error);
        loadingMessage.textContent = '抱歉，处理您的请求时出现了错误: ' + error.message;
        loadingMessage.classList.add('error');
    }
}

/**
 * 添加消息到聊天界面
 * @param {string} text - 消息文本
 * @param {string} sender - 发送者类型 ('user' 或 'assistant')
 * @returns {HTMLElement} - 创建的消息元素
 */
function addMessage(text, sender) {
    // 使用chatMessages容器而不是chatContainer
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) {
        console.error("Chat messages container not found.");
        return null;
    }
    
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.classList.add(sender === 'user' ? 'user' : 'assistant');
    
    if (sender === 'assistant' && text === '正在思考...') {
        messageElement.classList.add('loading');
    }
    
    messageElement.textContent = text;
    
    messagesContainer.appendChild(messageElement);
    
    // 自动滚动到底部
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageElement;
}

/**
 * 显示可视化内容
 */
function displayVisualization(imageUrl) {
    const visualizationContent = document.getElementById('visualizationContent');
    if (!visualizationContent) {
        console.error("Visualization content area not found.");
        return;
    }
    
    visualizationContent.innerHTML = ''; // 清空旧内容
    const img = document.createElement('img');
    img.src = imageUrl + '?t=' + new Date().getTime(); // 添加时间戳防止缓存
    img.alt = "可视化图表";
    img.style.maxWidth = '100%';
    img.style.maxHeight = '100%';
    img.style.objectFit = 'contain';
    img.onload = () => console.log('Visualization image loaded:', imageUrl);
    img.onerror = () => console.error('Failed to load visualization image:', imageUrl);
    visualizationContent.appendChild(img);
}

/**
 * 更新可视化内容
 */
function updateVisualization() {
    const visualizationContent = document.getElementById('visualizationContent');
    if (!visualizationContent) return;
    
    // 如果没有数据，显示占位符
    if (chatHistory.length === 0) {
        visualizationContent.innerHTML = `
            <div class="text-center text-muted">
                <i class="bi bi-diagram-3 fs-1 mb-2"></i>
                <p>开始对话后，这里将显示相关的可视化图表</p>
            </div>
        `;
        return;
    }
    
    // 根据当前标签显示对应的可视化内容
    switch (currentTab) {
        case 'knowledge-graph':
            displayVisualization(`/static/knowledge_graph_${currentUserId}.png`);
            break;
        case 'thought-chain':
            displayVisualization(`/static/thought_chain_${currentUserId}.png`);
            break;
        case 'cognitive-map':
            displayVisualization(`/static/cognitive_map_${currentUserId}.png`);
            break;
        case 'decision-model':
            displayVisualization(`/static/decision_model_${currentUserId}.png`);
            break;
        default:
            visualizationContent.innerHTML = `
                <div class="text-center text-muted">
                    <p>选择一个可视化标签查看内容</p>
                </div>
            `;
    }
}

/**
 * 使用返回的数据更新可视化
 */
function updateVisualizationWithData(data) {
    if (data.image_url || data.knowledge_graph_url) {
        const imageUrl = data.image_url || data.knowledge_graph_url;
        displayVisualization(imageUrl);
    }
}

/**
 * 初始化设置相关事件
 */
function initSettingsEvents() {
    console.log('Initializing settings events...');
    
    if (!toggleApiKey || !modelProvider || !testConnection) {
        console.warn('Some settings elements not found');
        return;
    }
    
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
                baseUrl.value = 'https://api.deepseek.com/v1';
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
        if (connectionStatus) {
            connectionStatus.innerHTML = '<span class="text-warning"><i class="bi bi-hourglass-split"></i> 测试中...</span>';
        }
        
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
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            
            if (response.ok) {
                if (connectionStatus) {
                    connectionStatus.innerHTML = '<span class="text-success"><i class="bi bi-check-circle"></i> 连接成功</span>';
                }
            } else {
                const errorData = await response.json();
                if (connectionStatus) {
                    connectionStatus.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle"></i> 连接失败: ${errorData.error || '未知错误'}</span>`;
                }
            }
        } catch (error) {
            console.error('Connection test error:', error);
            if (connectionStatus) {
                connectionStatus.innerHTML = '<span class="text-danger"><i class="bi bi-exclamation-triangle"></i> 连接失败: 网络错误</span>';
            }
        }
    });
    
    // 滑块值更新
    if (contextWindow && contextWindowValue) {
        contextWindow.addEventListener('input', () => {
            if (contextWindow.value >= 95) {
                contextWindowValue.textContent = '不限制';
            } else {
                contextWindowValue.textContent = contextWindow.value;
            }
        });
    }
    
    if (temperature && temperatureValue) {
        temperature.addEventListener('input', () => {
            temperatureValue.textContent = temperature.value;
        });
    }
    
    // 保存设置
    if (saveSettings) {
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
            showNotification('设置保存成功！', 'success');
        });
    }
}

/**
 * 加载保存的设置
 */
function loadSettings() {
    console.log('Loading settings...');
    
    const savedSettings = localStorage.getItem('mindpilotSettings');
    if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        
        if (modelProvider) modelProvider.value = settings.provider || 'deepseek';
        if (apiKey) apiKey.value = settings.apiKey || 'sk-5c35391ff9f04c73a3ccafff36fed371';
        if (modelName) modelName.value = settings.modelName || 'DeepSeek-R1';
        if (baseUrl) baseUrl.value = settings.baseUrl || 'https://api.deepseek.com/v1';
        if (temperature) temperature.value = settings.temperature || 0.7;
        if (temperatureValue) temperatureValue.textContent = settings.temperature || 0.7;
        if (contextWindow) contextWindow.value = settings.contextWindow || 50;
        
        if (contextWindowValue) {
            if (settings.contextWindow >= 95) {
                contextWindowValue.textContent = '不限制';
            } else {
                contextWindowValue.textContent = settings.contextWindow || 50;
            }
        }
        
        const streamingCheckbox = document.getElementById('streamingResponse');
        if (streamingCheckbox && settings.streamingResponse !== undefined) {
            streamingCheckbox.checked = settings.streamingResponse;
        }
        
        const saveHistoryCheckbox = document.getElementById('saveHistory');
        if (saveHistoryCheckbox && settings.saveHistory !== undefined) {
            saveHistoryCheckbox.checked = settings.saveHistory;
        }
        
        // 更新全局设置
        modelSettings = settings;
    } else {
        // 默认设置
        modelSettings = {
            provider: 'deepseek',
            apiKey: 'sk-5c35391ff9f04c73a3ccafff36fed371',
            modelName: 'deepseek-chat',
            baseUrl: 'https://api.deepseek.com/v1',
            temperature: 0.7,
            contextWindow: 50,
            streamingResponse: true,
            saveHistory: true
        };
    }
}

/**
 * 获取当前模型设置
 */
function getModelSettings() {
    return {
        provider: modelProvider?.value || 'deepseek',
        apiKey: apiKey?.value || 'sk-5c35391ff9f04c73a3ccafff36fed371',
        modelName: modelName?.value || 'deepseek-chat',
        baseUrl: baseUrl?.value || 'https://api.deepseek.com/v1',
        temperature: temperature?.value || 0.7,
        contextWindow: contextWindow?.value || 50,
        streamingResponse: document.getElementById('streamingResponse')?.checked || true,
        saveHistory: document.getElementById('saveHistory')?.checked || true
    };
}

/**
 * 初始化另存为选择器
 */
function initSaveAsSelection(selectButton) {
    console.log('Initializing save as selection...');
    
    selectButton.addEventListener('click', async () => {
        console.log('另存为按钮被点击');
        
        // 检查是否有对话历史
        if (!chatHistory || chatHistory.length === 0) {
            showNotification('当前没有对话内容可以保存', 'warning');
            return;
        }
        
        try {
            // 直接使用File System Access API的另存为功能
            if ('showSaveFilePicker' in window) {
                console.log('使用系统另存为对话框...');
                
                // 获取当前对话名称作为建议文件名
                const conversationNameInput = document.getElementById('conversationName');
                let suggestedName = '对话';
                if (conversationNameInput && conversationNameInput.value.trim()) {
                    suggestedName = conversationNameInput.value.trim();
                } else {
                    suggestedName = `对话_${new Date().toLocaleString('zh-CN').replace(/[\/\s:]/g, '_')}`;
                }
                
                // 打开系统另存为对话框
                const fileHandle = await window.showSaveFilePicker({
                    suggestedName: `${suggestedName}.json`,
                    types: [{
                        description: 'JSON对话文件',
                        accept: { 'application/json': ['.json'] }
                    }],
                    startIn: 'documents'
                });
                
                console.log('用户选择了文件:', fileHandle.name);
                
                // 准备保存的数据
                const conversationData = {
                    id: 'file_' + Math.random().toString(36).substring(2, 9),
                    name: fileHandle.name.replace(/\.json$/, ''),
                    created_at: new Date().toISOString(),
                    user_id: currentUserId,
                    chat_history: chatHistory,
                    user_profile: {},
                    knowledge_entities: []
                };
                
                // 创建可写流
                const writable = await fileHandle.createWritable();
                
                // 写入数据
                await writable.write(JSON.stringify(conversationData, null, 2));
                
                // 关闭文件
                await writable.close();
                
                showNotification(`对话已成功保存到: ${fileHandle.name}`, 'success');
                
                // 关闭模态框
                if (window.saveConversationModal) {
                    window.saveConversationModal.hide();
                }
                
            } else {
                // 浏览器不支持File System Access API
                console.log('浏览器不支持File System Access API，使用降级方案');
                showNotification('您的浏览器不支持系统另存为对话框，请使用"保存"按钮', 'warning');
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('用户取消了文件选择');
            } else {
                console.error('另存为失败:', error);
                showNotification('另存为失败: ' + error.message, 'error');
            }
        }
    });
}

// 将函数暴露到全局作用域，供HTML onclick使用
window.loadConversation = loadConversation;
window.loadConversationById = loadConversationById;
window.showRenameModal = showRenameModal;
window.generateGraphFromConversation = generateGraphFromConversation;
window.deleteConversation = deleteConversation;
window.deleteConversationById = deleteConversationById;