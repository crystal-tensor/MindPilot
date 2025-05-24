from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import datetime
import uuid

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__, static_folder='static')

# 确保静态文件夹存在
if not os.path.exists('static'):
    os.makedirs('static')

# 用户会话存储
user_sessions = {}

# 模型提供方配置
model_providers = {
    'deepseek': {
        'base_url': 'https://api.deepseek.com/v1',  # 确保包含 /v1
        'default_model': 'deepseek-chat'
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'default_model': 'gpt-4'
    },
    'gemini': {
        'base_url': 'https://generativelanguage.googleapis.com/v1',
        'default_model': 'gemini-pro'
    },
    'claude': {
        'base_url': 'https://api.anthropic.com/v1',
        'default_model': 'claude-3-opus'
    },
    'tongyi': {
        'base_url': 'https://dashscope.aliyuncs.com/api/v1',
        'default_model': 'qwen-max'
    },
    'douban': {
        'base_url': 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1',
        'default_model': 'ernie-bot-4'
    }
}

# 导入OpenAI客户端
from openai import OpenAI

# 大模型接口


# 关键词提取
def extract_keywords(text):
    """从文本中提取关键词"""
    # 实际项目中应使用更复杂的NLP技术
    words = text.replace('。', ' ').replace('，', ' ').replace('？', ' ').replace('！', ' ').replace('：', ' ').replace('；', ' ').split()
    # 扩展停用词列表
    stopwords = ['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', 
                '吗', '啊', '呢', '吧', '呀', '哦', '么', '那', '这个', '那个', '什么', '为什么', '怎么', '如何', '可以', '可能', '应该', '需要']
    
    # 提取关键词并计算词频
    keyword_freq = {}
    for word in words:
        if len(word) > 1 and word not in stopwords:
            if word in keyword_freq:
                keyword_freq[word] += 1
            else:
                keyword_freq[word] = 1
    
    # 按词频排序并返回前10个关键词
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:10]]

# 生成知识图谱
def generate_knowledge_graph(user_id, conversation_data=None):
    """生成用户知识图谱"""
    # 收集所有对话内容
    all_chat_history = []
    all_knowledge_entities = set()
    all_user_profile = defaultdict(int)
    
    # 如果提供了特定对话数据，使用它
    if conversation_data:
        chat_history = conversation_data.get('chat_history', [])
        user_profile = conversation_data.get('user_profile', {})
        knowledge_entities = set(conversation_data.get('knowledge_entities', []))
        
        all_chat_history.extend(chat_history)
        all_knowledge_entities.update(knowledge_entities)
        for key, value in user_profile.items():
            all_user_profile[key] += value
    else:
        # 收集当前用户会话数据
        if user_id in user_sessions:
            session = user_sessions[user_id]
            all_chat_history.extend(session['chat_history'])
            all_knowledge_entities.update(session['knowledge_entities'])
            for key, value in session['user_profile'].items():
                all_user_profile[key] += value
    
    # 收集所有保存的对话数据以获得更全面的知识图谱
    try:
        conversations_path = 'conversations'
        if os.path.exists(conversations_path):
            for filename in os.listdir(conversations_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(conversations_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            saved_conversation = json.load(f)
                        
                        # 添加保存对话的内容到总体数据中
                        saved_chat_history = saved_conversation.get('chat_history', [])
                        saved_user_profile = saved_conversation.get('user_profile', {})
                        saved_knowledge_entities = saved_conversation.get('knowledge_entities', [])
                        
                        all_chat_history.extend(saved_chat_history)
                        all_knowledge_entities.update(saved_knowledge_entities)
                        for key, value in saved_user_profile.items():
                            all_user_profile[key] += value
                            
                    except Exception as e:
                        print(f"读取保存对话文件失败: {filename}, 错误: {e}")
                        continue
    except Exception as e:
        print(f"扫描保存对话文件夹失败: {e}")
    
    print(f"知识图谱生成：总共收集了 {len(all_chat_history)} 条对话消息")
    print(f"知识图谱生成：总共收集了 {len(all_knowledge_entities)} 个知识实体")
    print(f"知识图谱生成：用户画像包含 {len(all_user_profile)} 个特征")
    
    # 创建图结构
    G = nx.Graph()
    
    # 提取所有关键词
    all_keywords = []
    keyword_cooccurrence = defaultdict(int)
    
    # 从所有对话历史中提取关键词
    for message in all_chat_history:
        if message.get('role') in ['user', 'assistant']:
            keywords = extract_keywords(message.get('content', ''))
            all_keywords.extend(keywords)
            
            # 计算关键词共现
            for i in range(len(keywords)):
                for j in range(i+1, len(keywords)):
                    pair = tuple(sorted([keywords[i], keywords[j]]))
                    keyword_cooccurrence[pair] += 1
    
    # 统计关键词频率
    keyword_freq = Counter(all_keywords)
    
    # 添加节点（只添加频率较高的关键词）
    min_freq = max(2, len(all_chat_history) // 50)  # 动态调整最小频率
    top_keywords = min(30, len(keyword_freq))  # 限制为前30个关键词
    
    for keyword, freq in keyword_freq.most_common(top_keywords):
        if freq >= min_freq:
            # 节点大小根据频率和重要性调整
            node_size = 100 + freq * 20
            node_color = '#7C3AED' if freq >= min_freq * 3 else '#C4B5FD'
            
            G.add_node(keyword, 
                      size=node_size,
                      color=node_color,
                      frequency=freq)
    
    # 添加边（共现关系）
    min_cooccur = max(1, len(all_chat_history) // 100)  # 动态调整最小共现次数
    for (word1, word2), cooccur_count in keyword_cooccurrence.items():
        if G.has_node(word1) and G.has_node(word2) and cooccur_count >= min_cooccur:
            G.add_edge(word1, word2, weight=cooccur_count)
    
    # 如果图太稀疏，添加一些基于用户画像的连接
    if len(G.edges()) < 5 and all_user_profile:
        profile_keywords = list(all_user_profile.keys())[:10]
        for keyword in profile_keywords:
            if G.has_node(keyword):
                # 与其他高频词建立连接
                for other_keyword in list(G.nodes())[:5]:
                    if other_keyword != keyword and not G.has_edge(keyword, other_keyword):
                        G.add_edge(keyword, other_keyword, weight=1)
    
    # 绘制图谱
    plt.figure(figsize=(14, 10))
    if len(G.nodes()) > 0:
        # 使用spring layout但增加更多迭代以获得更好的布局
        pos = nx.spring_layout(G, k=0.8, iterations=150, seed=42)
        
        # 节点大小和颜色
        node_sizes = [G.nodes[node].get('size', 100) for node in G.nodes()]
        node_colors = [G.nodes[node].get('color', '#C4B5FD') for node in G.nodes()]
        
        # 边的粗细
        edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
        edge_widths = [min(w * 2, 8) for w in edge_weights]  # 增加边的可见性
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='black',
            linewidths=1.5
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=edge_widths,
            alpha=0.6
        )
        
        # 绘制标签
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_color='black',
            font_weight='bold',
            font_family='Arial Unicode MS'
        )
        
        # 添加统计信息到标题
        title = f'综合知识图谱 (基于 {len(all_chat_history)} 条对话，{len(G.nodes())} 个概念)'
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    else:
        plt.text(0.5, 0.5, '暂无足够数据生成知识图谱\n请先进行一些对话', 
                ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
    
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图片
    graph_filename = f'knowledge_graph_{user_id}.png'
    if conversation_data and 'id' in conversation_data:
        graph_filename = f'knowledge_graph_{conversation_data["id"][:8]}.png'
    
    plt.savefig(f'static/{graph_filename}', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return graph_filename

# 生成思维链
def generate_thought_chain(question, user_id):
    """生成思维链可视化"""
    # 模拟思维链步骤
    thought_steps = [
        "理解问题：分析用户提出的决策问题",
        "收集信息：考虑相关因素和约束条件",
        "生成选项：列出可能的解决方案",
        "评估选项：分析每个选项的优缺点",
        "权衡取舍：考虑不同因素的重要性",
        "形成结论：提出最佳建议"
    ]
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点和边
    for i, step in enumerate(thought_steps):
        G.add_node(i, label=step)
        if i > 0:
            G.add_edge(i-1, i)
    
    # 绘制思维链
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # 节点
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#5B21B6', alpha=0.8)
    
    # 边
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, arrowsize=20)
    
    # 标签
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='white')
    
    plt.title('思维链分析', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'static/thought_chain_{user_id}.png', dpi=150, bbox_inches='tight')
    plt.close()

# 生成认知图谱（结合用户画像和思维链）
def generate_cognitive_map(user_id):
    """生成认知图谱"""
    if user_id not in user_sessions:
        return
    
    # 创建图结构
    G = nx.Graph()
    
    # 添加用户画像节点
    G.add_node('用户画像', size=300, color='#7C3AED')
    
    # 添加思维模式节点
    G.add_node('思维模式', size=300, color='#5B21B6')
    
    # 添加决策偏好节点
    G.add_node('决策偏好', size=300, color='#8B5CF6')
    
    # 从用户画像中提取特征
    profile = user_sessions[user_id]['user_profile']
    top_interests = sorted(profile.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # 添加兴趣节点
    for interest, weight in top_interests:
        G.add_node(interest, size=100 + weight * 30, color='#C4B5FD')
        G.add_edge('用户画像', interest, weight=weight)
    
    # 添加思维特征
    thought_features = ['分析型', '直觉型', '系统性思考', '创造性思考']
    for feature in thought_features:
        G.add_node(feature, size=150, color='#A78BFA')
        G.add_edge('思维模式', feature)
    
    # 添加决策偏好
    decision_features = ['风险规避', '长期规划', '快速决策', '数据驱动']
    for feature in decision_features:
        G.add_node(feature, size=150, color='#DDD6FE')
        G.add_edge('决策偏好', feature)
    
    # 绘制认知图谱
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # 节点大小和颜色
    node_sizes = [G.nodes[node].get('size', 100) for node in G.nodes()]
    node_colors = [G.nodes[node].get('color', '#C4B5FD') for node in G.nodes()]
    
    # 绘制
    nx.draw_networkx(
        G, pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color='gray',
        font_size=10,
        font_color='black',
        alpha=0.8
    )
    
    plt.title('用户认知图谱', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'static/cognitive_map_{user_id}.png', dpi=150, bbox_inches='tight')
    plt.close()

# 生成决策模型
def generate_decision_model(user_id):
    """生成决策模型可视化"""
    # 模拟决策因素和权重
    factors = ['经济效益', '时间成本', '风险程度', '社会影响', '长期价值']
    weights = [0.3, 0.15, 0.2, 0.15, 0.2]
    
    # 模拟选项评分
    options = ['方案A', '方案B', '方案C']
    scores = [
        [0.9, 0.3, 0.7, 0.5, 0.8],  # 方案A各因素评分
        [0.6, 0.8, 0.5, 0.7, 0.6],  # 方案B各因素评分
        [0.7, 0.6, 0.8, 0.9, 0.5]   # 方案C各因素评分
    ]
    
    # 计算加权得分
    weighted_scores = []
    for option_scores in scores:
        weighted_score = sum(s * w for s, w in zip(option_scores, weights))
        weighted_scores.append(weighted_score)
    
    # 绘制决策模型
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    ax2 = fig.add_subplot(1, 2, 2)

    # 雷达图 - 各方案在各因素上的表现
    angles = np.linspace(0, 2*np.pi, len(factors), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(0)
    
    for i, option in enumerate(options):
        option_scores = scores[i] + [scores[i][0]]  # 闭合数据
        ax1.plot(angles, option_scores, linewidth=2, label=option)
        ax1.fill(angles, option_scores, alpha=0.1)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(factors)
    ax1.set_ylim(0, 1)
    ax1.set_title('方案因素评估', fontsize=14)
    ax1.legend(loc='upper right')
    
    # 条形图 - 各方案的加权总分
    ax2.bar(options, weighted_scores, color=['#7C3AED', '#8B5CF6', '#A78BFA'])
    ax2.set_ylim(0, 1)
    ax2.set_title('方案综合评分', fontsize=14)
    ax2.set_ylabel('加权得分')
    
    # 标注最优解
    best_option_idx = weighted_scores.index(max(weighted_scores))
    ax2.annotate('帕累托最优解',
                xy=(options[best_option_idx], weighted_scores[best_option_idx]),
                xytext=(options[best_option_idx], weighted_scores[best_option_idx] + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center')
    
    plt.tight_layout()
    plt.savefig(f'static/decision_model_{user_id}.png', dpi=150, bbox_inches='tight')
    plt.close()

# 路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# 调用DeepSeek API
def call_deepseek_api(user_input, user_id, model, temperature, stream, api_key, base_url): # 添加 base_url 参数
    """调用DeepSeek API获取响应"""
    try:
        # 创建OpenAI客户端，使用DeepSeek的API
        client = OpenAI(api_key=api_key, base_url=base_url) # 使用传入的 base_url
        
        # 获取用户历史对话
        messages = []
        # 添加系统消息，无论是否有历史对话
        messages.append({"role": "system", "content": "你是一个认知图谱助手，可以帮助用户分析问题、做出决策并提供知识支持。"})
        
        # 获取历史对话
        if user_id in user_sessions and user_sessions[user_id]['chat_history']:
            for msg in user_sessions[user_id]['chat_history']:
                messages.append({"role": msg['role'], "content": msg['content']})
        
        # 添加用户当前输入
        messages.append({"role": "user", "content": user_input})
        
        # 调用API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream
        )
        
        # 获取回复内容
        if stream:
            # 处理流式响应
            content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            return content
        else:
            # 处理非流式响应
            return response.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek API调用错误: {str(e)}")
        # 发生错误时回退到模拟响应
        return None

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    app.logger.info(f"Received request data: {data}") # 添加日志记录
    user_input = data.get('message', '')
    user_id = data.get('user_id', 'default_user')
    # model_from_request = data.get('model') # 不再直接使用前端的模型名称，以避免错误
    temperature = data.get('temperature', 0.7)
    stream = data.get('stream', False)
    provider = data.get('provider', 'deepseek')
    
    api_key = data.get('apiKey', '')
    base_url_from_request = data.get('base_url')

    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    if provider == 'deepseek':
        if not api_key:
            return jsonify({'error': '未在请求体中提供DeepSeek API密钥'}), 400
        
        # 强制使用DeepSeek提供商配置中的默认模型，以确保模型有效
        actual_model = model_providers['deepseek']['default_model']
        
        # 确定要使用的 base_url
        # 如果前端提供了 base_url，则使用它；否则，使用 model_providers 中的配置
        actual_base_url = base_url_from_request or model_providers['deepseek']['base_url']
        
        app.logger.info(f"Forcing model to '{actual_model}' for DeepSeek. Using base_url: '{actual_base_url}'.")
        
        api_response = call_deepseek_api(user_input, user_id, actual_model, temperature, stream, api_key, actual_base_url)
        if api_response:
            # 存储用户对话历史
            if user_id not in user_sessions:
                user_sessions[user_id] = {
                    'chat_history': [],
                    'user_profile': defaultdict(int),
                    'knowledge_entities': set(),
                    'emotion_state': 'neutral'
                }
            
            # 添加用户消息到历史
            user_sessions[user_id]['chat_history'].append({
                'role': 'user',
                'content': user_input
            })
            
            # 添加助手回复到历史
            user_sessions[user_id]['chat_history'].append({
                'role': 'assistant',
                'content': api_response
            })
            
            # 提取关键词并更新用户画像
            keywords = extract_keywords(user_input)
            for keyword in keywords:
                user_sessions[user_id]['user_profile'][keyword] += 1
                user_sessions[user_id]['knowledge_entities'].add(keyword)
            
            # 生成各种可视化图谱
            generate_knowledge_graph(user_id)
            generate_thought_chain(user_input, user_id)
            generate_cognitive_map(user_id)
            generate_decision_model(user_id)
            
            # 返回响应和图片URL
            return jsonify({
                'response': api_response,
                'knowledge_graph_url': f'/static/knowledge_graph_{user_id}.png?t={os.path.getmtime(f"static/knowledge_graph_{user_id}.png") if os.path.exists(f"static/knowledge_graph_{user_id}.png") else 0}',
                'thought_chain_url': f'/static/thought_chain_{user_id}.png?t={os.path.getmtime(f"static/thought_chain_{user_id}.png") if os.path.exists(f"static/thought_chain_{user_id}.png") else 0}',
                'cognitive_map_url': f'/static/cognitive_map_{user_id}.png?t={os.path.getmtime(f"static/cognitive_map_{user_id}.png") if os.path.exists(f"static/cognitive_map_{user_id}.png") else 0}',
                'decision_model_url': f'/static/decision_model_{user_id}.png?t={os.path.getmtime(f"static/decision_model_{user_id}.png") if os.path.exists(f"static/decision_model_{user_id}.png") else 0}',
                'image_url': f'/static/knowledge_graph_{user_id}.png?t={os.path.getmtime(f"static/knowledge_graph_{user_id}.png") if os.path.exists(f"static/knowledge_graph_{user_id}.png") else 0}'
            })
        else:
            return jsonify({'error': 'DeepSeek API调用失败'}), 500
    else:
        # 如果不是 deepseek 提供商，或者未来支持其他提供商的逻辑
        return jsonify({'error': f'不支持的提供商: {provider} 或缺少必要的配置'}), 400

@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    """测试API连接"""
    data = request.json
    provider = data.get('provider', 'deepseek')
    # 从请求体中获取API密钥
    api_key = data.get('apiKey', '')
    model_name = data.get('modelName')
    base_url = data.get('baseUrl')
    
    # 验证必要参数
    if not api_key:
        return jsonify({'error': '缺少API密钥'}), 400
    
    if not model_name:
        return jsonify({'error': '缺少模型名称'}), 400
    
    # 模拟API连接测试
    # 实际项目中应该尝试连接到真实的API端点
    try:
        # 这里只是模拟成功，实际应用中应该进行真实的API调用测试
        # 例如，可以发送一个简单的请求到API，检查响应状态
        
        # 模拟不同提供商的验证逻辑
        if provider == 'deepseek' and not api_key.startswith('sk-'):
            return jsonify({'error': 'DeepSeek API密钥格式无效'}), 400
        
        if provider == 'openai' and not api_key.startswith('sk-'):
            return jsonify({'error': 'OpenAI API密钥格式无效'}), 400
        
        if provider == 'gemini' and len(api_key) < 10:
            return jsonify({'error': 'Gemini API密钥长度不足'}), 400
        
        # 返回成功响应
        return jsonify({
            'status': 'success',
            'message': '连接测试成功',
            'provider': provider,
            'model': model_name
        })
        
    except Exception as e:
        return jsonify({'error': f'连接测试失败: {str(e)}'}), 500

# 添加对话保存功能
@app.route('/api/save_conversation', methods=['POST'])
def save_conversation():
    """保存对话到本地文件"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        conversation_name = data.get('conversation_name', f"对话_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        save_path = data.get('save_path', 'conversations')  # 默认保存路径
        
        if user_id not in user_sessions:
            return jsonify({'error': '找不到用户会话'}), 404
        
        # 确保保存目录存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 准备保存的数据
        conversation_data = {
            'id': str(uuid.uuid4()),
            'name': conversation_name,
            'created_at': datetime.datetime.now().isoformat(),
            'user_id': user_id,
            'chat_history': user_sessions[user_id]['chat_history'],
            'user_profile': user_sessions[user_id].get('user_profile', {}),
            'knowledge_entities': list(user_sessions[user_id].get('knowledge_entities', set()))
        }
        
        # 生成文件名
        safe_name = "".join(c for c in conversation_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}_{conversation_data['id'][:8]}.json"
        filepath = os.path.join(save_path, filename)
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': '对话保存成功',
            'filepath': filepath,
            'conversation_id': conversation_data['id']
        })
        
    except Exception as e:
        return jsonify({'error': f'保存对话失败: {str(e)}'}), 500

@app.route('/api/load_conversations', methods=['GET'])
def load_conversations():
    """加载已保存的对话列表"""
    try:
        conversations_path = request.args.get('path', 'conversations')
        
        # 如果路径不存在，返回空列表而不是错误
        if not os.path.exists(conversations_path):
            print(f"路径不存在，将创建: {conversations_path}")
            try:
                os.makedirs(conversations_path, exist_ok=True)
            except Exception as e:
                print(f"创建目录失败: {e}")
            return jsonify({'conversations': []})
        
        if not os.path.isdir(conversations_path):
            return jsonify({'error': f'指定的路径不是一个文件夹: {conversations_path}'}), 400
        
        conversations = []
        total_files = 0
        error_files = 0
        
        for filename in os.listdir(conversations_path):
            if filename.endswith('.json'):
                total_files += 1
                filepath = os.path.join(conversations_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                        
                    # 验证必要字段
                    if not conversation.get('id'):
                        print(f"警告: 文件 {filename} 缺少ID字段")
                        error_files += 1
                        continue
                        
                    conversations.append({
                        'id': conversation.get('id'),
                        'name': conversation.get('name', '未命名对话'),
                        'created_at': conversation.get('created_at', ''),
                        'filepath': filepath,
                        'message_count': len(conversation.get('chat_history', []))
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误 {filename}: {e}")
                    error_files += 1
                    continue
                except Exception as e:
                    print(f"读取对话文件失败: {filename}, 错误: {e}")
                    error_files += 1
                    continue
        
        # 按创建时间排序
        conversations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        print(f"从 {conversations_path} 加载了 {len(conversations)} 个对话，共 {total_files} 个文件，{error_files} 个错误")
        
        return jsonify({
            'conversations': conversations,
            'total_files': total_files,
            'error_files': error_files,
            'path': conversations_path
        })
        
    except Exception as e:
        print(f"加载对话列表时发生错误: {e}")
        return jsonify({'error': f'加载对话列表失败: {str(e)}'}), 500

@app.route('/api/load_conversation/<conversation_id>', methods=['GET'])
def load_conversation(conversation_id):
    """加载特定对话"""
    try:
        conversations_path = request.args.get('path', 'conversations')
        
        # 检查路径是否存在
        if not os.path.exists(conversations_path):
            return jsonify({'error': f'指定的路径不存在: {conversations_path}'}), 404
        
        if not os.path.isdir(conversations_path):
            return jsonify({'error': f'指定的路径不是一个文件夹: {conversations_path}'}), 400
        
        # 搜索对话文件
        found_conversation = None
        for filename in os.listdir(conversations_path):
            if filename.endswith('.json'):
                filepath = os.path.join(conversations_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                        if conversation.get('id') == conversation_id:
                            found_conversation = conversation
                            print(f"找到对话文件: {filepath}")
                            print(f"对话包含 {len(conversation.get('chat_history', []))} 条消息")
                            break
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误 {filename}: {e}")
                    continue
                except Exception as e:
                    print(f"读取文件错误 {filename}: {e}")
                    continue
        
        if found_conversation:
            return jsonify(found_conversation)
        else:
            return jsonify({'error': f'在路径 {conversations_path} 中找不到ID为 {conversation_id} 的对话'}), 404
        
    except Exception as e:
        print(f"加载对话时发生错误: {e}")
        return jsonify({'error': f'加载对话失败: {str(e)}'}), 500

@app.route('/api/rename_conversation', methods=['POST'])
def rename_conversation():
    """重命名对话"""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        new_name = data.get('new_name')
        conversations_path = data.get('path', 'conversations')
        
        if not conversation_id or not new_name:
            return jsonify({'error': '缺少必要参数'}), 400
        
        for filename in os.listdir(conversations_path):
            if filename.endswith('.json'):
                filepath = os.path.join(conversations_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                    
                    if conversation.get('id') == conversation_id:
                        # 更新名称
                        conversation['name'] = new_name
                        conversation['updated_at'] = datetime.datetime.now().isoformat()
                        
                        # 生成新的文件名
                        safe_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        new_filename = f"{safe_name}_{conversation_id[:8]}.json"
                        new_filepath = os.path.join(conversations_path, new_filename)
                        
                        # 保存到新文件
                        with open(new_filepath, 'w', encoding='utf-8') as f:
                            json.dump(conversation, f, ensure_ascii=False, indent=2)
                        
                        # 删除旧文件（如果文件名不同）
                        if filepath != new_filepath:
                            os.remove(filepath)
                        
                        return jsonify({
                            'success': True,
                            'message': '重命名成功',
                            'new_filepath': new_filepath
                        })
                        
                except Exception as e:
                    continue
        
        return jsonify({'error': '找不到指定的对话'}), 404
        
    except Exception as e:
        return jsonify({'error': f'重命名失败: {str(e)}'}), 500

@app.route('/api/delete_conversation', methods=['DELETE'])
def delete_conversation():
    """删除对话"""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        conversations_path = data.get('path', 'conversations')
        
        if not conversation_id:
            return jsonify({'error': '缺少对话ID'}), 400
        
        for filename in os.listdir(conversations_path):
            if filename.endswith('.json'):
                filepath = os.path.join(conversations_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                    
                    if conversation.get('id') == conversation_id:
                        os.remove(filepath)
                        return jsonify({
                            'success': True,
                            'message': '删除成功'
                        })
                        
                except Exception as e:
                    continue
        
        return jsonify({'error': '找不到指定的对话'}), 404
        
    except Exception as e:
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

@app.route('/api/generate_knowledge_graph', methods=['POST'])
def generate_knowledge_graph_api():
    """生成知识图谱API"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        conversation_id = data.get('conversation_id')
        
        if conversation_id:
            # 从保存的对话生成知识图谱
            conversations_path = data.get('path', 'conversations')
            conversation_data = None
            
            for filename in os.listdir(conversations_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(conversations_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            conv = json.load(f)
                            if conv.get('id') == conversation_id:
                                conversation_data = conv
                                break
                    except Exception as e:
                        continue
            
            if not conversation_data:
                return jsonify({'error': '找不到指定的对话'}), 404
            
            graph_filename = generate_knowledge_graph(user_id, conversation_data)
        else:
            # 从当前会话生成知识图谱
            if user_id not in user_sessions:
                return jsonify({'error': '找不到用户会话'}), 404
            graph_filename = generate_knowledge_graph(user_id)
        
        return jsonify({
            'success': True,
            'graph_url': f'/static/{graph_filename}',
            'message': '知识图谱生成成功'
        })
        
    except Exception as e:
        return jsonify({'error': f'生成知识图谱失败: {str(e)}'}), 500

@app.route('/api/list_folders', methods=['GET'])
def list_folders():
    """列出可用的对话存储文件夹"""
    try:
        folders = []
        current_dir = os.getcwd()
        
        # 扫描当前目录下的文件夹
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.') and item != '__pycache__':
                # 检查是否包含JSON文件（对话文件）
                has_conversations = any(f.endswith('.json') for f in os.listdir(item_path))
                if has_conversations or item in ['conversations', 'saved_chats', 'archives', 'projects', 'backups']:
                    folders.append(item)
        
        # 添加一些常见的系统路径（如果存在）
        common_paths = [
            'conversations',
            'saved_chats', 
            'archives',
            'projects',
            'backups'
        ]
        
        for path in common_paths:
            if path not in folders and os.path.exists(path):
                folders.append(path)
        
        # 去重并排序
        folders = sorted(list(set(folders)))
        
        return jsonify({
            'folders': folders,
            'count': len(folders)
        })
        
    except Exception as e:
        return jsonify({'error': f'获取文件夹列表失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)