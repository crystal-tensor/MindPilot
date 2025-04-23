from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

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
        # 'base_url': 'https://api.deepseek.com/v1',
        # 'default_model': 'DeepSeek-R1'
        'base_url': 'https://api.deepseek.com',
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
def generate_knowledge_graph(user_id):
    """生成用户知识图谱"""
    if user_id not in user_sessions:
        return
    
    # 创建图结构
    G = nx.Graph()
    
    # 添加节点和边
    profile = user_sessions[user_id]['user_profile']
    entities = list(user_sessions[user_id]['knowledge_entities'])
    
    # 添加中心节点
    G.add_node('用户', size=500, color='#7C3AED')
    
    # 添加实体节点和连接
    for entity, weight in profile.items():
        if len(entity) > 1:  # 过滤掉单字实体
            G.add_node(entity, size=100 + weight * 50, color='#C4B5FD')
            G.add_edge('用户', entity, weight=weight)
    
    # 实体间连接（基于共现）
    chat_history = user_sessions[user_id]['chat_history']
    for message in chat_history:
        if message['role'] == 'user':
            msg_keywords = extract_keywords(message['content'])
            for i in range(len(msg_keywords)):
                for j in range(i+1, len(msg_keywords)):
                    if G.has_node(msg_keywords[i]) and G.has_node(msg_keywords[j]):
                        if G.has_edge(msg_keywords[i], msg_keywords[j]):
                            G[msg_keywords[i]][msg_keywords[j]]['weight'] += 1
                        else:
                            G.add_edge(msg_keywords[i], msg_keywords[j], weight=1)
    
    # 绘制图谱
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # 节点大小和颜色
    node_sizes = [G.nodes[node].get('size', 100) for node in G.nodes()]
    node_colors = [G.nodes[node].get('color', '#C4B5FD') for node in G.nodes()]
    
    # 边的粗细
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    
    # 绘制
    nx.draw_networkx(
        G, pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color='gray',
        width=[w/2 for w in edge_weights],
        font_size=10,
        font_color='black',
        alpha=0.8
    )
    
    plt.title('用户知识图谱', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'static/knowledge_graph_{user_id}.png', dpi=150, bbox_inches='tight')
    plt.close()

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
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
def call_deepseek_api(user_input, user_id, model, temperature, stream, api_key):
    """调用DeepSeek API获取响应"""
    try:
        # 创建OpenAI客户端，使用DeepSeek的API
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
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
    user_input = data.get('message', '')
    user_id = data.get('user_id', 'default_user')
    model = data.get('model', 'deepseek-chat')
    temperature = data.get('temperature', 0.7)
    stream = data.get('stream', False)  # 默认不使用流式响应
    provider = data.get('provider', 'deepseek')
    
    # 获取API密钥
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    # 尝试调用DeepSeek API
    if provider == 'deepseek' and api_key:
        api_response = call_deepseek_api(user_input, user_id, model, temperature, stream, api_key)
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
                'decision_model_url': f'/static/decision_model_{user_id}.png?t={os.path.getmtime(f"static/decision_model_{user_id}.png") if os.path.exists(f"static/decision_model_{user_id}.png") else 0}'
            })
        else:
            return jsonify({'error': 'DeepSeek API调用失败'}), 500
    else:
        return jsonify({'error': '未提供DeepSeek API密钥'}), 400

@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    """测试API连接"""
    data = request.json
    provider = data.get('provider', 'deepseek')
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)