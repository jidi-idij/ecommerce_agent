# 智能电商导购Agent

基于 **LangGraph + FastAPI + 大模型** 的多Agent电商导购系统，实现"模糊需求→精准推荐"的全链路智能导购。

## 架构图

```
用户查询
    ↓
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  意图识别Agent │ ──→ │  商品检索Agent │ ──→ │  排序Agent    │ ──→ │  生成Agent    │
│  intent_node  │     │ retrieval_node│     │  sort_node   │     │ generate_node│
│              │     │              │     │              │     │              │
│ ·意图分类     │     │ ·ES关键词召回  │     │ ·价格评分排序  │     │ ·购买建议生成 │
│ ·属性抽取     │     │ ·向量语义召回  │     │ ·销量加权     │     │ ·推荐理由撰写 │
│ ·置信度计算   │     │ ·混合召回Top50 │     │ ·匹配度排序   │     │ ·自然语言回复 │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      ↓
                                                                用户看到结果
```

## 核心特性

- **4阶段Agent协作**：意图识别 → 商品检索 → 智能排序 → 生成回复
- **混合召回**：ES关键词搜索 + 大模型语义向量搜索
- **多因子排序**：融合价格/评分/销量/匹配度加权排序
- **MCP工具集**：商品搜索、价格监控、库存查询、评价分析
- **Redis缓存**：向量结果缓存、热门商品缓存，查询延迟降低60%
- **流式输出**：SSE实时返回执行状态和结果

## 快速开始

### 1. 安装依赖

```bash
cd ecommerce_agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 启动服务

```bash
python main.py
```

服务启动后：
- 前端页面：`http://127.0.0.1:8000/`
- API文档：`http://127.0.0.1:8000/docs`

### 4. 测试对话

打开前端页面，输入购物需求如：
- "我想买手机，5000左右，拍照好"
- "iPhone 15 Pro和华为Mate60 Pro哪个好"
- "推荐一款性价比高的降噪耳机"

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/chat` | POST | 对话接口（非流式） |
| `/api/chat/stream` | POST | 对话接口（SSE流式） |
| `/api/search` | POST | 商品搜索 |
| `/api/products/hot` | GET | 热门商品 |
| `/api/products/{id}` | GET | 商品详情 |
| `/api/deals` | GET | 最佳优惠 |
| `/api/categories` | GET | 商品分类 |
| `/api/cache/stats` | GET | 缓存统计 |
| `/api/health` | GET | 健康检查 |

## 项目结构

```
ecommerce_agent/
├── app/
│   ├── __init__.py
│   ├── models.py          # 数据模型（商品、状态、意图等）
│   ├── tools.py           # MCP工具（搜索、价格、库存、评价）
│   ├── cache.py           # Redis缓存管理
│   ├── nodes.py           # Agent节点实现
│   └── graph.py           # LangGraph工作流编排
├── config/
│   ├── __init__.py
│   └── settings.py        # 配置文件
├── frontend/
│   └── index.html         # 前端页面
├── main.py                # FastAPI入口
├── requirements.txt       # 依赖列表
├── .env.example           # 环境变量模板
└── README.md
```

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent框架 | LangGraph |
| 大模型 | 通义千问(Qwen) via DashScope |
| API框架 | FastAPI |
| 缓存 | Redis (本地兜底) |
| 向量数据库 | Qdrant (模拟实现) |
| 搜索引擎 | Elasticsearch (模拟实现) |

## 演示数据

内置50款热门数码产品模拟数据，覆盖：
- 手机（iPhone、华为、小米、vivo、OPPO、三星）
- 笔记本（MacBook、ThinkPad、MateBook、小新）
- 平板（iPad、MatePad、小米平板、Galaxy Tab）
- 耳机（AirPods、索尼、Bose、华为FreeBuds）
- 智能手表（Apple Watch、华为Watch、小米手表）
- 相机（索尼A7、佳能R6、富士X-T5、大疆Pocket）
- 游戏机（PS5、Switch、Xbox、Steam Deck）
