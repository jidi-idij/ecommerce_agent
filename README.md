# 智能电商导购Agent v4.0

> 基于 LangGraph + FastAPI 的智能电商导购系统，支持用户模式和开发者模式（自动化测试 + LLM 诊断）。

## 项目结构

```
ecommerce_agent_v4.0/
├── config/
│   ├── __init__.py
│   └── settings.py              # 全局配置（LLM / Redis / 向量库）
├── frontend/
│   └── index.html               # 单页应用（用户模式 + 开发者模式）
├── app/
│   ├── __init__.py
│   ├── models.py                # 数据模型（商品 / 意图 / AgentState）
│   ├── cache.py                 # Redis 缓存管理（含本地兜底）
│   ├── tools.py                 # MCP 工具层（搜索 / 价格 / 库存 / 评价）
│   ├── nodes.py                 # LangGraph 节点（意图 / 检索 / 排序 / 生成）
│   ├── graph.py                 # 工作流编排 + 路由
│   └── monitor.py               # 精简版监控（单文件，核心功能完整）
├── main.py                      # FastAPI 主应用
├── test_templates.yaml          # 测试模板库（10个模板）
└── requirements.txt             # 精简依赖（11个包）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

方式一：创建 `.env` 文件（推荐）
```bash
# 项目根目录创建 .env
echo "DASHSCOPE_API_KEY=sk-your-key" > .env
```

方式二：直接修改配置
```python
# config/settings.py
api_key: str = "sk-your实际key"
```

### 3. 启动服务

```bash
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问

- 用户模式：`http://localhost:8000/`
- 开发者模式：`http://localhost:8000/dev`

## 核心架构

### LangGraph 工作流

```
用户查询 → intent_node（意图识别）→ 路由 → 
    ├── product_search → retrieval_node（双通道检索）→ sort_node（融合排序）→ generate_node（生成回复）
    ├── price_compare → price_compare_node（比价）
    ├── after_sales → after_sales_node（售后）
    └── general → general_chat_node（闲聊）
```

### 双通道检索 + 融合排序

| 通道 | 方法 | 说明 |
|------|------|------|
| **关键词搜索** | 关键词匹配 + 属性过滤 | 精准命中价格、品牌、分类 |
| **语义搜索** | 词频向量 + 余弦相似度 | 召回语义相关商品 |

**融合公式**：`score = 0.6 × kw_score + 0.4 × sem_score`

### Monitor 监控中心

单文件实现，功能完整：

- **会话追踪**：开始/结束/节点耗时/异常捕获
- **SSE 实时监控**：浏览器实时查看执行流程
- **批量测试**：YAML 模板生成 → 批量执行 → LLM 综合分析
- **LLM 诊断**：单条报告智能诊断 + 批量分析报告

### 开发者模式

点击「开发者模式」切换，功能：

| 按钮 | 功能 |
|------|------|
| 5条快速测试 | 生成5条用例并执行 |
| 10条标准测试 | 生成10条用例并执行 |
| 20条完整测试 | 生成20条用例并执行 |
| 实时监控 | SSE 流实时显示节点执行 |
| Agent回复内容 | 查看每条测试的完整生成内容 |
| 综合分析报告 | LLM 分析 + 自动兜底报告 |

## 数据模型

### 商品数据（50条）

覆盖7个分类：手机、笔记本、平板、耳机、智能手表、相机、游戏机

| 字段 | 说明 |
|------|------|
| title | 商品标题 |
| category | 分类 |
| brand | 品牌 |
| price | 现价 |
| original_price | 原价 |
| rating | 评分（1-5） |
| sales_count | 销量 |
| stock | 库存 |
| tags | 标签 |

### 意图类型

| 类型 | 说明 |
|------|------|
| product_search | 商品搜索 |
| price_compare | 比价 |
| after_sales | 售后 |
| recommendation | 推荐 |
| general | 闲聊 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 用户模式首页 |
| GET | `/dev` | 开发者模式首页 |
| POST | `/api/chat` | 单条对话 |
| POST | `/api/chat/stream` | SSE 流式对话 |
| GET | `/api/products/hot` | 热门商品 |
| GET | `/api/products/{id}` | 商品详情 |
| GET | `/api/deals` | 最佳优惠 |
| GET | `/api/categories` | 商品分类 |
| GET | `/api/cache/stats` | 缓存统计 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/monitor/stream` | SSE 监控流 |
| POST | `/api/dev/mode` | 切换开发者模式 |
| POST | `/api/dev/batch-test` | 批量测试 |
| GET | `/api/dev/batch-test/reports` | 历史报告 |
| GET | `/api/dev/batch-test/report/{sid}` | 单条报告 |
| GET | `/api/dev/batch-test/responses` | 回复内容列表 |

## 配置项

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | 空 |
| `REDIS_HOST` | Redis 主机 | localhost |
| `REDIS_PORT` | Redis 端口 | 6379 |

## 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| fastapi | ≥0.110 | Web 框架 |
| uvicorn | ≥0.29 | ASGI 服务器 |
| langchain | ≥0.3 | LangChain 核心 |
| langchain-openai | ≥0.2 | OpenAI 兼容客户端 |
| langgraph | ≥0.2 | 工作流编排 |
| openai | ≥1.30 | OpenAI SDK |
| redis | ≥5.0 | Redis 缓存 |
| PyYAML | ≥6.0 | YAML 解析 |
| python-dotenv | ≥1.0 | 环境变量 |
| pydantic | ≥2.0 | 数据验证 |

## License

MIT
