"""
智能电商导购Agent - 配置文件
包含大模型API、Redis、向量数据库等配置
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """大模型配置"""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "sk-ws-H.RYDIHEI.lko8.MEUCIBfa4jz5s4x_eKTSvEab6m-xjrV9OZNu6M9S-JECmMmPAiEAukWscoqVYhoVjU8s5f27nY26zNjHIDdVBZZU2bAvaF8"
    model: str = "qwen3.7-plus"
    embedding_model: str = "text-embedding-v1"
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    charset: str = "utf-8"
    decode_responses: bool = True
    socket_connect_timeout: int = 1
    socket_timeout: int = 1
    # 缓存过期时间（秒）
    cache_ttl: int = 3600  # 1小时
    # 热门商品缓存key前缀
    hot_product_prefix: str = "hot_product:"


@dataclass
class VectorDBConfig:
    """向量数据库配置（Qdrant）"""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "ecommerce_products"
    vector_size: int = 1536  # 向量维度
    # 混合搜索权重
    vector_weight: float = 0.7  # 向量搜索权重
    keyword_weight: float = 0.3  # 关键词搜索权重


@dataclass
class ElasticsearchConfig:
    """Elasticsearch配置"""
    hosts: list = field(default_factory=lambda: ["http://localhost:9200"])
    index_name: str = "products"
    # 关键词搜索字段权重
    title_boost: float = 3.0
    description_boost: float = 1.0
    category_boost: float = 2.0


@dataclass
class AppConfig:
    """应用配置"""
    # 召回数量
    top_k_vector: int = 50  # 向量召回TopK
    top_k_keyword: int = 50  # 关键词召回TopK
    top_k_rerank: int = 20  # 精排后返回Top20
    
    # 排序权重
    sort_weight_price: float = 0.2    # 价格权重
    sort_weight_rating: float = 0.35  # 评分权重
    sort_weight_sales: float = 0.3    # 销量权重
    sort_weight_match: float = 0.15   # 匹配度权重
    
    # 价格监控阈值
    price_drop_threshold: float = 0.05  # 降价5%触发通知
    
    # 日志
    log_level: str = "INFO"


# 全局配置实例
llm_config = LLMConfig()
redis_config = RedisConfig()
vector_db_config = VectorDBConfig()
es_config = ElasticsearchConfig()
app_config = AppConfig()


def load_from_env():
    """从环境变量加载配置"""
    if os.getenv("DASHSCOPE_API_KEY"):
        llm_config.api_key = os.getenv("DASHSCOPE_API_KEY")
    if os.getenv("REDIS_HOST"):
        redis_config.host = os.getenv("REDIS_HOST")
    if os.getenv("QDRANT_HOST"):
        vector_db_config.host = os.getenv("QDRANT_HOST")
    if os.getenv("ES_HOST"):
        es_config.hosts = [os.getenv("ES_HOST")]
