"""全局配置 - 精简版"""
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env = os.path.join(_root, '.env')
    if os.path.exists(_env):
        load_dotenv(_env, override=True)
except ImportError:
    pass


@dataclass
class LLMConfig:
    """大模型配置"""
    api_key: str = "sk-ws-H.RYDIHEI.lko8.MEUCIBfa4jz5s4x_eKTSvEab6m-xjrV9OZNu6M9S-JECmMmPAiEAukWscoqVYhoVjU8s5f27nY26zNjHIDdVBZZU2bAvaF8"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.7-max"
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_connect_timeout: int = 1
    socket_timeout: int = 1
    cache_ttl: int = 3600


@dataclass
class AppConfig:
    """应用配置"""
    top_k: int = 20
    price_drop_threshold: float = 0.05
    log_level: str = "INFO"


# 全局实例
llm_config = LLMConfig()
redis_config = RedisConfig()
app_config = AppConfig()


def load_from_env():
    """从环境变量加载"""
    if os.getenv("DASHSCOPE_API_KEY"):
        llm_config.api_key = os.getenv("DASHSCOPE_API_KEY")
    if os.getenv("REDIS_HOST"):
        redis_config.host = os.getenv("REDIS_HOST")


load_from_env()
