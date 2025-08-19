
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "visitethiopia"  
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    target_url: str = "https://www.visitethiopia.et"
    redis_cache_ttl: int = 3600  
    use_reranking: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

settings = Settings()