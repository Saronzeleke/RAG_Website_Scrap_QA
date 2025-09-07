
from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field
from urllib.parse import urlparse
from typing import List, Optional

class Settings(BaseSettings):
    # Database configuration
    db_host: str = Field(default="localhost", validation_alias="DB_HOST")
    db_port: int = Field(default=3306, validation_alias="DB_PORT")
    db_user: str = Field(default="root", validation_alias="DB_USER")
    db_password: str = Field(default="", validation_alias="DB_PASSWORD")
    db_name: str = Field(default="visitethiopia", validation_alias="DB_NAME")
    db_minsize: int = Field(default=1, validation_alias="DB_MINSIZE")
    db_maxsize: int = Field(default=10, validation_alias="DB_MAXSIZE")

    # Priority table configuration
    priority_table_prefixes: List[str] = Field(
        default=["bravo_", "core_pages","scraped_pages"],
        validation_alias="PRIORITY_TABLE_PREFIXES"
    )
    
    # Scraper configuration
    base_url: str = Field(default="https://visitethiopia.et", validation_alias="BASE_URL")
    crawl_concurrency: int = Field(default=10, validation_alias="CRAWL_CONCURRENCY")
    max_pages: int = Field(default=50000, validation_alias="MAX_PAGES")
    scrape_timeout: int = Field(default=1000, validation_alias="SCRAPE_TIMEOUT")
    
    # Politeness settings
    request_delay: float = Field(default=2.0, validation_alias="REQUEST_DELAY")
    max_retries: int = Field(default=4, validation_alias="MAX_RETRIES")
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; VisitEthiopiaBot/1.0; +https://visitethiopia.et/bot)",
        validation_alias="USER_AGENT"
    )
    
    # Retrieval configuration
    chunk_size: int = Field(default=1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, validation_alias="CHUNK_OVERLAP")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2", validation_alias="CROSS_ENCODER_MODEL")
    min_score_threshold: float = Field(default=0.18, validation_alias="MIN_SCORE_THRESHOLD")
    
    # Hybrid retrieval weights
    semantic_weight: float = Field(default=0.5, validation_alias="SEMANTIC_WEIGHT")
    bm25_weight: float = Field(default=0.3, validation_alias="BM25_WEIGHT")
    positional_weight: float = Field(default=0.2, validation_alias="POSITIONAL_WEIGHT")
    
    # Context compression
    max_context_chunks: int = Field(default=3, validation_alias="MAX_CONTEXT_CHUNKS")
    min_content_length: int = Field(default=50, validation_alias="MIN_CONTENT_LENGTH")
    # API configuration
    openrouter_api_key: str = Field(..., validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    redis_cache_ttl: int = Field(default=3600, validation_alias="REDIS_CACHE_TTL")
    rate_limit_requests: int = Field(default=100, validation_alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, validation_alias="RATE_LIMIT_WINDOW")
    
    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    # CORS Settings 
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8501",
            "https://your-streamlit-app.streamlit.app"
        ],
        validation_alias="ALLOWED_ORIGINS"
    )
    #  Computed Properties (via @property) 
    @property
    def redis_parsed(self):
        return urlparse(self.redis_url.strip())

    @property
    def redis_host(self) -> str:
        return self.redis_parsed.hostname or "localhost"

    @property
    def redis_port(self) -> int:
        return self.redis_parsed.port or 6379

    @property
    def redis_password(self) -> Optional[str]:
        return self.redis_parsed.password

    @property
    def redis_path(self) -> str:
        return self.redis_parsed.path.strip("/") or "0"

    # Model config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate
settings = Settings()