# db_manager.py
import aiomysql
import redis.asyncio as redis
import json
import logging
from datetime import datetime
from setting import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        try:
            self.pool = await aiomysql.create_pool(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                db=settings.mysql_database,
                autocommit=True
            )
            logger.info("MySQL connection pool created")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            raise

    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("MySQL connection pool closed")

async def query_all_db_content(pool: aiomysql.Pool, last_extraction: datetime = None) -> list:
    """
    Pulls published rows from the primary content tables and returns a list of dicts:
    {id, title, content, source, updated_at}
    """
    tables = [
        ("bravo_boats", "title", "content"),
        ("bravo_airport", "name", "description"),
        ("bravo_tours", "title", "content"),
        ("bravo_hotels", "title", "content"),
        ("bravo_spaces", "title", "content"),
        ("bravo_events", "title", "content"),
        ("bravo_cars", "title", "content")
    ]
    results = []

    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            for table, title_col, content_col in tables:
                try:
                    params = []
                    query = f"""
                        SELECT id, {title_col} AS title, {content_col} AS content, updated_at
                        FROM {table}
                        WHERE status = 'publish'
                    """
                    if last_extraction:
                        query += " AND updated_at > %s"
                        params.append(last_extraction)
                    await cursor.execute(query, params)
                    rows = await cursor.fetchall()
                    for row in rows:
                        updated = row.get("updated_at") or row.get("created_at") or datetime.now()
                        if isinstance(updated, (str,)):
                            updated_iso = updated
                        else:
                            updated_iso = updated.isoformat()
                        results.append({
                            "id": row["id"],
                            "title": row.get("title") or "",
                            "content": row.get("content") or "",
                            "source": table,
                            "updated_at": updated_iso
                        })
                except Exception as e:
                    logger.error(f"Error querying table {table}: {e}")
                    continue
    return results

async def query_db_content(pool: aiomysql.Pool, query: str, redis_client: redis.Redis = None) -> list:
    """
    Search structured DB content for 'query'.
    Attempts fulltext then falls back to LIKE for safety.
    Caches in redis if available.
    """
    cache_key = f"db_query:{query}"
    results = []
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Redis get error: {e}")

    tables = [
        ("bravo_boats", "title", "content"),
        ("bravo_airport", "name", "description"),
        ("bravo_tours", "title", "content"),
        ("bravo_hotels", "title", "content"),
        ("bravo_spaces", "title", "content"),
        ("bravo_events", "title", "content"),
        ("bravo_cars", "title", "content")
    ]

    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            for table, title_col, content_col in tables:
                try:
                    # First attempt: fulltext boolean search (if indexes exist)
                    sql_fulltext = f"""
                        SELECT id, {title_col} AS title, {content_col} AS content, updated_at
                        FROM {table}
                        WHERE status = 'publish'
                        AND MATCH({title_col}, {content_col}) AGAINST (%s IN BOOLEAN MODE)
                    """
                    try:
                        await cursor.execute(sql_fulltext, (query,))
                        rows = await cursor.fetchall()
                    except Exception:
                        # If fulltext fails (no index), fallback to LIKE
                        rows = []

                    if not rows:
                        like_q = f"%{query.replace('%','').replace('_','')}%"
                        like_sql = f"""
                            SELECT id, {title_col} AS title, {content_col} AS content, updated_at
                            FROM {table}
                            WHERE status = 'publish'
                            AND ({title_col} LIKE %s OR {content_col} LIKE %s)
                        """
                        await cursor.execute(like_sql, (like_q, like_q))
                        rows = await cursor.fetchall()

                    for row in rows:
                        updated = row.get("updated_at") or row.get("created_at") or datetime.now()
                        updated_iso = updated.isoformat() if not isinstance(updated, str) else updated
                        results.append({
                            "id": row["id"],
                            "title": row.get("title") or "",
                            "content": row.get("content") or "",
                            "source": table,
                            "updated_at": updated_iso
                        })
                except Exception as e:
                    logger.error(f"Error querying table {table}: {e}")
                    continue

    if redis_client and results:
        try:
            await redis_client.setex(cache_key, settings.redis_cache_ttl, json.dumps(results))
        except Exception as e:
            logger.error(f"Redis setex error: {e}")

    return results

