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
                    # Include description if it exists in the table
                    query = f"""
                        SELECT id, {title_col} AS title, {content_col} AS content,
                               COALESCE(description, '') AS description, updated_at
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
                        updated_iso = updated if isinstance(updated, str) else updated.isoformat()
                        # Combine content and description for richer context
                        combined_content = f"{row.get('content', '')} {row.get('description', '')}".strip()
                        results.append({
                            "id": row["id"],
                            "title": row.get("title", ""),
                            "content": combined_content,
                            "source": table,
                            "updated_at": updated_iso
                        })
                except Exception as e:
                    logger.error(f"Error querying table {table}: {e}")
                    continue
    return results

async def query_db_content(pool: aiomysql.Pool, query: str, redis_client: redis.Redis = None) -> list:
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
                    sql_fulltext = f"""
                        SELECT id, {title_col} AS title, {content_col} AS content,
                               COALESCE(description, '') AS description, updated_at
                        FROM {table}
                        WHERE status = 'publish'
                        AND MATCH({title_col}, {content_col}, description) AGAINST (%s IN BOOLEAN MODE)
                    """
                    try:
                        await cursor.execute(sql_fulltext, (query,))
                        rows = await cursor.fetchall()
                    except Exception:
                        rows = []

                    if not rows:
                        like_q = f"%{query.replace('%','').replace('_','')}%"
                        like_sql = f"""
                            SELECT id, {title_col} AS title, {content_col} AS content,
                                   COALESCE(description, '') AS description, updated_at
                            FROM {table}
                            WHERE status = 'publish'
                            AND ({title_col} LIKE %s OR {content_col} LIKE %s OR description LIKE %s)
                        """
                        await cursor.execute(like_sql, (like_q, like_q, like_q))
                        rows = await cursor.fetchall()

                    for row in rows:
                        updated = row.get("updated_at") or row.get("created_at") or datetime.now()
                        updated_iso = updated if isinstance(updated, str) else updated.isoformat()
                        # Combine content and description
                        combined_content = f"{row.get('content', '')} {row.get('description', '')}".strip()
                        results.append({
                            "id": row["id"],
                            "title": row.get("title", ""),
                            "content": combined_content,
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