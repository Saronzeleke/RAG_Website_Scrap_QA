import aiomysql
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from pydantic import BaseModel
import json

logger = logging.getLogger(__name__)

class ScrapedPage(BaseModel):
    id: str
    table_name: str
    data: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

class DBManager:
    def __init__(self, pool: aiomysql.Pool, priority_table_prefixes: List[str]):
        self.pool = pool
        self.priority_table_prefixes = priority_table_prefixes
        self.all_tables = []
        self.priority_tables = []
        self.non_priority_tables = []
        self.table_schema_cache = {}
        self.valid_table_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    async def initialize(self):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW TABLES")
                self.all_tables = [row[0] for row in await cur.fetchall()]
                
                # Dynamic priority table identification
                self.priority_tables = [
                    t for t in self.all_tables 
                    if any(t.startswith(prefix) for prefix in self.priority_table_prefixes)
                ]
                self.non_priority_tables = [t for t in self.all_tables if t not in self.priority_tables]
                
                logger.info(f"DBManager initialized. Priority tables: {self.priority_tables}")

    def _validate_table_name(self, table: str) -> bool:
        return bool(self.valid_table_pattern.match(table))

    async def _get_table_columns(self, table: str) -> List[str]:
        if not self._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")
            
        if table in self.table_schema_cache:
            return self.table_schema_cache[table]
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    sql = f"SHOW COLUMNS FROM `{table}`"
                    await cur.execute(sql)
                    columns = [row[0] for row in await cur.fetchall()]
                    self.table_schema_cache[table] = columns
                    return columns
                except Exception as e:
                    logger.error(f"Error fetching columns for {table}: {e}")
                    return []

    async def save_page_data(self, page_data: Dict[str, Any], table: str = "scraped_pages"):
         """Save page data to appropriate table with proper duplicate handling"""
         if not self._validate_table_name(table):
            raise ValueError(f"Invalid table name: {table}")
        
         if table not in self.all_tables:
            logger.error(f"Table {table} does not exist")
            raise ValueError(f"Table {table} does not exist")

         async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    columns = await self._get_table_columns(table)
                    if not columns:
                        raise ValueError(f"No columns found for {table}")

                    # Enhanced column mapping with more flexibility
                    column_mapping = {
                        'title': next((c for c in ['title', 'name', 'heading', 'page_title'] if c in columns), None),
                        'content': next((c for c in ['content', 'description', 'data', 'text', 'body', 'page_content'] if c in columns), None),
                        'url': next((c for c in ['url', 'link', 'source_url', 'page_url'] if c in columns), None),
                        'timestamp': next((c for c in ['created_at', 'updated_at', 'scraped_at', 'timestamp'] if c in columns), None),
                        'metadata': next((c for c in ['metadata', 'attributes', 'properties', 'extra_data'] if c in columns), None),
                        'is_dynamic': next((c for c in ['is_dynamic', 'is_dynamic_page', 'dynamic'] if c in columns), None)
                    }

                    insert_data = {}
                    if column_mapping['title']:
                        insert_data[column_mapping['title']] = page_data.get('title', '')[:255]
                    if column_mapping['content']:
                        insert_data[column_mapping['content']] = page_data.get('content', '')[:65535]
                    if column_mapping['url']:
                        insert_data[column_mapping['url']] = page_data.get('url', '')[:512]
                    if column_mapping['timestamp']:
                        insert_data[column_mapping['timestamp']] = 'NOW()'
                    if column_mapping['metadata']:
                        insert_data[column_mapping['metadata']] = json.dumps({
                            'is_dynamic': page_data.get('is_dynamic', False),
                            'source': 'web_crawler',
                            'crawled_at': datetime.now().isoformat()
                        })[:1000]
                
                    if column_mapping['is_dynamic']:
                        insert_data[column_mapping['is_dynamic']] = page_data.get('is_dynamic', False)

                    if not insert_data:
                        raise ValueError(f"No compatible columns found in {table}")

                    # Use INSERT ... ON DUPLICATE KEY UPDATE to handle duplicates
                    columns_sql = ', '.join([f'`{k}`' for k in insert_data.keys()])
                    values_sql = ', '.join(['%s' if v != 'NOW()' else 'NOW()' for v in insert_data.values()])
                
                    # Create UPDATE clause for duplicate key scenario
                    update_clause = ', '.join([
                        f'`{k}` = VALUES(`{k}`)' 
                        for k in insert_data.keys() 
                        if k != column_mapping['url']  # Don't update URL on duplicate
                    ])
                
                    values = [v for v in insert_data.values() if v != 'NOW()']

                    query = f"""
    INSERT INTO `{table}` ({columns_sql}) 
    VALUES ({values_sql}) AS new
    ON DUPLICATE KEY UPDATE {', '.join([f'`{k}` = new.`{k}`' for k in insert_data.keys() if k != column_mapping['url']])}
"""
                
                    await cur.execute(query, values)
                    await conn.commit()
                
                    logger.debug(f"Saved/updated page in {table}: {page_data.get('url')}")
                    return True
                
                except Exception as e:
                    logger.error(f"Error saving page data to {table}: {e}")
                    await conn.rollback()
                    raise
    async def get_scraped_pages(self, limit_per_table: int = 100) -> List[ScrapedPage]:
        pages = []
        
        for table in self.priority_tables:
            try:
                table_pages = await self._fetch_table_pages(table, limit=None)
                pages.extend(table_pages)
            except Exception as e:
                logger.error(f"Error fetching from priority table {table}: {e}")

        for table in self.non_priority_tables:
            try:
                table_pages = await self._fetch_table_pages(table, limit=limit_per_table)
                pages.extend(table_pages)
            except Exception as e:
                logger.error(f"Error fetching from non-priority table {table}: {e}")

        return pages

    async def _fetch_table_pages(self, table: str, limit: Optional[int] = None) -> List[ScrapedPage]:
        if not self._validate_table_name(table):
            return []
            
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                columns = await self._get_table_columns(table)
                if not columns:
                    return []
                    
                query = f"SELECT {', '.join([f'`{col}`' for col in columns])} FROM `{table}`"
                if limit:
                    query += f" LIMIT {limit}"
                    
                await cur.execute(query)
                rows = await cur.fetchall()
                
                return [
                    ScrapedPage(
                        id=f"{table}_{row.get('id', str(uuid.uuid4()))}",
                        table_name=table,
                        data={k: str(v) if isinstance(v, (datetime, bytes)) else v for k, v in row.items() if v is not None}
                    ) for row in rows
                ]

    async def query_data(self, query_str: str, limit: int = 50) -> List[Dict]:
        if not query_str.strip():
            return []
            
        results = []
        is_id_query = query_str.startswith("id:")
        
        # Search priority tables first
        for table in self.priority_tables:
            try:
                table_results = await self._search_table(table, query_str, is_id_query, limit=None)
                results.extend(table_results)
            except Exception as e:
                logger.error(f"Error querying priority table {table}: {e}")

        # If no results found, search non-priority tables
        if not results and not is_id_query:
            for table in self.non_priority_tables:
                try:
                    table_results = await self._search_table(table, query_str, is_id_query, limit=limit//2)
                    results.extend(table_results)
                except Exception as e:
                    logger.error(f"Error querying non-priority table {table}: {e}")

        logger.info(f"Query '{query_str}' returned {len(results)} results")
        return results

    async def _search_table(self, table: str, query_str: str, is_id_query: bool, limit: Optional[int] = None) -> List[Dict]:
        if not self._validate_table_name(table):
            return []
            
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                columns = await self._get_table_columns(table)
                if not columns:
                    return []
                
                if is_id_query:
                    id_value = query_str.split(":", 1)[1]
                    if 'id' in columns:
                        query = f"SELECT * FROM `{table}` WHERE `id` = %s"
                        params = (id_value,)
                    else:
                        return []
                else:
                    search_columns = [col for col in columns if col in ['content', 'description', 'name', 'title', 'data']]
                    if not search_columns:
                        return []
                    conditions = [f"`{col}` LIKE %s" for col in search_columns]
                    query = f"SELECT * FROM `{table}` WHERE {' OR '.join(conditions)}"
                    params = [f"%{query_str}%"] * len(search_columns)
                
                if limit:
                    query += f" LIMIT {limit}"
                
                await cur.execute(query, params)
                rows = await cur.fetchall()
                
                return [{
                    'id': f"{table}_{row.get('id', str(uuid.uuid4()))}",
                    'table_name': table,
                    'title': row.get('title', row.get('name', '')) or '',
                    'content': row.get('content', row.get('description', row.get('data', ''))) or '',
                    'url': row.get('url', row.get('ical_import_url', '')) or ''
                } for row in rows]

    async def check_changes(self, since: datetime) -> List[Dict]:
        changes = []
        for table in self.priority_tables:
            try:
                table_changes = await self._check_table_changes(table, since)
                changes.extend(table_changes)
            except Exception as e:
                logger.error(f"Error checking changes in {table}: {e}")
        
        logger.info(f"Detected {len(changes)} changes since {since}")
        return changes

    async def _check_table_changes(self, table: str, since: datetime) -> List[Dict]:
        if not self._validate_table_name(table):
            return []
            
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                columns = await self._get_table_columns(table)
                timestamp_columns = ['updated_at', 'scraped_at', 'created_at', 'modified_at']
                ts_column = next((col for col in timestamp_columns if col in columns), None)
                
                if ts_column and 'id' in columns:
                    await cur.execute(
                        f"SELECT id, '{table}' as table_name, `{ts_column}` as changed_at FROM `{table}` WHERE `{ts_column}` > %s",
                        (since,)
                    )
                    rows = await cur.fetchall()
                    return [dict(row) for row in rows]
        return []

    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("DBManager connection pool closed")