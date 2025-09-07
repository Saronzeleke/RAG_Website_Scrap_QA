from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
import asyncio
import logging
import logging.config
from urllib.parse import urljoin, urlparse
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import yaml
import json
import re
import time
from datetime import datetime, timedelta
from db_manager import DBManager
from retrieval import Retriever
from setting import settings
import aiomysql
import warnings
import hashlib
from typing import Any
from spellchecker import SpellChecker
from nltk.corpus import wordnet
import nltk
warnings.filterwarnings("ignore", category=FutureWarning)
from openai import AsyncOpenAI
import unicodedata
from tenacity import retry, stop_after_attempt, wait_exponential
from crawler import run_crawler, crawler_status

app = FastAPI(title="VisitEthiopia QA System")
logging.config.dictConfig(yaml.load(open("logging_config.yaml"), Loader=yaml.SafeLoader))
logger = logging.getLogger(__name__)

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
spell_checker = SpellChecker()

openrouter_client = AsyncOpenAI(
    base_url=settings.openrouter_base_url,
    api_key=settings.openrouter_api_key,
    default_headers={
        "HTTP-Referer": "http://127.0.0.1:8000",  
        "X-Title": "VisitEthiopia QA System"       
    }
)

# Middleware for logging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            raise
        finally:
            process_time = (time.time() - start) * 1000
            client = request.client.host if request.client else "unknown"
            logger.info(json.dumps({
                "method": request.method,
                "path": str(request.url),
                "status_code": status,
                "latency_ms": round(process_time, 2),
                "client": client
            }))
        return response

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=500)
allowed_hosts = getattr(settings, "allowed_hosts", ["127.0.0.1", "localhost"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
origins = getattr(settings, "allowed_origins",[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "https://your-streamlit-app.streamlit.app"
    ])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)

    @validator('query')
    def prevent_sql_injection(cls, v):
        if any(char in v for char in [';', '--', '/*', '*/']):
            raise ValueError('Invalid characters in query')
        return v

class AnswerResponse(BaseModel):
    answer: str
    format: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    was_helpful: bool

class CacheInvalidationRequest(BaseModel):
    query: Optional[str] = None
    pattern: Optional[str] = None

# Crawl constants
SELENIUM_TIMEOUT = getattr(settings, "scrape_timeout", 1000)
INFINITE_SCROLL_STEPS = 12
INFINITE_SCROLL_DELAY_SEC = 2.0
MAX_CRAWL_TIME = timedelta(hours=24)
MAX_DEPTH = 1000
MIN_CONTENT_LENGTH = 10
TRACKING_PARAM_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source", "trk", "msclkid", "yclid", "igshid")

def normalize_url(url: str, base_domain: str = None) -> str:
    url = unicodedata.normalize('NFKC', url).strip()
    url = re.sub(r'\s+', ' ', url).strip()  
    
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    parsed = urlparse(url)
    if base_domain and parsed.netloc != base_domain:
        return None  
    return urljoin(url, parsed.path)

def same_domain(u: str, domain: str) -> bool:
    return urlparse(u).netloc == domain

async def on_page_handler(page_data: Dict[str, Any]):
    """Handle each scraped page"""
    try:
        if 'is_dynamic' not in page_data:
            page_data['is_dynamic'] = False 
        await app.state.db.save_page_data(page_data, "scraped_pages")
        
        # Add to retriever
        doc_id = f"scraped_{hashlib.md5(page_data['url'].encode()).hexdigest()}"
        app.state.retriever.add_documents([{
            'id': doc_id,
            'table_name': 'scraped_pages',
            'title': page_data.get('title', ''),
            'content': page_data.get('content', ''),
            'url': page_data.get('url', '')
        }])
        
        logger.debug(f"Processed page: {page_data['url']}")
        
    except Exception as e:
        logger.error(f"Error processing page {page_data.get('url')}: {e}")

def preprocess_query(query: str) -> str:
    """Preprocess query with spellcheck and synonym expansion"""
    try:
        # Spellcheck
        words = query.split()
        corrected_words = []
        for word in words:
            corrected = spell_checker.correction(word)
            corrected_words.append(corrected if corrected else word)
        corrected_query = " ".join(corrected_words)
        
        # Synonym expansion (limit to top synonym per word)
        expanded_words = []
        for word in corrected_words:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
                    if len(synonyms) >= 1: 
                        break
                if synonyms:
                    break
            expanded_words.append(word)
            if synonyms:
                expanded_words.append(list(synonyms)[0])
        
        expanded_query = " ".join(expanded_words)
        logger.debug(f"Preprocessed query: '{query}' -> '{expanded_query}'")
        return expanded_query
    except Exception as e:
        logger.warning(f"Query preprocessing failed: {e}")
        return query

async def analyze_query(query: str) -> dict:
    prompt = f"""Analyze this travel query and extract key elements. Return strict JSON:
Query: "{query}"
Output format:
{{
  "intent": "fact|comparison|how_to|list|description|summary|other",
  "key_entities": {{"locations": ["extracted locations"], "activities": ["extracted activities"], "events": ["extracted events"], "timeframes": ["extracted timeframes"]}},
  "needs": ["price", "duration", "requirements", "other extracted needs"]
}} 
Be precise and only include elements present in the query."""
    try:
        resp = await openrouter_client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[{"role":"user","content":prompt}],
            response_format={"type":"json_object"},
            temperature=0
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return {"intent": "other", "key_entities": {}, "needs": []}

def _postprocess_answer(text: str) -> str:
    unsafe = ["i think","i believe","probably","maybe","perhaps","i'm not sure","i don't know","as an ai","outside of my knowledge"]
    cleaned = text
    for p in unsafe:
        cleaned = re.sub(rf"\b{re.escape(p)}\b","",cleaned,flags=re.I)
    if not cleaned.strip() or any(s in cleaned.lower() for s in ["no information","not found"]):
        return f"I couldn't find specific details. Please visit {settings.base_url} for more information."
    cleaned = cleaned.strip()
    if not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'
    cleaned = re.sub(r'^\s*[\-•]\s*$', '', cleaned, flags=re.MULTILINE)
    return cleaned

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are an expert travel assistant specialized in Ethiopia tourism from {settings.base_url}. 
Answer ONLY using the provided context. Do not add external knowledge or assumptions.
If context lacks information, say so briefly.
Question: {query}
Context:
{context}
Instructions:
1) Be specific, factual, and concise. Include exact numbers, dates, names from context.
2) For 'how_to' queries, provide numbered steps.
3) Use bullet points for multiple items or options.
4) Cite sources with [URL] inline where relevant.
5) Keep response under 500 words.
Answer:
"""
    try:
        resp = await openrouter_client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=1000, top_p=0.9
        )
        return _postprocess_answer(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"I couldn't generate an answer. Please visit {settings.base_url}"

def _detect_expected_list_length(query: str) -> int:
    m = re.search(r'(top|list|best)\s(\d+)', query.lower())
    if m:
        return min(int(m.group(2)), 10)
    if "compare" in query.lower() or "vs" in query.lower(): return 2
    if "top" in query.lower() or "best" in query.lower(): return 5
    return 3

async def _generate_list_answer(query: str, context: str) -> str:
    expected = _detect_expected_list_length(query)
    prompt = f"""
You are an expert travel assistant for {settings.base_url}.
Generate a numbered list of exactly {expected} items based ONLY on the context.
Question: {query}
Context:
{context}
Rules:
- Format each item: [Number]. [Name/Title] - [Brief description, 20-50 words]. [URL if available]
- Prioritize items from scraped_pages or priority sources.
- If fewer than {expected} items in context, note "Limited information available" and list what you have.
- Add a 'Note:' section at end for any disclaimers.
Start directly with the list:
"""
    try:
        resp = await openrouter_client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1, max_tokens=1000, top_p=0.9
        )
        return _postprocess_answer(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"I couldn't generate an answer. Please visit {settings.base_url}"

async def _generate_comparison_table(query: str, context: str) -> str:
    prompt = f"""
Create a Markdown comparison table based ONLY on the provided context.
Query: {query}
Context:
{context}
Instructions:
- Columns: | Feature | Option 1 | Option 2 | ... (extract options from query/context)
- Rows: At least 4 criteria (e.g., Price, Location, Rating, Amenities, Duration)
- Use exact data from context; put 'N/A' if missing
- Add a row for 'Source' with [URL] for each option
- Keep table clean and readable
Start directly with the table:
"""
    try:
        resp = await openrouter_client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1, max_tokens=1000, top_p=0.9
        )
        return _postprocess_answer(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"I couldn't generate an answer. Please visit {settings.base_url}"

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    try:
        redis_client = redis.Redis(
            host=settings.redis_host, 
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True
        )
        await FastAPILimiter.init(redis_client)
        app.state.redis = redis_client
        logger.info("Redis connection established")
        logger.info(f"DB Config: host={settings.db_host}, user={settings.db_user}, db={settings.db_name}")
        app.state.db = DBManager(
            pool=await aiomysql.create_pool(
                host=settings.db_host, 
                port=settings.db_port, 
                user=settings.db_user,
                password=settings.db_password, 
                db=settings.db_name, 
                maxsize=10,
                minsize=1,     
                echo=False,   
            ),
            priority_table_prefixes=settings.priority_table_prefixes
        )
        await app.state.db.initialize()
        
        app.state.retriever = Retriever(
            embedding_model=settings.embedding_model,
            cross_encoder_model=settings.cross_encoder_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        await load_priority_documents()
        asyncio.create_task(run_crawler(
            base_url=str(settings.base_url),
            on_page_callback=on_page_handler,
            max_pages=settings.max_pages
        ))
        asyncio.create_task(db_refresher())

        logger.info("Startup completed successfully")

    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise

async def load_priority_documents():
    try:
        pages = await app.state.db.get_scraped_pages()
        documents = []
        for page in pages:
            title = page.data.get('title', page.data.get('name', 'Untitled'))
            content = page.data.get('content', page.data.get('description', page.data.get('data', '')))
            url = page.data.get('url', page.data.get('ical_import_url', ''))
            if content and len(content) >= MIN_CONTENT_LENGTH:
                documents.append({
                    'id': page.id,
                    'table_name': page.table_name,
                    'title': title,
                    'content': content,
                    'url': url
                })
        if documents:
            logger.info(f"Indexing {len(documents)} documents into retriever (startup)")
            app.state.retriever.add_documents(documents)
    except Exception as e:
        logger.error(f"load_priority_documents failed: {e}")

async def db_refresher():
    redis_client = app.state.redis
    while True:
        try:
            last_check = datetime.now() - timedelta(hours=1)
            changes = await app.state.db.check_changes(last_check)
            for change in changes:
                results = await app.state.db.query_data(f"id:{change['id']}")
                app.state.retriever.add_documents(results)
            await redis_client.set("last_db_check", datetime.now().isoformat())
        except Exception as e:
            logger.error(f"DB refresher error: {e}")
        await asyncio.sleep(3600)

@app.on_event("shutdown")
async def shutdown_event():
    app.state.db.pool.close()
    await app.state.db.pool.wait_closed()
    await app.state.redis.close()
@app.get("/crawl_status")
async def get_crawl_status():
    """Return current crawler status"""
    try:
        status = crawler_status
        return {
            "status": "running" if status.get("is_running", False) else "idle",
            "crawled_pages": status.get("crawled_count", 0),
            "queue_size": status.get("queue_size", 0),
            "last_crawled_url": status.get("last_crawled_url", ""),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving crawl status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve crawl status")

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        # Preprocess query
        processed_query = preprocess_query(query.query)
        
        cache_key = f"qa:{hashlib.md5(processed_query.encode()).hexdigest()}"
        cached_answer = await app.state.redis.get(cache_key)
        if cached_answer:
            return {
                "answer": cached_answer,
                "sources": [],
                "format": "cached",
                "cached": True
            }
        # Search both database and vector store
        db_results = await app.state.db.query_data(processed_query)
        vector_results = app.state.retriever.retrieve(
            processed_query,
            top_k=5,
            semantic_weight=settings.semantic_weight,
            bm25_weight=settings.bm25_weight,
            positional_weight=settings.positional_weight
        )

        # Combine results
        all_context = []

        # Add database results
        for result in db_results:
            all_context.append({
                'content': f"{result.get('title', '')} {result.get('content', '')}",
                'url': result.get('url', ''),
                'source': 'database'
            })

        # Add vector results
        for result in vector_results:
            all_context.append({
                'content': f"{result['metadata'].get('title', '')} {result.get('content', '')}",
                'url': result['metadata'].get('url', ''),
                'source': 'vector_store',
                'score': result.get('final_score', 0)
            })

        # Sort by relevance (database first, then by score)
        all_context.sort(key=lambda x: (0 if x['source'] == 'database' else 1, -x.get('score', 0)))

        # Build context text
        context_text = "\n\n".join([
            f"Source: {ctx['url']}\nContent: {ctx['content'][:1000]}"
            for ctx in all_context[:settings.max_context_chunks]
        ])

        if not context_text.strip():
            raise HTTPException(status_code=404, detail="No relevant information found. For more information please vist : {settings.base_url}.")

        # Analyze intent
        analysis = await analyze_query(processed_query)
        intent = analysis.get("intent", "other")

        # Generate answer based on intent
        if intent == "list":
            response = await _generate_list_answer(processed_query, context_text)
        elif intent == "comparison":
            response = await _generate_comparison_table(processed_query, context_text)
        elif intent in ["description", "summary"]:
            # Return full content from best result
            best_result = vector_results[0] if vector_results else all_context[0]
            response = best_result['content'].strip()
        else:
            response = await _generate_answer(processed_query, context_text)

        # Extract unique sources
        sources = list(set(ctx['url'] for ctx in all_context if ctx['url']))

        # Cache the response
        await app.state.redis.setex(cache_key, 3600, response)

        return {
            "answer": response,
            "sources": sources,
            "format": intent
        }

    except Exception as e:
        logger.error(f"Error processing query {query.query}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, request: Request, rate_limit=Depends(RateLimiter(times=5, seconds=60))):
    try:
        if not feedback.question or not feedback.answer:
            raise HTTPException(status_code=400, detail="Question and answer are required")
        feedback_data = {
            "question about https://visitethiopia.et": feedback.question,
            "answer": feedback.answer,
            "helpful": feedback.was_helpful,
            "timestamp": datetime.now().isoformat(),
        }
        await app.state.redis.rpush("feedback_log", json.dumps(feedback_data))
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@app.post("/admin/cache/invalidate")
async def invalidate_cache(request: CacheInvalidationRequest):
    try:
        if request.query:
            cache_key = f"qa:{hashlib.md5(request.query.encode()).hexdigest()}"
            deleted = await app.state.redis.delete(cache_key)
            return {"message": f"Invalidated {deleted} cache entry for query: {request.query}"}
        elif request.pattern:
            keys = await app.state.redis.keys(f"qa:*{request.pattern}*")
            if keys:
                deleted = await app.state.redis.delete(*keys)
                return {"message": f"Invalidated {deleted} cache entries matching pattern: {request.pattern}"}
            else:
                return {"message": "No cache entries found matching the pattern"}
        else:
            raise HTTPException(status_code=400, detail="Either query or pattern must be provided")
    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    try:
        await app.state.redis.ping()
        health_status["components"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    try:
        async with app.state.db.pool.acquire():
            pass
        health_status["components"]["mysql"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["mysql"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)