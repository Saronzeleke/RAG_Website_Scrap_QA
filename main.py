import logging
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import redis.asyncio as redis
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
import asyncio
import json
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import re
from setting import settings
from db_manager import DatabaseManager, query_all_db_content, query_db_content
from retrieval import RetrievalSystem, hybrid_retrieval
from urllib.robotparser import RobotFileParser as BaseRobotParser
import xml.etree.ElementTree as ET  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('qa.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  
app = FastAPI(title="Ethiopian_qa")

MAX_DEPTH = 50                    
MAX_PAGES = 10000              
CRAWL_CONCURRENCY = 50             
CONTEXT_CHAR_LIMIT = 16000         
SELENIUM_TIMEOUT = 100         
DYNAMIC_WAIT_TIME = 3             
MIN_CONTENT_LENGTH = 30           

class QueryRequest(BaseModel):
    question: str

class ScrapedPage(BaseModel):
    url: str
    title: str
    content: str
    depth: int
    is_dynamic: bool

class WebScraper:
    def __init__(self):
        self.driver = None
        self.robots_parser = {}
        self.session = None

    async def init(self):
        self.driver = self._setup_selenium()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Mozilla/5.0"}
        )

    def _setup_selenium(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(SELENIUM_TIMEOUT)
        return driver

    async def _check_robots_txt(self, url):
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self.robots_parser:
            rp = BaseRobotParser()
            try:
                robots_url = f"{domain}/robots.txt"
                async with self.session.get(robots_url) as resp:
                    if resp.status == 200:
                        rp.parse(await resp.text())
                self.robots_parser[domain] = rp
            except Exception:
                self.robots_parser[domain] = None
        return self.robots_parser.get(domain)

    async def _is_dynamic_page(self, url: str) -> bool:
        dynamic_indicators = [
            r'<div[^>]*class=".*(react|vue|angular).*"',
            r'<script[^>]*src=".*(react|vue|angular|jquery).*"',
            r'data-role=".*(dialog|popup|tab|accordion).*"',
            r'<div[^>]*id=".*(modal|lightbox|carousel).*"'
        ]
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    if any(re.search(pattern, html, re.I) for pattern in dynamic_indicators):
                        return True
        except Exception:
            logger.debug(f"Dynamic check failed for {url}")
        dynamic_keywords = [
            'search', 'filter', 'booking', 'tour', 
            'hotel', 'product', 'destination', 'packages',
            'ajax', 'load-more', 'results', 'checkout'
        ]
        return any(kw in url.lower() for kw in dynamic_keywords)

    async def scrape_page(self, url: str, depth: int) -> Optional[ScrapedPage]:
        try:
            robots = await self._check_robots_txt(url)
            if robots and not robots.can_fetch("*", url):
                logger.debug(f"Skipping {url} (disallowed by robots.txt)")
                return None
            is_dynamic = await self._is_dynamic_page(url)
            if is_dynamic:
                content = await self._scrape_with_selenium(url)
            else:
                content = await self._scrape_static(url)
            if not content or len(content.get("content", "")) < MIN_CONTENT_LENGTH:
                return None
            return ScrapedPage(
                url=url,
                title=content.get("title", ""),
                content=content["content"],
                depth=depth,
                is_dynamic=is_dynamic
            )
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {str(e)[:200]}")
            return None

    async def _scrape_with_selenium(self, url: str) -> Optional[Dict]:
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(DYNAMIC_WAIT_TIME)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            body = self.driver.find_element(By.TAG_NAME, "body")
            content = body.text.strip()
            return {
                "title": self.driver.title,
                "content": content,
                "url": url
            }
        except Exception as e:
            logger.warning(f"Selenium failed for {url}: {str(e)[:200]}")
            return None

    def _extract_content(self, html: str, url: str) -> Dict:
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(['script', 'style', 'iframe', 'noscript', 'svg', 'footer', 'nav']):
            element.decompose()
        metadata = {
            'headings': [],
            'tables': [],
            'lists': []
        }
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6','header']):
            metadata['headings'].append({
                'level': heading.name,
                'text': heading.get_text(strip=True)
            })
        for table in soup.find_all('table'):
            caption = table.find('caption')
            metadata['tables'].append({
                'caption': caption.get_text(strip=True) if caption else "",
                'rows': [[cell.get_text(" ", strip=True) for cell in row.find_all(['th', 'td'])] 
                         for row in table.find_all('tr')]
            })
        for list_tag in soup.find_all(['ul', 'ol']):
            metadata['lists'].append({
                'type': list_tag.name,
                'items': [item.get_text(" ", strip=True) for item in list_tag.find_all('li')]
            })
        title = ""
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title = og_title['content']
        elif soup.title:
            title = soup.title.text
        content_parts = []
        for section in soup.find_all(['article', 'main', 'section', 'div.content']):
            for p in section.find_all('p'):
                content_parts.append(p.get_text(" ", strip=True))
        if not content_parts and soup.body:
            content_parts.append(soup.body.get_text(" ", strip=True))
        full_content = {
            "title": title,
            "main_text": " ".join(content_parts),
            "metadata": metadata,
            "url": url
        }
        return full_content

    async def _scrape_static(self, url: str) -> Optional[Dict]:
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
                return self._extract_content(html, url)
        except Exception as e:
            logger.debug(f"Static scrape failed for {url}: {str(e)[:200]}")
            return None

    async def close(self):
        if self.driver:
            self.driver.quit()
        if self.session:
            await self.session.close()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    app.state.scraper = WebScraper()
    await app.state.scraper.init()
    app.state.mysql = DatabaseManager()
    await app.state.mysql.connect()
    app.state.redis = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True
    )
    await app.state.redis.ping()
    await _initialize_data()
    logger.info("Application ready")

async def _initialize_data():
    scraped_pages = await _crawl_site(settings.target_url)
    logger.info(f"Scraped {len(scraped_pages)} web pages")
    db_rows = await query_all_db_content(app.state.mysql.pool)
    logger.info(f"Loaded {len(db_rows)} database records")
    documents = _create_documents(scraped_pages, db_rows)
    app.state.retrieval = RetrievalSystem()
    app.state.retrieval.add_documents(documents)
    logger.info(f"Indexed {len(documents)} documents")

async def _fetch_sitemap_urls(base_url: str):
    sitemap_url = f"{base_url}/sitemap.xml"
    async with aiohttp.ClientSession() as session:
        async with session.get(sitemap_url) as resp:
            if resp.status == 200:
                xml = await resp.text()
                root = ET.fromstring(xml)
                urls = []
                for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    urls.append(loc.text)
                sub_urls = []
                for sub in urls:
                    async with session.get(sub) as sub_resp:
                        if sub_resp.status == 200:
                            sub_xml = await sub_resp.text()
                            sub_root = ET.fromstring(sub_xml)
                            for sub_loc in sub_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                                sub_urls.append(sub_loc.text)
                return sub_urls
    return []

async def _crawl_site(base_url: str) -> List[ScrapedPage]:
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    sitemap_urls = await _fetch_sitemap_urls(base_url)
    queue = [(url, 0) for url in sitemap_urls] + [(base_url, 0)]  
    scraped = set()
    results = []
    start_time = datetime.now()
    MAX_CRAWL_TIME = timedelta(hours=24)  
    
    while queue and len(scraped) < MAX_PAGES and (datetime.now() - start_time) < MAX_CRAWL_TIME:
        url, depth = queue.pop(0)
        url = _normalize_url(url, base_url)
        if url in scraped:
            continue
        if urlparse(url).netloc != base_domain:
            continue
        page = await app.state.scraper.scrape_page(url, depth)
        if page:
            results.append(page)
            scraped.add(url)
            links = await _extract_links(
                page.content if not page.is_dynamic else app.state.scraper.driver.page_source,
                base_url
            )
            queue.extend((link, depth+1) for link in links if link not in scraped)
            if len(results) % 10 == 0:
                logger.info(f"Progress: {len(results)} pages scraped, {len(queue)} URLs in queue")
        await asyncio.sleep(0.5)
    
    logger.info(f"Scraped {len(results)} pages (stopped after {len(scraped)} unique URLs or time limit)")
    return results

def _normalize_url(url: str, base_url: str) -> str:
    url = urldefrag(url)[0].split('?')[0].rstrip('/').lower()
    if not urlparse(url).netloc:
        url = urljoin(base_url, url)
    return url.rstrip('/')

async def _extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
            continue
        full_url = urljoin(base_url, href)
        full_url = _normalize_url(full_url, base_url)
        if not _is_resource_url(full_url):
            links.add(full_url)
    return list(links)

def _is_resource_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.zip', '.mp3', '.mp4', '.gif'])

def _create_documents(scraped_pages: List[ScrapedPage], db_rows: List[Dict]) -> List[Dict]:
    documents = []
    for i, page in enumerate(scraped_pages):
        documents.append({
            "id": f"web_{i}",
            "title": page.title,
            "content": page.content,
            "source": "web",
            "url": page.url
        })
    for row in db_rows:
        documents.append({
            "id": f"db_{row['id']}",
            "title": row.get("title", ""),
            "content": row.get("content", ""),
            "source": row.get("source", "mysql"),
            "url": ""
        })
    return documents

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    await app.state.scraper.close()
    await app.state.mysql.close()
    await app.state.redis.close()
    logger.info("Clean shutdown complete")

def _requires_list_answer(query: str) -> bool:
    list_phrases = [
        "list all", "list of", "what are the", "name all",
        "give me all", "show me all", "enumerate", "how many",
        "options for", "choices for", "types of", "kinds of",
        "varieties of", "examples of", "compare", "vs",
        "top [0-9]+", "best [0-9]+", "name [0-9]+"
    ]
    query_lower = query.lower()
    if any(re.search(phrase, query_lower) for phrase in list_phrases):
        return True
    if query_lower.startswith(('what are ', 'which are ', 'where are ')) and any(word.endswith('s') for word in query.split()[:5]):
        return True
    return False

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty question")
    try:
        cache_key = f"qa:{query}"
        if await app.state.redis.exists(cache_key):
            cached = await app.state.redis.get(cache_key)
            return {"answer": cached, "cached": True, "format": "list" if _requires_list_answer(query) else "paragraph"}
        db_results = await query_db_content(app.state.mysql.pool, query)
        chunks = await hybrid_retrieval(query, app.state.retrieval)
        context = _build_context(chunks, db_results)
        if not context:
            return {
                "answer": f"Information not found. Visit {settings.target_url}",
                "sources": [],
                "format": "paragraph"
            }
        if _requires_list_answer(query):
            answer = await _generate_list_answer(query, context)
            answer_type = "list"
        else:
            answer = await _generate_answer(query, context)
            answer_type = "paragraph"
        await app.state.redis.setex(cache_key, settings.redis_cache_ttl, answer)
        return {
            "answer": answer,
            "sources": [c.get("url", "") for c in chunks[:3] if c.get("url")],
            "format": answer_type
        }
    except Exception as e:
        logger.error(f"QA failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal error")

async def analyze_query(query: str) -> dict:
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url
    )
    prompt = f"""Analyze this travel query and return JSON response:
Query: "{query}"
Output format:
{{
  "intent": "fact|comparison|how_to|list_request",
  "entities": ["location", "activity", "timeframe"],
  "needs": ["pricing", "duration", "requirements"]
}}"""
    try:
        response = await client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"intent": "unknown", "entities": [], "needs": []}

def _build_context(chunks, db_results):
    context_parts = [f"[WEB] {doc['title']}\n{doc['content']}" for doc in chunks if doc.get("source") == "web"]
    context_parts.extend([f"[DB] {doc['title']}\n{doc['content']}" for doc in db_results])
    return "\n\n---\n\n".join(context_parts)[:CONTEXT_CHAR_LIMIT]

async def _generate_answer(query: str, context: str) -> str:
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url
    )
    prompt = f"""You are an expert travel assistant for Ethiopia. Answer the question using ONLY the provided context.
Question: {query}
Context:
{context}
Instructions:
1. Be specific and factual
2. Include relevant numbers, dates, or names when available
3. For comparison questions, list items clearly
4. For "how to" questions, provide step-by-step instructions
5. If multiple options exist, list them with bullet points
6. Cite sources using [URL] notation when available
Answer:"""
    try:
        response = await client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            top_p=0.9
        )
        answer = response.choices[0].message.content
        return _postprocess_answer(answer)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"I couldn't generate an answer. Please visit {settings.target_url}"

def _postprocess_answer(text: str) -> str:
    text = re.sub(r'\b(I think|I believe|probably|maybe|perhaps)\b', '', text, flags=re.I)
    if not text.endswith(('.', '!', '?')):
        text = text.rstrip() + '.'
    text = re.sub(r'^\s*[\-•]\s*$', '', text, flags=re.M)
    return text.strip()

def _clean_list_response(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    list_start = 0
    for i, line in enumerate(lines):
        if re.match(r'^\d+\.\s', line):
            list_start = i
            break
    current_num = 1
    for line in lines[list_start:]:
        if re.match(r'^\d+\.\s', line):
            line = re.sub(r'^\d+\.', f'{current_num}.', line)
            current_num += 1
        cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines)
    return re.sub(r'\n\n\d+\.\s.*$', '', result, flags=re.DOTALL)

async def _generate_list_answer(query: str, context: str) -> str:
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url
    )
    expected_items = _detect_expected_list_length(query)
    prompt = f"""You are an expert travel assistant for Ethiopia. 
Generate a numbered list answering this query:
Question: {query}
Available Context:
{context}
Instructions:
1. Create a numbered list with {expected_items} items
2. Each item should follow this format:
   [Number]. [Name/Title] - [Brief Description] [Relevant URL if available]
3. Keep descriptions concise (10-20 words)
4. Include specific details like prices, ratings when available
5. Prioritize items mentioned first in the context
6. If URLs are available, include them in brackets at the end
7. Add "Note:" at the end if any important disclaimers exist
Example:
1. Simien Mountains - Stunning mountain range with rare wildlife [https://www.visitethiopia.et/simien]
2. Lalibela Churches - 12th century rock-hewn churches [https://www.visitethiopia.et/lalibela]
Begin your response with the numbered list:"""
    response = await client.chat.completions.create(
        model="mistralai/mistral-small-3.2-24b-instruct:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
        top_p=0.9
    )
    return _clean_list_response(response.choices[0].message.content)

def _detect_expected_list_length(query: str) -> int:
    num_match = re.search(r'(top|list|best)\s(\d+)', query.lower())
    if num_match:
        return min(int(num_match.group(2)), 10)
    if "compare" in query.lower() or "vs" in query.lower():
        return 2
    if "top" in query.lower() or "best" in query.lower():
        return 5
    return 3

def _format_list_answer(text: str) -> str:
    lines = text.split('\n')
    list_start = next((i for i, line in enumerate(lines) if re.match(r'^\d+\.\s', line)), 0)
    formatted_lines = []
    expected_num = 1
    for line in lines[list_start:]:
        if re.match(r'^\d+\.\s', line):
            line = re.sub(r'^\d+\.', f'{expected_num}.', line)
            expected_num += 1
        formatted_lines.append(line)
    return '\n'.join(formatted_lines)

async def _generate_comparison_table(query: str, context: str) -> str:
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url
    )
    prompt = f"""Create a comparison table for this query using Markdown format:
Query: {query}
Format:
| Feature | Option 1 | Option 2 | Option 3 |
|---------|----------|----------|----------|
| Price   | $X       | $Y       | $Z       |
| ...     | ...      | ...      | ...      |
Context:
{context}
Include at least 4 comparison criteria."""
    response = await client.chat.completions.create(
        model="mistralai/mistral-small-3.2-24b-instruct:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
        top_p=0.9
    )
    return response.choices[0].message.content

def _validate_answer(text: str) -> str:
    unsafe_phrases = [
        "i think", "i believe", "as an ai",
        "i'm not sure", "i don't know"
    ]
    if any(phrase in text.lower() for phrase in unsafe_phrases):
        return f"I couldn't find specific details. Please visit {settings.target_url}"
    return text.strip()

async def get_vendor_info(query: str, context: str):
    if " vs " in query.lower() or "compare" in query.lower():
        answer = await _generate_comparison_table(query, context)
        return _validate_answer(answer)
    if "how can i" in query.lower() or "steps" in query.lower():
        raw_list = await _generate_list_answer(query, context)
        return _format_list_answer(raw_list)
    raw_answer = await _generate_answer(query, context)
    return _validate_answer(raw_answer)

async def _process_feedback_batch():
    feedbacks = await app.state.redis.lrange("feedback_log", 0, -1)
    list_feedback = [json.loads(f) for f in feedbacks if json.loads(f).get("is_list")]
    if list_feedback:
        avg_items = sum(f.get("item_count", 3) for f in list_feedback) / len(list_feedback)
        success_rate = sum(f.get("helpful", False) for f in list_feedback) / len(list_feedback)
        logger.info(f"List answer metrics: Avg items={avg_items:.1f}, Success rate={success_rate*100:.1f}%")
        global DEFAULT_LIST_LENGTH
        DEFAULT_LIST_LENGTH = min(max(round(avg_items), 3), 7)

async def log_feedback(question: str, answer: str, was_helpful: bool):
    feedback = {
        "question": question,
        "answer": answer,
        "helpful": was_helpful,
        "timestamp": datetime.now().isoformat()
    }
    try:
        await app.state.redis.rpush("feedback_log", json.dumps(feedback))
        if await app.state.redis.llen("feedback_log") > 100:
            await _process_feedback_batch()
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)