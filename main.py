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
from selenium.common.exceptions import TimeoutException, WebDriverException
import os
import re
from setting import settings
from db_manager import DatabaseManager
from retrival import RetrievalSystem
from db_manager import query_all_db_content
from retrival import hybrid_retrieval
from db_manager import query_db_content
# from robots import RobotFileParser 
from urllib.robotparser import RobotFileParser as BaseRobotParser

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
app = FastAPI(title=" Ethiopian_qa")
#**CONFIGURATION**
MAX_DEPTH = 50                    
MAX_PAGES = 5000                   
CRAWL_CONCURRENCY = 50             
CONTEXT_CHAR_LIMIT = 16000         
SELENIUM_TIMEOUT = 45             
DYNAMIC_WAIT_TIME = 3             
MIN_CONTENT_LENGTH = 200           

#**MODELS**

class QueryRequest(BaseModel):
    question: str

class ScrapedPage(BaseModel):
    url: str
    title: str
    content: str
    depth: int
    is_dynamic: bool
#**CORE SCRAPER ENGINE**
class WebScraper:
    def __init__(self):
        self.driver = None
        self.robots_parser = {}
        self.session = None

    async def init(self):
        """Initialize Selenium & HTTP session"""
        self.driver = self._setup_selenium()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Mozilla/5.0"}
        )

    def _setup_selenium(self):
        """Configure Chrome for headless scraping"""
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
        """Respect robots.txt rules"""
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
        """Improved dynamic page detection"""
        dynamic_indicators = [
        # DOM patterns
        r'<div[^>]*class=".*(react|vue|angular).*"',
        r'<script[^>]*src=".*(react|vue|angular|jquery).*"',
        # Common dynamic patterns
        r'data-role=".*(dialog|popup|tab|accordion).*"',
        r'<div[^>]*id=".*(modal|lightbox|carousel).*"'
       ]
    
        # First check HTML for dynamic indicators
        try:
            async with self.session.get(url) as resp:
             if resp.status == 200:
                html = await resp.text()
                if any(re.search(pattern, html, re.I) for pattern in dynamic_indicators):
                    return True
        except Exception:
            logger.debug(f"Dynamic check failed for {url}")
        pass
    
        # Fallback to URL patterns
        dynamic_keywords = [
        'search', 'filter', 'booking', 'tour', 
        'hotel', 'product', 'destination', 'packages',
        'ajax', 'load-more', 'results', 'checkout'
        ]
        return any(kw in url.lower() for kw in dynamic_keywords)

    async def scrape_page(self, url: str, depth: int) -> Optional[ScrapedPage]:
        """Scrape a single page (static or dynamic)"""
        try:
            # Check robots.txt
            robots = await self._check_robots_txt(url)
            if robots and not robots.can_fetch("*", url):
                logger.debug(f"Skipping {url} (disallowed by robots.txt)")
                return None

            # Dynamic page handling
            is_dynamic = await self._is_dynamic_page(url)  # Add await
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
        """Scrape JavaScript-heavy pages"""
        try:
            self.driver.get(url)
            
            # Wait for core content
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Scroll to trigger lazy loading
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(DYNAMIC_WAIT_TIME)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # Extract all text
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
        """Advanced content extraction with semantic preservation"""
        soup = BeautifulSoup(html, "html.parser")
    
        # Remove unwanted elements
        for element in soup(['script', 'style', 'iframe', 'noscript', 'svg', 'footer', 'nav']):
          element.decompose()

        # Extract metadata
        metadata = {
        'headings': [],
        'tables': [],
        'lists': []
        }
    
        # Capture headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6','header']):
         metadata['headings'].append({
            'level': heading.name,
            'text': heading.get_text(strip=True)
          })
        
    
         # Capture tables with context
         for table in soup.find_all('table'):
           caption = table.find('caption')
           metadata['tables'].append({
            'caption': caption.get_text(strip=True) if caption else "",
            'rows': [[cell.get_text(" ", strip=True) for cell in row.find_all(['th', 'td'])] 
                    for row in table.find_all('tr')]
            })
    
         # Capture lists
        for list_tag in soup.find_all(['ul', 'ol']):
          metadata['lists'].append({
            'type': list_tag.name,
            'items': [item.get_text(" ", strip=True) for item in list_tag.find_all('li')]
          })
    
          # Get title (prefer og:title if available)
        title = ""
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
          title = og_title['content']
        else:
          title = soup.title.text if soup.title else ""
    
         # Extract main content with semantic priority
        content_parts = []
        for section in soup.find_all(['article', 'main', 'section', 'div.content']):
           # Preserve paragraph structure
           for p in section.find_all('p'):
            content_parts.append(p.get_text(" ", strip=True))
    
          # Fallback to body if no structured content found
           if not content_parts:
              body = soup.body
        if body:
            content_parts.append(body.get_text(" ", strip=True))
    
               # Combine with metadata
        full_content = {
           "title": title,
            "main_text": " ".join(content_parts),
             "metadata": metadata,
             "url": url
              }
    
        return full_content
    async def _scrape_static(self, url: str) -> Optional[Dict]:
        """Fast static page scraping"""
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
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
        if self.session:
            await self.session.close()
#**FASTAPI CORE**
@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    
    # Initialize services
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
    
    #Load initial data
    await _initialize_data()
    logger.info("Application ready")
async def _initialize_data():
    """Load scraped web content first, then DB."""
    #Scrape website first
    scraped_pages = await _crawl_site(settings.target_url)
    logger.info(f"Scraped {len(scraped_pages)} web pages")
    
    #Then load from MySQL
    db_rows = await query_all_db_content(app.state.mysql.pool)
    logger.info(f"Loaded {len(db_rows)} database records")
    
    #Build document store (web first in list for priority if needed)
    documents = _create_documents(scraped_pages, db_rows)  
    
    app.state.retrieval = RetrievalSystem()
    app.state.retrieval.add_documents(documents)
    logger.info(f"Indexed {len(documents)} documents")

async def _crawl_site(base_url: str) -> List[ScrapedPage]:
    """Breadth-first site crawler without page/depth limits, but with safeguards."""
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    queue = [(base_url, 0)]
    scraped = set()
    results = []
    start_time = datetime.now()
    MAX_CRAWL_TIME = timedelta(hours=12)  
    
    while queue and len(scraped) < MAX_PAGES and (datetime.now() - start_time) < MAX_CRAWL_TIME:
        url, depth = queue.pop(0)
        url = _normalize_url(url, base_url)
        
        if url in scraped:
            continue
        
        # Domain control
        if urlparse(url).netloc != base_domain:
            continue
            
        page = await app.state.scraper.scrape_page(url, depth)
        if page:
            results.append(page)
            scraped.add(url)
            
            # Queue new links (no depth limit)
            links = await _extract_links(
                page.content if not page.is_dynamic 
                else app.state.scraper.driver.page_source,
                base_url
            )
            queue.extend((link, depth+1) for link in links if link not in scraped)
         # 🔹 Progress logging every 10 pages
            if len(results) % 10 == 0:
                logger.info(f"Progress: {len(results)} pages scraped, {len(queue)} URLs in queue")
        await asyncio.sleep(0.5)  # Polite delay to avoid overwhelming servers
    
    logger.info(f"Scraped {len(results)} pages (stopped after {len(scraped)} unique URLs or time limit)")
    return results

def _normalize_url(url: str, base_url: str) -> str:
    """Standardize URL format"""
    url = urldefrag(url)[0].split('?')[0].rstrip('/').lower()
    if not urlparse(url).netloc:
        url = urljoin(base_url, url)
    return url.lower().rstrip('/')

async def _extract_links(html: str, base_url: str) -> List[str]:
    """Extract all unique links from HTML"""
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
    """Filter out non-HTML resources"""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in [
        '.pdf', '.jpg', '.jpeg', '.png', 
        '.zip', '.mp3', '.mp4', '.gif'
    ])
def _create_documents(scraped_pages: List[ScrapedPage], db_rows: List[Dict]) -> List[Dict]:
    """Combine scraped content first, then DB."""
    documents = []
    
    # Add scraped pages first
    for i, page in enumerate(scraped_pages):
        documents.append({
            "id": f"web_{i}",
            "title": page.title,
            "content": page.content,
            "source": "web",
            "url": page.url
        })
    
    # Add database records
    for row in db_rows:
        documents.append({
            "id": f"db_{row['id']}",
            "title": row.get("title", ""),
            "content": row.get("content", ""),
            "source": row.get("source", "mysql"),
            "url": ""
        })
    
    return documents
    print(f"First web doc: {documents[0]}")  # After _create_documents()
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    await app.state.scraper.close()
    await app.state.mysql.close()
    await app.state.redis.close()
    logger.info("Clean shutdown complete")

# **QA ENDPOINT**
def _requires_list_answer(query: str) -> bool:
    """Enhanced list detection with query analysis"""
    list_phrases = [
        "list all", "list of", "what are the", "name all",
        "give me all", "show me all", "enumerate", "how many",
        "options for", "choices for", "types of", "kinds of",
        "varieties of", "examples of", "compare", "vs",
        "top [0-9]+", "best [0-9]+", "name [0-9]+"
    ]
    
    # Check for direct list phrases
    query_lower = query.lower()
    if any(re.search(phrase, query_lower) for phrase in list_phrases):
        return True
        
    # Check for plural questions
    if (query_lower.startswith(('what are ', 'which are ', 'where are ')) and \
       any(word.endswith('s') for word in query.split()[:5])):
        return True
        
    return False
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Enhanced QA endpoint with list support"""
    query = request.question.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Empty question")
    
    try:
        # Cache che
        cache_key = f"qa:{query}"
        if await app.state.redis.exists(cache_key):
            cached = await app.state.redis.get(cache_key)
            return {"answer": cached, "cached": True, "format": "list" if _requires_list_answer(query) else "paragraph"}
        
        # Hybrid retrieval
        db_results = await query_db_content(app.state.mysql.pool, query)
        chunks = await hybrid_retrieval(query, app.state.retrieval)
        
        # Build context
        context = _build_context(chunks, db_results)
        if not context:
            return {
                "answer": f"Information not found. Visit {settings.target_url}",
                "sources": [],
                "format": "paragraph"
            }
        
        # Generate appropriate answer type
        if _requires_list_answer(query):
            answer = await _generate_list_answer(query, context)
            answer_type = "list"
        else:
            answer = await _generate_answer(query, context)
            answer_type = "paragraph"
        
        # Cache response
        await app.state.redis.setex(
            cache_key, 
            settings.redis_cache_ttl, 
            answer
        )
        
        return {
            "answer": answer,
            "sources": [c.get("url", "") for c in chunks[:3] if c.get("url")],
            "format": answer_type
        }
    except Exception as e:
        logger.error(f"QA failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal error")

async def analyze_query(query: str) -> dict:
    """Classify query intent and extract entities"""
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
    # Put web content FIRST
    context_parts = [
        f"[WEB] {doc['title']}\n{doc['content']}"
        for doc in chunks if doc.get("source") == "web"
    ]
    # Add DB content later
    context_parts.extend([
        f"[DB] {doc['title']}\n{doc['content']}" 
        for doc in db_results
    ])
    return "\n\n---\n\n".join(context_parts)[:CONTEXT_CHAR_LIMIT]
# def _build_context(chunks: List[Dict], db_results: List[Dict]) -> str:  # Swapped params
#     """Combine relevant information for LLM, web first."""
#     context_parts = []
    
#     for doc in chunks + db_results:
#         if not doc.get("content"):
#             continue
            
#         source = doc.get("source", "unknown")
#         context_parts.append(
#             f"[{source.upper()}] {doc.get('title', '')}\n"
#             f"URL: {doc.get('url', '')}\n"
#             f"Content: {doc['content']}"
#         )
    
#     context = "\n\n---\n\n".join(context_parts)
#     return context[:CONTEXT_CHAR_LIMIT]
async def _generate_answer(query: str, context: str) -> str:
    """Enhanced answer generation with structured prompting"""
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url
    )
    
    # Build structured prompt
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
    """Clean and validate the generated answer"""
    # Remove hedging phrases
    text = re.sub(r'\b(I think|I believe|probably|maybe|perhaps)\b', '', text, flags=re.I)
    
    # Ensure complete sentences
    if not text.endswith(('.', '!', '?')):
        text = text.rstrip() + '.'
        
    # Remove empty bullet points
    text = re.sub(r'^\s*[\-•]\s*$', '', text, flags=re.M)
    
    return text.strip()
def _clean_list_response(text: str) -> str:
    """Ensure consistent list formatting"""
    lines = text.split('\n')
    cleaned_lines = []
    
    # Find where the numbered list starts
    list_start = 0
    for i, line in enumerate(lines):
        if re.match(r'^\d+\.\s', line):
            list_start = i
            break
    
    # Process only from list start onward
    current_num = 1
    for line in lines[list_start:]:
        # Fix numbering if needed
        if re.match(r'^\d+\.\s', line):
            line = re.sub(r'^\d+\.', f'{current_num}.', line)
            current_num += 1
        cleaned_lines.append(line)
    
    # Remove any trailing non-list content
    result = '\n'.join(cleaned_lines)
    return re.sub(r'\n\n\d+\.\s.*$', '', result, flags=re.DOTALL)
async def _generate_list_answer(query: str, context: str) -> str:
    """Enhanced list generation with better formatting control"""
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url
    )
    
    # Detect expected list length from query
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
        temperature=0.1,  # Very low for consistent formatting
        max_tokens=800,
        top_p=0.9
    )
    
    return _clean_list_response(response.choices[0].message.content)

def _detect_expected_list_length(query: str) -> int:
    """Determine how many items the user likely wants"""
    # Check for explicit numbers ("top 5", "list 3")
    num_match = re.search(r'(top|list|best)\s(\d+)', query.lower())
    if num_match:
        return min(int(num_match.group(2)), 10)  # Cap at 10 items
    
    # Default based on question type
    if "compare" in query.lower() or "vs" in query.lower():
        return 2
    if "top" in query.lower() or "best" in query.lower():
        return 5
    return 3  # Default number of items

query=" how can i become a vendor"
context="""You are an assistant for {setting.target_url}question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.\nQuestion: 
    {question} \nContext: {context} \nAnswer:")"""
def _format_list_answer(text: str) -> str:
    """Ensure consistent list formatting"""
    # Remove any introductory sentences before the list
    lines = text.split('\n')
    list_start = next((i for i, line in enumerate(lines) 
                     if re.match(r'^\d+\.\s', line)), 0)
    
    # Ensure proper numbering
    formatted_lines = []
    expected_num = 1
    for line in lines[list_start:]:
        if re.match(r'^\d+\.\s', line):
            line = re.sub(r'^\d+\.', f'{expected_num}.', line)
            expected_num += 1
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)
# raw_list = await _generate_list_answer(query, context)
# formatted_list = _format_list_answer(raw_list)
async def _generate_comparison_table(query: str, context: str) -> str:
    """Generate markdown-style comparison tables"""
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

def _validate_answer(text: str) -> str:
    """Ensure answer quality"""
    unsafe_phrases = [
        "i think", "i believe", "as an ai",
        "i'm not sure", "i don't know"
    ]
    
    if any(phrase in text.lower() for phrase in unsafe_phrases):
        return f"I couldn't find specific details. Please visit {settings.target_url}"
    return text.strip()
async def get_vendor_info(query: str, context: str):
    # Handle comparison queries
    if " vs " in query.lower() or "compare" in query.lower():
        answer = await _generate_comparison_table(query, context)
        return _validate_answer(answer)
    
    # Handle list-style answers
    if "how can i" in query.lower() or "steps" in query.lower():
        raw_list = await _generate_list_answer(query, context)
        return _format_list_answer(raw_list)
    
    # Default case for general answers
    raw_answer = await _generate_answer(query, context)
    return _validate_answer(raw_answer)
async def _process_feedback_batch():
    """Analyze feedback to improve list answers"""
    feedbacks = await app.state.redis.lrange("feedback_log", 0, -1)
    list_feedback = [json.loads(f) for f in feedbacks if json.loads(f).get("is_list")]
    
    if list_feedback:
        avg_items = sum(f.get("item_count", 3) for f in list_feedback) / len(list_feedback)
        success_rate = sum(f.get("helpful", False) for f in list_feedback) / len(list_feedback)
        
        logger.info(
            f"List answer metrics: Avg items={avg_items:.1f}, "
            f"Success rate={success_rate*100:.1f}%"
        )
        
        # Could adjust default list length based on feedback
        global DEFAULT_LIST_LENGTH
        DEFAULT_LIST_LENGTH = min(max(round(avg_items), 3, 7))
async def log_feedback(question: str, answer: str, was_helpful: bool):
    """Store user feedback for continuous improvement"""
    feedback = {
        "question": question,
        "answer": answer,
        "helpful": was_helpful,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Store in Redis temporarily
        await app.state.redis.rpush(
            "feedback_log", 
            json.dumps(feedback)
        )
        
        # Periodically process feedback (e.g., weekly)
        if await app.state.redis.llen("feedback_log") > 100:
            await _process_feedback_batch()
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
# import logging
# import aiohttp
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime, timedelta
# import redis.asyncio as redis
# from openai import AsyncOpenAI
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse, urljoin, urldefrag
# import mimetypes
# import asyncio
# import json
# from typing import List
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, WebDriverException

# from retrival import chunk_documents, RetrievalSystem, hybrid_retrieval
# from setting import settings
# from db_manager import DatabaseManager, query_all_db_content, query_db_content
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow logs
# # Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.FileHandler('qa.log'), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# app = FastAPI()
# # Constants
# MAX_DEPTH = 30
# MAX_PAGES = 1000
# TIMEOUT = aiohttp.ClientTimeout(total=30)
# CRAWL_CONCURRENCY = 100
# CONTEXT_CHAR_LIMIT = 16000  # keep prompt payload reasonable
# SELENIUM_TIMEOUT = 30  # seconds to wait for dynamic content

# class QueryRequest(BaseModel):
#     question: str

# def setup_selenium():
#     """Configure Selenium WebDriver with headless options"""
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     # chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--window-size=1920,1080")
#     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
#     try:
#         driver = webdriver.Chrome(options=chrome_options)
#         driver.set_page_load_timeout(SELENIUM_TIMEOUT)
#         return driver
#     except Exception as e:
#         logger.error(f"Failed to initialize Selenium: {e}")
#         return None

# async def scrape_with_selenium(driver, url):
#     """Scrape dynamic content using Selenium"""
#     try:
#         driver.get(url)
#         # Wait for main content to load
#         WebDriverWait(driver, SELENIUM_TIMEOUT).until(
#             EC.presence_of_element_located((By.TAG_NAME, "body"))
#         )
        
#         # Scroll to bottom to trigger lazy-loaded content
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         await asyncio.sleep(2)  # Give time for dynamic content to load
        
#         # Get the page source after dynamic content has loaded
#         page_source = driver.page_source
#         soup = BeautifulSoup(page_source, "html.parser")
        
#         # Remove unwanted elements
#         for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
#             element.decompose()
            
#         return str(soup)
#     except TimeoutException:
#         logger.warning(f"Timeout while loading {url}")
#         return None
#     except WebDriverException as e:
#         logger.warning(f"Selenium error for {url}: {e}")
#         return None

# @app.on_event("startup")
# async def startup_event():
#     logger.info("Service startup: init DB, Redis, load data, build retrieval")
#     # Initialize Selenium driver
#     app.state.selenium_driver = setup_selenium()

#     # MySQL
#     app.state.mysql = None
#     for attempt in range(3):
#         try:
#             app.state.mysql = DatabaseManager()
#             await app.state.mysql.connect()
#             break
#         except Exception as e:
#             logger.error(f"MySQL connect attempt {attempt+1} failed: {e}")
#             if attempt == 2:
#                 raise
#             await asyncio.sleep(2)

#     # Redis
#     try:
#         app.state.redis = redis.Redis(
#             host=settings.redis_host,
#             port=settings.redis_port,
#             db=settings.redis_db,
#             decode_responses=True
#         )
#         await app.state.redis.ping()
#         logger.info("Redis connected")
#     except Exception as e:
#         logger.warning(f"Redis unavailable: {e}")
#         app.state.redis = None

#     # Load DB content + web content
#     last_extraction = datetime.now() - timedelta(days=365)
#     try:
#         db_rows = await query_all_db_content(app.state.mysql.pool, last_extraction)
#         logger.info(f"DB rows loaded: {len(db_rows)}")
#     except Exception as e:
#         logger.error(f"Failed loading DB rows: {e}")
#         db_rows = []

#     try:
#         web_content = await scrape_site(settings.target_url, app.state.selenium_driver)
#         logger.info(f"Web pages scraped: {len(web_content)}")
#     except Exception as e:
#         logger.error(f"Failed scraping site: {e}")
#         web_content = []

#     # Build documents list (DB rows first, then web pages)
#     documents = []
#     for r in db_rows:
#         documents.append({
#             "id": f"db_{r['id']}_{r['source']}",
#             "title": r.get("title", ""),
#             "content": r.get("content", ""),
#             "source": r.get("source", ""),
#             "updated_at": r.get("updated_at", datetime.now().isoformat())
#         })
    
#     for i, page in enumerate(web_content):
#         documents.append({
#             "id": f"web_{i}",
#             "title": page.get("title", ""),
#             "content": page.get("content", ""),
#             "source": "web",
#             "url": page.get("url", ""),
#             "updated_at": datetime.now().isoformat()
#         })

#     # Chunk documents and initialize retrieval system
#     app.state.chunks = chunk_documents(documents)
#     logger.info(f"Total chunks created: {len(app.state.chunks)}")

#     # Create persistent retrieval and add chunk documents
#     app.state.retrieval = RetrievalSystem()
#     docs_for_retrieval = [
#         {
#             "id": c["id"],
#             "title": c.get("title", ""),
#             "content": c.get("content", ""),
#             "source": c.get("source", ""),
#             "updated_at": c.get("updated_at", ""),
#             "url": c.get("url", "")
#         } for c in app.state.chunks
#     ]
#     app.state.retrieval.add_documents(docs_for_retrieval)
#     logger.info("Retrieval system initialized")

# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("Shutting down")
#     if hasattr(app.state, "selenium_driver") and app.state.selenium_driver:
#         app.state.selenium_driver.quit()
#         logger.info("Selenium driver closed")
#     if app.state.mysql:
#         await app.state.mysql.close()
#     if getattr(app.state, "redis", None):
#         await app.state.redis.close()
#     logger.info("Shutdown complete")

# def is_resource_url(url: str) -> bool:
#     p = urlparse(url)
#     path = p.path.lower()
#     for bad in ('.pdf', '.jpg', '.jpeg', '.png', '.zip', '.mp3', '.mp4', '.svg', '.woff', '.ttf'):
#         if path.endswith(bad):
#             return True
#     return False

# async def fetch_url(session, url, sem, selenium_driver=None):
#     async with sem:
#         try:
#             # Use Selenium for dynamic pages
#             if selenium_driver and is_dynamic_page(url):
#                 html = await scrape_with_selenium(selenium_driver, url)
#                 if html:
#                     return html, url
#                 # Fall back to regular fetch if Selenium fails
                
#             # Regular fetch for static pages
#             async with session.get(url, allow_redirects=True) as resp:
#                 if resp.status != 200:
#                     return None, url
#                 ct = resp.headers.get("Content-Type", "")
#                 if not ct.startswith("text/html"):
#                     return None, url
#                 text = await resp.text()
#                 return text, url
#         except Exception as e:
#             logger.debug(f"fetch {url} failed: {e}")
#             return None, url

# def is_dynamic_page(url: str) -> bool:
#     """Heuristic to determine if a page likely has dynamic content"""
#     dynamic_paths = ['/search', '/tours', '/hotels', '/events', '/booking']
#     return any(path in url for path in dynamic_paths)

# def extract_page_content(html: str, url: str) -> dict:
#     """Extract structured content from HTML"""
#     soup = BeautifulSoup(html, "html.parser")
    
#     # Extract title
#     title = ""
#     if soup.title:
#         title = soup.title.text.strip()
    
#     # Extract main content - prioritize semantic HTML5 elements
#     main_content = soup.find(['main', 'article', 'div.article', 'div.content'])
#     if not main_content:
#         main_content = soup.body
    
#     # Remove unwanted elements
#     for element in main_content(['script', 'style','footer', 'iframe', 'noscript']):
#         element.decompose()
    
#     # Extract text from important elements
#     content_elements = []
#     for el in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article', 'section', 'div','form','nav']):
#         text = el.get_text(" ", strip=True)
#         if text and len(text.split()) > 3:
#             content_elements.append(text)
    
#     content = " ".join(content_elements)
    
#     return {
#         "title": title,
#         "content": content,
#         "url": url
#     }

# async def scrape_site(base_url: str, selenium_driver=None) -> List[dict]:
#     """
#     Enhanced crawler with:
#     - Selenium for dynamic pages
#     - Better content extraction
#     - URL normalization
#     - Polite crawling
#     """
#     headers = {"User-Agent": "Mozilla/5.0"}
#     parsed_base = urlparse(base_url)
#     base_netloc = parsed_base.netloc.replace("www.", "")
#     scraped = set()
#     content = []
#     queue = [(base_url, 0)]
#     sem = asyncio.Semaphore(CRAWL_CONCURRENCY)
#     session = aiohttp.ClientSession(headers=headers, timeout=TIMEOUT)
    
#     try:
#         while queue and len(scraped) < MAX_PAGES:
#             current_url, depth = queue.pop(0)
#             current_url = urldefrag(current_url)[0]
            
#             # Skip if already scraped or invalid
#             if current_url in scraped:
#                 continue
#             if depth > MAX_DEPTH:
#                 continue
#             if is_resource_url(current_url):
#                 continue
            
#             # Fetch page content
#             html, fetched_url = await fetch_url(session, current_url, sem, selenium_driver)
#             if not html:
#                 continue
            
#             # Extract structured content
#             page_data = extract_page_content(html, current_url)
#             if page_data.get("content"):
#                 content.append(page_data)
#                 scraped.add(current_url)
            
#             # Add links to queue if we have depth remaining
#             if depth < MAX_DEPTH:
#                 soup = BeautifulSoup(html, "html.parser")
#                 for a in soup.find_all("a", href=True):
#                     href = a['href']
#                     full = urljoin(base_url, href)
#                     full = urldefrag(full)[0]
#                     parsed = urlparse(full)
                    
#                     # Only follow same-domain links
#                     if parsed.netloc.replace("www.", "") == base_netloc:
#                         if not is_resource_url(full) and full not in scraped:
#                             queue.append((full, depth+1))
        
#         await session.close()
#     except Exception as e:
#         logger.error(f"Error in scrape_site: {e}")
#         await session.close()
    
#     logger.info(f"Scraped {len(scraped)} pages from {base_url}")
#     return content

# @app.post("/ask")
# async def ask_question(request: QueryRequest):
#     start_time = datetime.now()
#     query = request.question.strip()
#     if not query:
#         raise HTTPException(status_code=400, detail="Question cannot be empty")

#     try:
#         # Redis cache
#         if getattr(app.state, "redis", None):
#             cache_key = f"qa:{query}"
#             cached = await app.state.redis.get(cache_key)
#             if cached:
#                 logger.info("Returning cached answer")
#                 return {"answer": cached, "cached": True, "processing_time": 0.0}

#         # 1) DB search
#         db_results = await query_db_content(app.state.mysql.pool, query, app.state.redis)
#         logger.info(f"DB search returned {len(db_results)} documents")

#         # 2) Retrieval search
#         relevant_chunks = await hybrid_retrieval(query, app.state.retrieval, k=settings.retrieval_k)
#         logger.info(f"Retrieval returned {len(relevant_chunks)} chunks")

#         # Build context parts
#         context_parts = []
#         for doc in db_results:
#             content_text = doc.get('content', '') or ""
#             if content_text.strip():
#                 context_parts.append(f"[Database] {doc.get('title','')}\n{content_text}")

#         for doc in relevant_chunks:
#             content_text = doc.get('content', '') or ""
#             if content_text.strip():
#                 src = doc.get('source', 'web')
#                 url = doc.get('url', '')
#                 context_parts.append(f"[{src.upper()}] {doc.get('title','')}\nURL: {url}\n{content_text}")

#         # Fallback if no context
#         if not context_parts:
#             processing_time = (datetime.now() - start_time).total_seconds()
#             logger.info(f"No context for query '{query}', returning fallback.")
#             return {
#                 "answer": f"I couldn't find specific details in the provided data. Please visit {settings.target_url} for more information.",
#                 "processing_time": processing_time
#             }

#         # Compose context
#         context = "\n\n---\n\n".join(context_parts)
#         if len(context) > CONTEXT_CHAR_LIMIT:
#             context = context[:CONTEXT_CHAR_LIMIT]

#         # System message to prevent hallucination
#         system_message = {
#             "role": "system",
#             "content": (
#                 "You are a strict assistant for Ethiopia Tourism. ANSWER ONLY using the provided CONTEXT. "
#                 "If asked about tours, hotels, or attractions, provide specific details from the context. "
#                 "For pricing or availability questions, direct users to the website for the most current information. "
#                 "If the context does not contain enough information, respond EXACTLY with: "
#                 f"'I couldn't find specific details in the provided data. Please visit {settings.target_url} for more information.'"
#             )
#         }

#         user_prompt = (
#             "Context:\n{context}\n\n"
#             "Question: {query}\n\n"
#             "Instructions:\n"
#             "- Be specific and factual\n"
#             "- Include URLs when available\n"
#             "- If unsure, use the exact fallback phrase\n"
#         ).format(context=context, query=query)

#         # Call LLM
#         client = AsyncOpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)
#         response = await client.chat.completions.create(
#             model="mistralai/mistral-small-3.2-24b-instruct:free",
#             messages=[system_message, {"role": "user", "content": user_prompt}],
#             max_tokens=500,
#             temperature=0.3  # Lower temperature for more deterministic answers
#         )

#         # Extract answer
#         answer = response.choices[0].message.content.strip()

#         # Safety checks
#         unsafe_markers = ["i think", "i believe", "as an ai", "as far as i know", "i'm not sure"]
#         if any(m in answer.lower() for m in unsafe_markers):
#             logger.warning(f"LLM output contained unsafe markers; returning fallback. Raw: {answer[:200]}")
#             answer = f"I couldn't find specific details in the provided data. Please visit {settings.target_url} for more information."

#         # Cache answer
#         if getattr(app.state, "redis", None):
#             await app.state.redis.setex(f"qa:{query}", settings.redis_cache_ttl, answer)

#         processing_time = (datetime.now() - start_time).total_seconds()
        
#         # Include sources for provenance
#         sources = []
#         for doc in (db_results or [])[:3]:
#             sources.append({"type": "db", "title": doc.get("title", ""), "id": doc.get("id", "")})
#         for doc in (relevant_chunks or [])[:3]:
#             sources.append({
#                 "type": doc.get("source", "web"), 
#                 "title": doc.get("title", ""), 
#                 "id": doc.get("id", ""),
#                 "url": doc.get("url", "")
#             })

#         logger.info(f"Processed query '{query[:50]}' in {processing_time:.2f}s")
#         return {
#             "answer": answer,
#             "processing_time": processing_time,
#             "sources": sources
#         }

#     except Exception as e:
#         logger.exception(f"Error processing question: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=8000,reload=True)