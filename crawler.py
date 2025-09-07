import asyncio
import aiohttp
import logging
import random
import socket
from urllib.parse import urlparse, urljoin, urlunparse
from typing import Set, Deque, Dict, Any, Optional, Callable
from collections import deque
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from urllib.robotparser import RobotFileParser
from xml.etree.ElementTree import parse
from io import StringIO
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from setting import settings

logger = logging.getLogger(__name__)

# Global crawler status for monitoring
crawler_status = {
    "is_running": False,
    "crawled_count": 0,
    "queue_size": 0,
    "last_crawled_url": "",
    "start_time": None,
    "last_error": "",
    "failed_urls": 0,
    "success_rate": 0.0,
    "failed_urls_list": []
}

# Define tracking and sorting parameters to filter out
TRACKING_PARAMS = {
    'set_currency', 'orderby', 'utm_', 'fbclid', 'gclid',
    'ref', 'source', 'mc_cid', 'mc_eid', 'msclkid', 'yclid', 'igshid'
}

# Define URL patterns that should never be crawled
EXCLUDED_PATHS = {
    '/forgot-password', '/profile', '/admin' '/account'
}

# Crawling constants
INFINITE_SCROLL_STEPS = 25  
INFINITE_SCROLL_DELAY_SEC = 3.5 
SELENIUM_TIMEOUT = getattr(settings, "scrape_timeout", 1000) 
MIN_CONTENT_LENGTH = getattr(settings, "min_content_length", 10)

# Rotate user agents to avoid blocks
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0"
]

class ProfessionalCrawler:
    def __init__(self, base_url: str, concurrency: int = 10, delay: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.base_domain = urlparse(base_url).netloc
        self.concurrency = concurrency
        self.delay = delay
        self.visited: Set[str] = set()
        self.queue: Deque[str] = deque()
        self.failed_urls: Deque[str] = deque()
        self.robots_parser = RobotFileParser()
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_retries = getattr(settings, "max_retries", 4)

    async def check_network(self, url: str) -> bool:
        """Check network connectivity to the host"""
        parsed = urlparse(url)
        try:
            socket.gethostbyname(parsed.netloc)
            return True
        except socket.gaierror:
            logger.error(f"Cannot resolve host {parsed.netloc}")
            return False

    async def initialize(self):
        """Initialize the crawler session, robots.txt, and sitemap parsing"""
        global crawler_status
        crawler_status["is_running"] = True
        crawler_status["start_time"] = datetime.now().isoformat()
        crawler_status["failed_urls"] = 0
        crawler_status["failed_urls_list"] = []

        proxy = getattr(settings, "proxy", None)
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': random.choice(USER_AGENTS)},
            timeout=aiohttp.ClientTimeout(total=100),
            connector=aiohttp.TCPConnector(ssl=False) if proxy else None
        )

        try:
            robots_url = f"{self.base_url}/robots.txt"
            if await self.check_network(robots_url):
                async with self.session.get(robots_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        self.robots_parser.parse(content.splitlines())
                        logger.info(f"Successfully parsed robots.txt for {self.base_domain}")
                        
                        sitemaps = re.findall(r'^Sitemap:\s*(.+)', content, re.MULTILINE | re.IGNORECASE)
                        if sitemaps:
                            await self._parse_sitemaps(sitemaps)
                        else:
                            logger.info("No sitemaps found in robots.txt, trying common sitemap locations")
                            await self._try_common_sitemaps()
                    else:
                        logger.warning(f"Failed to fetch robots.txt: {response.status}")
                        self.robots_parser.parse([])
                        await self._try_common_sitemaps()
            else:
                logger.warning("Network check failed for robots.txt, trying sitemap")
                self.robots_parser.parse([])
                await self._try_common_sitemaps()
        except Exception as e:
            logger.error(f"Failed to parse robots.txt: {str(e)}")
            crawler_status["last_error"] = str(e)
            self.robots_parser.parse([])
            await self._try_common_sitemaps()

        if not self.queue:
            self.queue.append(self.base_url)
            common_paths = [
                '/tour', '/event', '/attractions', '/events', '/register', '/contact', '/boat',
                '/space', '/hotel',  '/tour/list',
                '/destination', '/locations','/car','/page'
            ]
            for path in common_paths:
                full_url = urljoin(self.base_url, path)
                if self.should_crawl(full_url) and full_url not in self.visited and full_url not in self.queue:
                    self.queue.append(full_url)
            crawler_status["queue_size"] = len(self.queue)

        logger.info(f"Initialized crawler with {len(self.queue)} URLs in queue")

    async def _try_common_sitemaps(self):
        """Try common sitemap locations"""
        common_sitemaps = [
            f"{self.base_url}/sitemap.xml",
            f"{self.base_url}/sitemap_index.xml",
            f"{self.base_url}/sitemap",
            f"{self.base_url}/sitemap.txt"
        ]
        for sitemap_url in common_sitemaps:
            try:
                if await self.check_network(sitemap_url):
                    async with self.session.get(sitemap_url, timeout=100) as response:
                        if response.status == 200:
                            content = await response.text()
                            await self._parse_sitemap(content, sitemap_url)
                            logger.info(f"Successfully parsed sitemap {sitemap_url}")
                        else:
                            logger.debug(f"No sitemap found at {sitemap_url}: {response.status}")
            except Exception as e:
                logger.debug(f"Failed to fetch sitemap {sitemap_url}: {str(e)}")

    async def _parse_sitemap(self, content: str, sitemap_url: str):
        """Parse a single sitemap (XML or text-based)"""
        try:
            tree = parse(StringIO(content))
            for url_elem in tree.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                url = url_elem.text.strip()
                if self.should_crawl(url):
                    normalized = self.normalize_url(url)
                    if normalized not in self.visited and normalized not in self.queue:
                        self.queue.append(normalized)
            logger.info(f"Added {len(self.queue)} URLs from sitemap {sitemap_url}")
            crawler_status["queue_size"] = len(self.queue)
        except Exception:
            urls = [line.strip() for line in content.splitlines() if line.strip().startswith('http')]
            for url in urls:
                if self.should_crawl(url):
                    normalized = self.normalize_url(url)
                    if normalized not in self.visited and normalized not in self.queue:
                        self.queue.append(normalized)
            logger.info(f"Added {len(self.queue)} URLs from text sitemap {sitemap_url}")
            crawler_status["queue_size"] = len(self.queue)

    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and sorting query parameters"""
        try:
            parsed = urlparse(url)
            parsed = parsed._replace(fragment="")
            query_parts = parsed.query.split('&') if parsed.query else []
            filtered_params = sorted([param for param in query_parts if not any(param.startswith(prefix) for prefix in TRACKING_PARAMS)])
            parsed = parsed._replace(query='&'.join(filtered_params) if filtered_params else '')
            normalized = urlunparse(parsed)
            return normalized if urlparse(normalized).netloc == self.base_domain else url
        except Exception as e:
            logger.error(f"URL normalization failed for {url}: {str(e)}")
            return url

    def should_crawl(self, url: str) -> bool:
        """Check if URL should be crawled based on domain, robots.txt, and content type"""
        try:
            if not url or not url.startswith(('http://', 'https://')):
                return False
            parsed = urlparse(url)
            if parsed.netloc != self.base_domain:
                return False
            if not self.robots_parser.can_fetch('*', url):
                logger.debug(f"URL {url} blocked by robots.txt")
                return False
            if any(parsed.path.startswith(prefix) for prefix in EXCLUDED_PATHS):
                return False
            excluded_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.zip', '.rar', '.exe', '.tar', '.gz', '.xls'}
            if any(parsed.path.lower().endswith(ext) for ext in excluded_extensions):
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking crawl eligibility for {url}: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_static(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse static content (HTML)"""
        try:
            if await self.check_network(url):
                async with self.session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Static fetch failed for {url}: {response.status}")
                        crawler_status["failed_urls"] += 1
                        crawler_status["failed_urls_list"].append(f"{url} (Status: {response.status})")
                        return None
                    html = await response.text()
                    return self._extract_html(html, url)
            else:
                logger.warning(f"Network check failed for {url}")
                crawler_status["failed_urls"] += 1
                crawler_status["failed_urls_list"].append(f"{url} (Network check failed)")
                return None
        except Exception as e:
            logger.error(f"Static fetch failed for {url}: {str(e)}")
            crawler_status["failed_urls"] += 1
            crawler_status["failed_urls_list"].append(f"{url} (Error: {str(e)})")
            crawler_status["last_error"] = str(e)
            return None

    def _extract_html(self, html: str, url: str, is_dynamic: bool = False) -> Optional[Dict[str, Any]]:
        """Extract text, metadata, and links from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for element in soup(['script', 'style', 'footer', 'aside']):
                element.decompose()

            title = soup.find('title').get_text().strip() if soup.find('title') else "No Title"
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_content = meta_desc['content'].strip() if meta_desc else ''

            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_=re.compile(r'content|main|body|tour-list|location-details', re.I)) or
                soup.body
            )
            text = main_content.get_text(' ', strip=True) if main_content else soup.get_text(' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()

            structured_data = []
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    structured_data.append(json.loads(script.string))
                except json.JSONDecodeError:
                    continue

            if len(text) < MIN_CONTENT_LENGTH and not meta_content and not structured_data:
                logger.warning(f"Insufficient content for {url} (length: {len(text)})")
                return None

            links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if self.should_crawl(full_url):
                    links.add(self.normalize_url(full_url))

            return {
                'url': url,
                'title': title,
                'content': f"{meta_content} {text}",
                'links': list(links),
                'is_dynamic': is_dynamic,
                'metadata': {
                    'meta_description': meta_content,
                    'structured_data': structured_data,
                    'crawled_at': datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"HTML extraction failed for {url}: {str(e)}")
            crawler_status["last_error"] = str(e)
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_dynamic(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch dynamic content using Selenium with infinite scroll handling"""
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
        options.add_argument("--disable-blink-features=AutomationControlled")  
        if hasattr(settings, "proxy") and settings.proxy:
            options.add_argument(f"--proxy-server={settings.proxy}")
        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(SELENIUM_TIMEOUT)
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            last_height = driver.execute_script("return document.body.scrollHeight")
            for _ in range(INFINITE_SCROLL_STEPS):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(INFINITE_SCROLL_DELAY_SEC)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "main, article, div.content, div.main, div.tour-list, div.location-details"))
                )
            except Exception:
                logger.debug(f"No main content element found for {url}, capturing available content")

            html = driver.page_source
            return self._extract_html(html, url, is_dynamic=True)
        except Exception as e:
            logger.error(f"Dynamic fetch failed for {url}: {str(e)}")
            crawler_status["failed_urls"] += 1
            crawler_status["failed_urls_list"].append(f"{url} (Dynamic fetch error: {str(e)})")
            crawler_status["last_error"] = str(e)
            return None
        finally:
            if driver:
                driver.quit()

    async def crawl_page(self, url: str, on_page_callback: Callable) -> bool:
        """Crawl a single page with fallback logic"""
        global crawler_status
        normalized_url = self.normalize_url(url)
        if normalized_url in self.visited:
            return False

        self.visited.add(normalized_url)
        crawler_status["last_crawled_url"] = normalized_url

        page_data = await self.fetch_static(normalized_url)

        if not page_data or len(page_data.get('content', '')) < MIN_CONTENT_LENGTH:
            logger.debug(f"Static fetch insufficient for {url}, trying dynamic")
            page_data = await asyncio.get_event_loop().run_in_executor(
                None, self.fetch_dynamic, normalized_url
            )

        if page_data and page_data.get('content'):
            await on_page_callback(page_data)
            for link in page_data.get('links', []):
                n_url = self.normalize_url(link)
                if n_url not in self.visited and n_url not in self.queue:
                    self.queue.append(n_url)
            
            crawler_status["crawled_count"] += 1
            crawler_status["queue_size"] = len(self.queue)
            crawler_status["success_rate"] = (crawler_status["crawled_count"] / 
                                            (crawler_status["crawled_count"] + crawler_status["failed_urls"]) * 100) if (crawler_status["crawled_count"] + crawler_status["failed_urls"]) > 0 else 0
            logger.info(f"Successfully crawled {url}")
            return True

        self.failed_urls.append(normalized_url)
        crawler_status["failed_urls"] += 1
        crawler_status["success_rate"] = (crawler_status["crawled_count"] / 
                                         (crawler_status["crawled_count"] + crawler_status["failed_urls"]) * 100) if (crawler_status["crawled_count"] + crawler_status["failed_urls"]) > 0 else 0
        return False

    async def crawl(self, on_page_callback: Callable, max_pages: int = 50000):
        """Main crawl method"""
        global crawler_status
        await self.initialize()
        crawled_count = 0
        last_request_time = 0.0
        retry_attempts = 0
        max_retry_attempts = 3

        while self.queue and crawled_count < max_pages:
            current_url = self.queue.popleft()
            current_time = time.time()
            if current_time - last_request_time < self.delay:
                await asyncio.sleep(self.delay - (current_time - last_request_time))

            async with self.semaphore:
                success = await self.crawl_page(current_url, on_page_callback)
                if success:
                    crawled_count += 1
                    last_request_time = time.time()
                    if crawled_count % 10 == 0:
                        logger.info(f"Crawled {crawled_count} pages. Queue: {len(self.queue)}, Success rate: {crawler_status['success_rate']:.2f}%")
                else:
                    logger.warning(f"Failed to crawl {current_url}")

            if not self.queue and crawled_count < max_pages and retry_attempts < max_retry_attempts:
                logger.warning(f"Queue empty before reaching max_pages. Retrying failed URLs ({len(self.failed_urls)}) and common paths.")
                retry_attempts += 1
                self.queue.extend(self.failed_urls)
                self.failed_urls.clear()
                common_paths = [
                    '/tour', '/tours', '/attractions', '/events', '/register', '/contact', '/about',
                    '/location/the-marvel-east-and-bale-mountains', '/hotels', '/news', '/tour/list',
                    '/destination', '/locations'
                ]
                for path in common_paths:
                    full_url = urljoin(self.base_url, path)
                    if self.should_crawl(full_url) and full_url not in self.visited and full_url not in self.queue:
                        self.queue.append(full_url)
                await self._try_common_sitemaps()
                crawler_status["queue_size"] = len(self.queue)
                logger.info(f"Reseeded queue with {len(self.queue)} URLs")

        if self.failed_urls and crawled_count < max_pages:
            logger.info(f"Final retry for {len(self.failed_urls)} failed URLs")
            self.queue.extend(self.failed_urls)
            self.failed_urls.clear()
            while self.queue and crawled_count < max_pages:
                current_url = self.queue.popleft()
                async with self.semaphore:
                    success = await self.crawl_page(current_url, on_page_callback)
                    if success:
                        crawled_count += 1
                        last_request_time = time.time()
                        if crawled_count % 10 == 0:
                            logger.info(f"Crawled {crawled_count} pages. Queue: {len(self.queue)}, Success rate: {crawler_status['success_rate']:.2f}%")
                    else:
                        logger.warning(f"Final retry failed for {current_url}")

        crawler_status["is_running"] = False
        logger.info(f"Crawling completed. Total pages: {crawled_count}, Failed: {crawler_status['failed_urls']}, Success rate: {crawler_status['success_rate']:.2f}%")
        if crawler_status["failed_urls_list"]:
            with open("failed_urls.txt", "w") as f:
                f.write("\n".join(crawler_status["failed_urls_list"]))
            logger.info(f"Saved {len(crawler_status['failed_urls_list'])} failed URLs to failed_urls.txt")
        if self.session:
            await self.session.close()

async def run_crawler(base_url: str, on_page_callback: Callable, max_pages: int = 50000):
    """Run the professional crawler"""
    global crawler_status
    crawler_status = {
        "is_running": True,
        "crawled_count": 0,
        "queue_size": 0,
        "last_crawled_url": "",
        "start_time": datetime.now().isoformat(),
        "last_error": "",
        "failed_urls": 0,
        "success_rate": 0.0,
        "failed_urls_list": []
    }
    crawler = ProfessionalCrawler(
        base_url=base_url,
        concurrency=getattr(settings, "crawl_concurrency", 5),  
        delay=getattr(settings, "request_delay", 3.0)  
    )
    await crawler.crawl(on_page_callback, max_pages)