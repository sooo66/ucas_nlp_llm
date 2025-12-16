import os
import time
import json
import random
import toml
import tomllib
from typing import Dict, Any, List, Set
from tqdm import tqdm
from loguru import logger
import requests
import re
import nltk
import jieba
from nltk.tokenize import word_tokenize
import concurrent.futures
from bs4 import BeautifulSoup

# ======================================================
# BaseCrawler
# ======================================================
class BaseCrawler:
    """所有爬虫的基础类，提供通用 HTTP 和重试逻辑"""

    def __init__(self, site_config: Dict[str, Any], defaults: Dict[str, Any]):
        self.site = site_config
        self.defaults = defaults

        # User-Agent
        ua_from_site = site_config.get("user_agent")
        ua_from_defaults = defaults.get("user_agent")
        self.user_agent = ua_from_site if ua_from_site is not None else ua_from_defaults

        self.session = requests.Session()
        if self.user_agent:
            self.session.headers.update({"User-Agent": str(self.user_agent)})

        # 控制参数
        self.requests_per_minute = defaults.get("requests_per_minute", 60)
        self.delay = 60.0 / self.requests_per_minute
        self.retry_max = defaults.get("retry_max", 3)
        self.retry_backoff = defaults.get("retry_backoff", 1.5)
        self.max_file_size_mb = defaults.get("max_file_size_mb", 50)

    def _get(self, url: str, params: Dict[str, Any] | None) -> Any:
        """GET 请求，带重试"""
        for attempt in range(1, self.retry_max + 1):
            try:
                r = self.session.get(url, params=params, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.warning(f"GET failed ({attempt}/{self.retry_max}): {e}")
                time.sleep(self.retry_backoff ** attempt)
        return None

    def crawl(self):
        raise NotImplementedError("Subclasses must implement crawl()")


class MediaWikiCrawler:
    """Wikipedia / MediaWiki 通用爬虫"""
    
    def __init__(self, site_config, defaults):
        self.site = site_config
        self.defaults = defaults
        self.api_url = site_config["entrypoints"][0]
        self.api_conf = site_config["api"]
        self.lang = site_config["lang"]

        # 设置 User-Agent 和其他请求参数
        ua_from_site = site_config.get("user_agent")
        ua_from_defaults = defaults.get("user_agent")
        self.user_agent = ua_from_site if ua_from_site is not None else ua_from_defaults
        self.session = requests.Session()
        if self.user_agent:
            self.session.headers.update({"User-Agent": str(self.user_agent)})
        
        # 控制参数
        self.requests_per_minute = defaults.get("requests_per_minute", 60)
        self.delay = 60.0 / self.requests_per_minute
        self.retry_max = defaults.get("retry_max", 3)
        self.retry_backoff = defaults.get("retry_backoff", 1.5)

        # 输出路径设置
        self.output_dir = site_config.get("output", {}).get("dir", defaults.get("output_root", "data/raw"))
        self.output_prefix = f"{self.output_dir}/{self.site['name']}"
        
        # 已爬取标题集合
        self.crawled_titles: Set[str] = set()
        self._load_crawled_titles()

    def _load_crawled_titles(self):
        """从已有文件中加载已爬取的标题"""
        fname = f"{self.output_prefix}.jsonl"
        if os.path.exists(fname):
            logger.info(f"Loading existing titles from {fname}")
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        title = item.get("title", "")
                        if title:
                            self.crawled_titles.add(title)
                    except Exception as e:
                        logger.warning(f"Failed to parse line: {e}")
            logger.info(f"Loaded {len(self.crawled_titles)} existing titles")

    def _get(self, url, params):
        """GET 请求，带重试"""
        for attempt in range(1, self.retry_max + 1):
            try:
                r = self.session.get(url, params=params, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.warning(f"GET failed ({attempt}/{self.retry_max}): {e}")
                time.sleep(self.retry_backoff ** attempt)
        return None

    def get_random_titles(self, count=100) -> List[str]:
        """获取随机页面标题，过滤掉已爬取的"""
        params = {"action": "query", "list": "random", "rnlimit": min(count, 500), "rnnamespace": 0, "format": "json"}
        data = self._get(self.api_url, params)
        if not data:
            return []
        
        all_titles = [p["title"] for p in data.get("query", {}).get("random", [])]
        # 过滤掉已爬取的标题
        new_titles = [t for t in all_titles if t not in self.crawled_titles]
        logger.info(f"Got {len(all_titles)} titles, {len(new_titles)} are new")
        return new_titles

    def clean_text(self, text: str) -> str:
        """清理wikitext中的无效部分，包括中文和英文的无效章节"""
        
        # 中文和英文的stop_keywords列表，包含常见的无效章节
        stop_keywords = [
            "参见", "参考文献", "外部链接", "参考资料", "进一步阅读", "外部资源", "附录", "注释", "书目", "链接", "脚注",  # 中文
            "See also", "References", "External links", "Further reading", "Bibliography", "External resources", "Notes"  # 英文
        ]
        
        # 根据 `== title ==` 区分章节，并去除每个标题中的 `==`
        sections = re.split(r"(={2,}.*?={2,})", text)  # 根据 `== title ==` 区分章节
        sections_cleaned = []

        skip_section = False  # 标记是否跳过该章节
        for section in sections:
            if re.match(r"={2,}.*?={2,}", section):  # 这是标题部分
                # 去掉标题中的 `==` 符号
                section = section.strip("= \n")  # 去掉两边的 `==`
                # 检查标题是否包含需要跳过的关键词
                if any(keyword in section for keyword in stop_keywords):
                    skip_section = True
                else:
                    skip_section = False

            if skip_section:
                continue  # 如果是要跳过的章节，直接跳过该部分

            sections_cleaned.append(section)

        # 将所有清理后的章节合并为一个字符串
        cleaned_text = " ".join(sections_cleaned)
        cleaned_text = " ".join(cleaned_text.split())  # 去除多余空格
        return cleaned_text

    def count_existing_words(self) -> int:
        total = 0
        fname = f"{self.output_prefix}.jsonl"
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        total += len(self.tokenize(item.get("content", "")))
                    except Exception:
                        continue
        return total

    def tokenize(self, text: str) -> List[str]:
        """根据语言选择合适的分词工具"""
        if self.lang == 'zh':  # 如果是中文，使用 jieba 分词
            return list(jieba.cut(text))
        else:  # 如果是英文，使用 nltk 的 word_tokenize 分词
            return word_tokenize(text)
        
    def get_page_content(self, title: str) -> Dict[str, Any]:
        params = {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json"}
        data = self._get(self.api_url, params)
        if not data:
            return {}
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            return {"title": p.get("title", ""), "content": p.get("extract", ""), "url": f"https://{self.lang}.wikipedia.org/wiki/{p.get('title')}"}
        return {}

    def crawl(self):
        """主逻辑"""
        total_target = self.api_conf.get("total_words", 5_000_000)
        current = self.count_existing_words()
        remaining = total_target - current
        if remaining <= 0:
            logger.success(f"[{self.lang}] Already reached {total_target:,} words.")
            return

        logger.info(f"[{self.lang}] Current: {current:,} | Target: {total_target:,} | Need: {remaining:,}")
        logger.info(f"[{self.lang}] Already crawled {len(self.crawled_titles)} titles")

        # 使用多线程来加速爬取过程
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            loop_words_written = 0

            # 每次循环获取随机页面标题并并行处理
            while remaining > 0:
                random_titles = self.get_random_titles(100)  # 获取随机页面标题（已过滤）
                
                if not random_titles:
                    logger.warning(f"[{self.lang}] No new titles found, requesting more...")
                    time.sleep(1)
                    continue
                
                logger.info(f"[{self.lang}] Processing {len(random_titles)} new titles")

                # 提交每个页面抓取任务到线程池
                future_to_title = {}
                for title in random_titles:
                    future = executor.submit(self.process_page, title)
                    future_to_title[future] = title

                # 等待所有任务完成并处理结果
                for future in concurrent.futures.as_completed(future_to_title):
                    title = future_to_title[future]
                    try:
                        data = future.result()
                        if data and data.get("content"):
                            content = self.clean_text(data["content"])
                            words = len(self.tokenize(content))
                            remaining -= words
                            loop_words_written += words

                            # 将数据写入文件
                            with open(f"{self.output_prefix}.jsonl", "a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                            
                            # 记录已爬取的标题
                            self.crawled_titles.add(title)
                        else:
                            logger.warning(f"Failed to get valid content for {title}")

                    except Exception as e:
                        logger.error(f"Error processing {title}: {e}")

                logger.info(f"[{self.lang}] This loop added {loop_words_written} words. Total crawled titles: {len(self.crawled_titles)}")

        logger.success(f"[{self.lang}] Done! Added {loop_words_written:,} words. Total now ≥ {total_target:,}")

    def process_page(self, title: str):
        """处理每个页面的请求"""
        return self.get_page_content(title)
    

class HTMLCrawler(BaseCrawler):
    """新华网爬虫（种子扩展 + 正文精准提取）"""

    def __init__(self, site_config: Dict[str, Any], defaults: Dict[str, Any]):
        super().__init__(site_config, defaults)

        self.entrypoints = site_config["entrypoints"]
        self.article_url_patterns = [
            re.compile(p, re.IGNORECASE) for p in site_config["scraping"].get("article_url_patterns", [])
        ]
        self.content_selectors = site_config["scraping"].get("content_selectors", [])
        self.title_selectors = site_config["scraping"].get("title_selectors", [])
        self.exclude_selectors = site_config["scraping"].get("exclude_selectors", [])
        self.output_dir = site_config["output"]["dir"]
        self.lang = site_config.get("lang", "zh")

        # URL 去重 + 链接池
        self.crawled_urls: Set[str] = set()
        self.link_pool: List[str] = []
        self.recent_success: List[str] = []  # 用于扩展
        self._load_crawled_urls()
        self._seed_pool()

    def _load_crawled_urls(self):
        """加载已爬取的 URL"""
        path = f"{self.output_dir}/{self.site['name']}.jsonl"
        if os.path.exists(path):
            logger.info(f"Loading crawled URLs from {path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if url := item.get("url"):
                            self.crawled_urls.add(url)
                    except Exception:
                        continue
            logger.info(f"Loaded {len(self.crawled_urls)} crawled URLs")

    def _seed_pool(self):
        """初始化种子链接池"""
        self.link_pool = [u for u in self.entrypoints if u not in self.crawled_urls]
        random.shuffle(self.link_pool)
        logger.info(f"Seeded {len(self.link_pool)} valid entrypoints")

    def get(self, url: str, **kwargs) -> requests.Response | None:
        """发送 HTTP 请求，重试机制"""
        for attempt in range(1, self.retry_max + 1):
            try:
                time.sleep(self.delay)
                r = self.session.get(url, timeout=15, **kwargs)
                r.raise_for_status()
                return r
            except Exception as e:
                logger.warning(f"GET {url} failed ({attempt}/{self.retry_max}): {e}")
                time.sleep(self.retry_backoff ** attempt)
        return None

    def extract_article_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """从页面中提取符合条件的文章链接"""
        candidates = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#"):
                continue
            full = self._absolute_url(base_url, href)
            if any(p.search(full) for p in self.article_url_patterns):
                if full not in self.crawled_urls and full not in candidates:
                    candidates.append(full)
        return candidates

    def _absolute_url(self, base: str, href: str) -> str:
        """将相对路径转换为绝对路径"""
        if href.startswith("http"):
            return href
        if href.startswith("//"):
            return "https:" + href
        from urllib.parse import urljoin
        return urljoin(base, href)

    def expand_pool(self, from_url: str):
        """扩展链接池"""
        resp = self.get(from_url)
        if not resp:
            return
        soup = BeautifulSoup(resp.text, "html.parser")
        new_links = self.extract_article_links(soup, from_url)
        added = 0
        for link in new_links:
            if link not in self.crawled_urls and link not in self.link_pool:
                self.link_pool.append(link)
                added += 1
        if added:
            logger.info(f"Expanded pool +{added} from {from_url}")

    def get_random_article_links(self, count: int = 50) -> List[str]:
        """随机获取未爬取的文章链接"""
        links = []
        expanded = False
        while len(links) < count and self.link_pool:
            url = self.link_pool.pop(0)
            if url in self.crawled_urls:
                continue
            self.expand_pool(url)  # 扩展链接池
            if url not in self.crawled_urls:
                links.append(url)
            expanded = True
        if expanded and len(self.link_pool) < 100:
            self._try_expand_from_recent()
        return links

    def _try_expand_from_recent(self):
        """根据最近成功的链接进一步扩展链接池"""
        if not self.recent_success:
            return
        for url in random.sample(self.recent_success, min(3, len(self.recent_success))):
            if url not in self.crawled_urls:
                self.expand_pool(url)
                self.crawled_urls.add(url)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """从页面中提取文章标题"""
        for sel in self.title_selectors:
            if tag := soup.select_one(sel):
                return tag.get_text(strip=True)
        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """从页面中提取文章内容"""
        for sel in self.content_selectors:
            if container := soup.select_one(sel):
                self._remove_noise(container)
                return container.get_text(separator=" ", strip=True)
        try:
            from lxml import etree
            tree = etree.HTML(str(soup))
            for xpath in self.site["scraping"].get("content_xpath", []):
                nodes = tree.xpath(xpath)
                if nodes:
                    text = " ".join(node.text_content() for node in nodes if node.text_content())
                    return re.sub(r"\s+", " ", text).strip()
        except Exception as e:
            logger.debug(f"XPath failed: {e}")
        return ""

    def _remove_noise(self, container: BeautifulSoup):
        """去除页面中的噪音内容"""
        for sel in self.exclude_selectors:
            for el in container.select(sel):
                el.decompose()
        for el in container.find_all(string=lambda t: t and any(k in t for k in ["纠错", "责任编辑", "编辑："])):
            el.extract()

    def process_page(self, url: str) -> Dict[str, Any]:
        """处理每一篇文章页面"""
        resp = self.get(url)
        if not resp:
            return {}
        soup = BeautifulSoup(resp.text, "html.parser")

        title = self._extract_title(soup)
        content = self._extract_content(soup)
        if not title or not content:
            logger.debug(f"Skip non-article: {url}")
            return {}

        content = self.clean_text(content)
        data = {"title": title, "content": content, "url": url}

        self.recent_success.append(url)
        if len(self.recent_success) > 50:
            self.recent_success.pop(0)

        self.expand_pool(url)
        return data

    def clean_text(self, text: str) -> str:
        """清理文本中的不必要部分"""
        stop_keywords = [
            "参见", "参考文献", "外部链接", "参考资料", "进一步阅读", "外部资源", "附录", "注释", "书目", "链接", "脚注",
            "See also", "References", "External links", "Further reading", "Bibliography", "External resources", "Notes"
        ]
        sections = re.split(r"(={2,}.*?={2,})", text)
        cleaned = []
        skip = False
        for part in sections:
            if re.match(r"={2,}.*?={2,}", part):
                header = part.strip("= \n")
                skip = any(k in header for k in stop_keywords)
                continue
            if not skip:
                cleaned.append(part)
        return " ".join(" ".join(cleaned).split())

    def count_existing_words(self) -> int:
        """计算已爬取的字数"""
        total = 0
        fname = f"{self.output_dir}/{self.site['name']}.jsonl"
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        total += len(self.tokenize(item.get("content", "")))
                    except Exception:
                        continue
        return total

    def tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        if self.lang == "zh":
            return list(jieba.cut(text))
        return word_tokenize(text)

    def crawl(self):
        """启动爬取过程"""
        total_target = self.site["scraping"].get("total_words", 5_000_000)
        current = self.count_existing_words()
        remaining = total_target - current
        if remaining <= 0:
            logger.success(f"[{self.site.get('lang','?')}] Already reached {total_target:,} words.")
            return

        logger.info(f"[{self.site.get('lang','?')}] Current: {current:,} | Target: {total_target:,} | Need: {remaining:,}")
        logger.info(f"[{self.site.get('lang','?')}] Already crawled {len(self.crawled_urls)} urls")

        os.makedirs(self.output_dir, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            words_this_loop = 0
            while remaining > 0:
                links = self.get_random_article_links(50)
                if not links:
                    logger.warning(f"[{self.site.get('lang','?')}] No new links, retry after 2s...")
                    time.sleep(2)
                    continue

                logger.info(f"[{self.site.get('lang','?')}] Got {len(links)} new links, start fetching...")

                future_to_url = {executor.submit(self.process_page, u): u for u in links}

                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        data = future.result()
                        if data and data.get("content"):
                            words = len(self.tokenize(data["content"]))
                            remaining -= words
                            words_this_loop += words

                            with open(f"{self.output_dir}/{self.site['name']}.jsonl", "a", encoding="utf-8") as f:
                                f.write(json.dumps(data, ensure_ascii=False) + "\n")

                            self.crawled_urls.add(url)
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")

                logger.info(f"[{self.site.get('lang','?')}] Loop added {words_this_loop:,} words "
                            f"(remaining {remaining:,})")

        logger.success(f"[{self.site.get('lang','?')}] Finished! Total words >= {total_target:,}")
# ======================================================
# TOML 加载与入口
# ======================================================
def load_sites(toml_path: str):
    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"TOML not found: {toml_path}")
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
        defaults = cfg.get("defaults", {})
        sites = cfg.get("sites", [])

        return defaults, sites


def main():
    os.makedirs("logs", exist_ok=True)
    defaults, sites = load_sites("sites.toml")

    for site in sites:
        if site["method"] == "mediawiki_api":
            crawler = MediaWikiCrawler(site, defaults)
            crawler.crawl()
        elif site["method"] == "html_scraping":
            crawler = HTMLCrawler(site, defaults)
            crawler.crawl()
        else:
            logger.warning(f"Unsupported method: {site['method']}")


if __name__ == "__main__":
    main()