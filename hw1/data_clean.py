
import re
import threading
from typing import Optional, Set

import jieba
from datasketch import MinHash, MinHashLSH


# ----------------------------------------------------------------------
# 1. Bad-phrase filter (compiled once)
# ----------------------------------------------------------------------
BAD_PHRASES = re.compile(
    r'lorem ipsum|terms of use|privacy policy|cookie policy|'
    r'uses? cookies|use of cookies|'
    r'使用条款|隐私政策|cookie ?的使用|'
    r'[\[\]{}]',                     # 常见占位符
    re.IGNORECASE
)

# ----------------------------------------------------------------------
# 2. URL / Email / Punctuation cleaner (compiled once)
# ----------------------------------------------------------------------
URL_EMAIL_RE = re.compile(
    r'https?://\S+|www\.\S+|'                     # URL
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'  # Email
)
PUNCT_RE = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fff\s.,!?;:\'"()—–-]')
WHITESPACE_RE = re.compile(r'\s{2,}')


# ----------------------------------------------------------------------
# 3. LSH deduplication (thread-safe singleton)
# ----------------------------------------------------------------------
class _DedupLSH:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, threshold: float = 0.8, num_perm: int = 128):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = MinHashLSH(threshold=threshold, num_perm=num_perm)
        return cls._instance

def get_lsh(threshold: float = 0.8, num_perm: int = 128) -> MinHashLSH:
    """Thread-safe LSH singleton."""
    return _DedupLSH(threshold, num_perm)


# ----------------------------------------------------------------------
# 4. Core pipeline functions
# ----------------------------------------------------------------------
def rule_based_filter(text: str) -> Optional[str]:
    """
    Step 1: Remove boilerplate / legal text.
    Returns cleaned text or ``None`` if filtered out.
    """
    if not text:
        return None

    stripped = text.strip()
    if len(stripped) < 100:
        return None

    if BAD_PHRASES.search(stripped):
        return None

    # Normalize whitespace early
    cleaned = WHITESPACE_RE.sub(' ', stripped)
    return cleaned


def regex_filter(text: str) -> Optional[str]:
    """
    Step 2: Strip URL, email, and unwanted punctuation.
    """
    if not text:
        return None

    # Remove URLs & emails
    text = URL_EMAIL_RE.sub('', text)

    # Keep only allowed characters
    text = PUNCT_RE.sub(' ', text)

    # Collapse whitespace
    text = WHITESPACE_RE.sub(' ', text).strip()
    return text if text else None


def _make_shingles(tokens: list[str], n: int = 5) -> Set[str]:
    """Generate n-gram shingles from token list."""
    if len(tokens) < n:
        return set()
    return {' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def minhash_dedup(
    text: str,
    doc_id: str,
    *,
    lang: str = 'en',
    min_tokens: int = 10,
    shingle_n: int = 5,
    lsh_threshold: float = 0.8,
    lsh_num_perm: int = 128
) -> Optional[str]:
    """
    Step 3: Near-duplicate detection using MinHash + LSH.
    Returns original (cleaned) text if not duplicate, else ``None``.
    """
    if not text:
        return None

    # Tokenize according to language
    if lang == 'zh':
        tokens = jieba.lcut(text, cut_all=False)
    else:
        tokens = text.split()

    if len(tokens) < min_tokens:
        return None

    shingles = _make_shingles(tokens, n=shingle_n)
    if not shingles:
        return None

    # Build MinHash
    m = MinHash(num_perm=lsh_num_perm)
    for s in shingles:
        m.update(s.encode('utf8'))

    # Thread-safe LSH query/insert
    lsh = get_lsh(threshold=lsh_threshold, num_perm=lsh_num_perm)
    with threading.Lock():          # datasketch is not thread-safe by default
        if lsh.query(m):
            return None
        lsh.insert(doc_id, m)

    return text


# ----------------------------------------------------------------------
# 5. One-stop pipeline
# ----------------------------------------------------------------------
def filter_and_dedup_pipeline(
    raw_text: str,
    doc_id: str,
    *,
    lang: str = 'en',
    min_length: int = 100,
    **dedup_kwargs
) -> Optional[str]:
    """
    Full pipeline:
      1. rule_based_filter
      2. regex_filter
      3. minhash_dedup
    """
    text = rule_based_filter(raw_text)
    if text is None:
        return None

    if len(text) < min_length:
        return None

    text = regex_filter(text)
    if text is None:
        return None

    return minhash_dedup(text, doc_id, lang=lang, **dedup_kwargs)


# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from typing import Dict

# 假设前面的 filter_and_dedup_pipeline 已定义在当前文件中
# from your_module import filter_and_dedup_pipeline  # 如需模块化可取消注释

# ----------------------------------------------------------------------
# 配置路径
# ----------------------------------------------------------------------
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

LANG_CONFIG = {
    "en": {
        "raw": RAW_DIR / "en_wiki" / "en_wikipedia.jsonl",
        "clean": CLEAN_DIR / "en_wiki" / "en_wikipedia_clean.jsonl"
    },
    "zh": {
        "raw": RAW_DIR / "zh_wiki" / "zh_wikipedia.jsonl",
        "clean": CLEAN_DIR / "zh_wiki" / "zh_wikipedia_clean.jsonl"
    }
}

# 确保输出目录存在
for lang in LANG_CONFIG:
    CLEAN_DIR.joinpath(f"{lang}_wiki").mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 主处理流程
# ----------------------------------------------------------------------
def process_wiki_file(lang: str):
    config = LANG_CONFIG[lang]
    input_path = config["raw"]
    output_path = config["clean"]

    if not input_path.exists():
        print(f"[WARN] 输入文件不存在: {input_path}")
        return

    print(f"[START] 处理 {lang.upper()} 数据: {input_path}")

    total, kept, filtered_out = 0, 0, 0
    seen_doc_ids = set()  # 防止同一文件内重复 doc_id

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON 解析失败 (行 {line_num}): {e}")
                continue

            # 必填字段检查
            title = doc.get("title", "").strip()
            content = doc.get("content", "").strip()
            url = doc.get("url", "").strip()

            if not title or not content:
                filtered_out += 1
                continue

            total += 1

            # 生成唯一 doc_id（url 去重 + 防冲突）
            doc_id = f"{lang}_{hash(url) & 0xFFFFFFFF}"  # 32-bit hash 避免过长
            if doc_id in seen_doc_ids:
                filtered_out += 1
                continue
            seen_doc_ids.add(doc_id)

            # 核心清洗 + 去重
            cleaned_content = filter_and_dedup_pipeline(
                raw_text=content,
                doc_id=doc_id,
                lang=lang,
                min_length=120,           # 可根据需求调整
                lsh_threshold=0.8,
                shingle_n=5
            )

            if cleaned_content is None:
                filtered_out += 1
                continue

            # 写入清洗后的数据（保留 title 和 url）
            cleaned_doc = {
                "title": title,
                "content": cleaned_content,
                "url": url
            }
            fout.write(json.dumps(cleaned_doc, ensure_ascii=False) + "\n")
            kept += 1

            if total % 10000 == 0:
                print(f"  [PROGRESS] 已处理 {total} 条，保留 {kept} 条")

    print(f"[DONE] {lang.upper()} 处理完成")
    print(f"   总计: {total}，保留: {kept}，过滤: {filtered_out}")
    print(f"   输出: {output_path}\n")


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 可按需开启语言
    for lang in ["en"]:
        process_wiki_file(lang)