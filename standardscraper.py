import requests
import logging

# Suppress pypdf warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)
from bs4 import BeautifulSoup, Tag
import asyncio
import aiohttp
import re
from urllib.parse import urlparse
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, model, Function, FunctionType
import tiktoken
from bs4 import NavigableString
import io
import glob
import os
from collections import deque


import datetime


# Collection Names
COLLECTION_WEB = "calsol"
processed_urls_collection = "processed_urls"

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calsol.db")
client = MilvusClient(DB_PATH)
embedding_fn = model.DefaultEmbeddingFunction()  # sentence-transformers/all-MiniLM-L6-v2 - 256 token max
max_tokens = 450

# --- Memory Safety Limits ---
MAX_QUEUE_SIZE = 5000
MAX_HTML_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_CHUNKS_PER_PAGE = 200
MAX_TEXT_PER_PAGE_CHARS = 100_000
# ----------------------------

def scrape(start_urls):
    if isinstance(start_urls, str):
        start_urls = [start_urls]
        
    queue = deque()
    
    # 1. Load pending URLs from DB
    pending_urls = get_pending_urls()
    print(f"Loaded {len(pending_urls)} pending URLs from database.")
    for url in pending_urls:
        queue.append(url) # Pending URLs from DB are assumed to be normalized when inserted
        
    # 2. Add start_urls if not already in DB
    for url in start_urls:
        norm_url = normalize_url(url)
        if not is_url_known(norm_url):
            # Enforce queue limit for start_urls
            if len(queue) >= MAX_QUEUE_SIZE:
                print(f"Queue full ({MAX_QUEUE_SIZE}), dropping start URL: {norm_url}")
                continue
            add_url_to_db(norm_url, status=0) # 0 = Pending
            queue.append(norm_url)
    
    print(f"Starting scrape with {len(queue)} items in queue.")

    while queue:
        url = queue.popleft()
        # url from queue should already be normalized
        
        # Double check status
        if is_url_processed(url):
            print(f"Skipping already processed URL: {url}")
            continue

        if not is_valid_scheme(url):
            mark_url_processed(url) 
            continue

        print(f"Processing: {url}")
        try:
            # Use stream=True to check headers and enforce size limits before full download
            response = requests.get(url, timeout=10, stream=True)
            if response.status_code != 200:
                print(f"Failed to fetch {url}: Status {response.status_code}")
                mark_url_processed(url) 
                continue
            
            # Determine limit based on expected content type (naive check)
            # We'll refine this logic below, but strictly enforce a hard cap for safety.
            limit = MAX_HTML_SIZE_BYTES
            
            # Check Content-Length header if present
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > limit:
                print(f"Skipping {url}: Content-Length {content_length} exceeds limit {limit}")
                mark_url_processed(url)
                continue

            # Safe read with strict bounding
            content_buffer = io.BytesIO()
            downloaded_size = 0
            aborted = False
            
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                if downloaded_size > limit:
                    print(f"Aborting {url}: Download size exceeded limit {limit}")
                    aborted = True
                    break
                content_buffer.write(chunk)
            
            if aborted:
                mark_url_processed(url)
                continue
                
            # Patch the response object with the safely downloaded content
            # This ensures downstream code (response.text, etc.) works as expected
            response._content = content_buffer.getvalue()
            
        except Exception as e:
            print(f"Failed to fetch {url} at time [{str(datetime.datetime.now())}]: {e}")
            mark_url_processed(url)
            continue


        if ".jpg" in url or ".png" in url or ".gif" in url or ".jpeg" in url:
            mark_url_processed(url)
            continue
            
        soup = BeautifulSoup(response.text, 'html.parser')
        process_scrape_html(soup, url)
        
        # Extract links
        current_domain = extract_domain(url)
        links = soup.select('a')
        for x in links:
            href = x.get('href')
            if href is None:
                continue
            
            if isinstance(href, bytes):
                try:
                    href = href.decode('utf-8')
                except UnicodeDecodeError:
                    continue
            
            if href.startswith('/'):
                parsed_uri = urlparse(url)
                base = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
                href = base + href
            elif not href.startswith('http'):
                if not href.startswith('mailto:') and not href.startswith('tel:'):
                     href = url.rsplit('/', 1)[0] + '/' + href

            # Normalize the extracted link
            norm_href = normalize_url(href)
            
            if extract_domain(norm_href) == current_domain:
                if not is_url_known(norm_href):
                    # Enforce queue limit
                    if len(queue) >= MAX_QUEUE_SIZE:
                        print(f"Queue full ({MAX_QUEUE_SIZE}), dropping discovered URL: {norm_href}")
                    else:
                        add_url_to_db(norm_href, status=0) # Add as pending
                        queue.append(norm_href)
        
        # Mark current URL as processed
        mark_url_processed(url)

def normalize_url(url):
    try:
        parsed = urlparse(url)
        # Lowercase scheme and netloc
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path
        
        # Remove trailing slash from path
        if path and path.endswith('/'):
            path = path[:-1]
            
        # Reconstruct without fragment
        # scheme://netloc/path;parameters?query
        # We ignore params for now as they are rarely used in this context
        # We keep query params as they might be significant
        
        normalized = f"{scheme}://{netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
            
        return normalized
    except:
        return url

def is_url_known(url):
    res = client.query(
        collection_name=processed_urls_collection,
        filter=f'url == "{url}"',
        output_fields=["url"]
    )
    return len(res) > 0

def is_url_processed(url):
    res = client.query(
        collection_name=processed_urls_collection,
        filter=f'url == "{url}" and status == 1',
        output_fields=["url"]
    )
    return len(res) > 0

def get_pending_urls():
    # Fetch all URLs with status 0
    # Note: Milvus query limit might need pagination for large sets, 
    # but for now let's grab a reasonable batch.
    res = client.query(
        collection_name=processed_urls_collection,
        filter='status == 0',
        output_fields=["url"],
        limit=10000 # Cap for now
    )
    return [r["url"] for r in res]

def add_url_to_db(url, status=0):
    client.insert(
        collection_name=processed_urls_collection,
        data=[{"url": url, "status": status}]
    )

def mark_url_processed(url):
    # Milvus doesn't support update/partial update easily in all versions.
    # We typically delete and re-insert, or use upsert if available.
    # MilvusClient.upsert() is available in newer SDKs.
    client.upsert(
        collection_name=processed_urls_collection,
        data=[{"url": url, "status": 1}]
    )





def extract_domain(url):
    return urlparse(url).netloc
def is_valid_scheme(url):
    try:
        x = urlparse(url).scheme
        return "http" in x
    except:
        return False
        
def process_scrape_html(soup, url):
    
    chunks = chunk_soup(soup, url) # an array of dicts
    if chunks: 
        db_store(chunks, COLLECTION_WEB)
    print(f"Visited page; URL: {url}. Inserted: {bool(chunks)}")

def chunk_soup(soup, url):
    """
    Improved chunking logic for RAG:
    1. Prune navigational/irrelevant elements.
    2. Group content by heading (semantic chunking).
    3. Enforce a minimum token length for paragraphs by merging short, adjacent blocks.
    4. Handle max token length by splitting by sentence (with overlap) or list item.
    5. Inject hierarchical context (Title > H1 > H2...) into every chunk.
    6. [NEW] Recursive traversal to capture content from all block-level elements (div, table, etc.).
    """
    
    tokenizer = embedding_fn.tokenizer
    MIN_TOKENS = 15
    OVERLAP_SENTENCES = 2
    OVERLAP_LIST_ITEMS = 3
    
    # --- Helper Functions ---
    
    def count_tokens(text):
        return len(tokenizer.encode(text))

    def split_paragraph(text, context_str):
        # Prepend context if not already present
        full_text = text
        if context_str and context_str not in text:
            full_text = f"{context_str}\n{text}"

        sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
        chunks = []
        curr = []
        
        sentences_content = re.split(r'(?<=[.!?])\s+', text.strip())
        
        curr = []
        for sent in sentences_content:
            curr.append(sent)
            # Check size with context
            joined_content = " ".join(curr)
            candidate = f"{context_str}\n{joined_content}" if context_str else joined_content
            
            if count_tokens(candidate) > max_tokens:
                # Backtrack
                overlap_size = min(len(curr) - 1, OVERLAP_SENTENCES)
                prev_content = " ".join(curr[:-1])
                
                chunk_text = f"{context_str}\n{prev_content}" if context_str else prev_content
                if count_tokens(chunk_text) >= MIN_TOKENS:
                    chunks.append(chunk_text)
                
                curr = curr[-overlap_size:]
                curr.append(sent)
        
        final_content = " ".join(curr)
        final_chunk = f"{context_str}\n{final_content}" if context_str else final_content
        if final_content and count_tokens(final_chunk) >= MIN_TOKENS:
            chunks.append(final_chunk)
            
        return chunks

    def serialize_html(elements):
        return "".join(str(e) for e in elements)
    
    # --- Pruning ---
    irrelevant_selectors = [
        'header', 'footer', 'nav', '.sidebar', '.ad', '.ads',
        '#menu', '#navigation', '.skip-link', 'script', 'style', 'noscript'
    ]
    for selector in irrelevant_selectors:
        for element in soup.select(selector):
            element.decompose()
            
    # --- Main Chunker Logic ---

    all_chunks = []
    
    # Extract Page Title
    page_title = ""
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    if not page_title:
        page_title = urlparse(url).netloc
        
    # Heading Stack: list of (level, text)
    heading_stack = [] 
    
    buffer = [] # List of strings
    
    site_name = urlparse(url).netloc
    total_chars_processed = 0

    BLOCK_TAGS = {
        "p", "div", "article", "section", "aside", "blockquote", "figure", 
        "table", "ul", "ol", "dl", "form", "nav", "header", "footer", "address", "main"
    }
    
    def get_current_context():
        path = [page_title] + [h[1] for h in heading_stack]
        return " > ".join(path)

    def flush_buffer(force=False):
        nonlocal buffer
        if not buffer:
            return

        text = "".join(buffer).strip()
        if not text:
            buffer = []
            return

        # If not forcing flush, and text is too short, keep accumulating
        if not force and count_tokens(text) < MIN_TOKENS:
            return

        # Create chunks
        context_str = get_current_context()
        out = []
        
        full_text = f"{context_str}\n{text}" if context_str else text
        
        if count_tokens(full_text) > max_tokens:
            out = split_paragraph(text, context_str)
        else:
            out = [full_text]
            
        current_heading = heading_stack[-1][1] if heading_stack else page_title
        
        for txt in out:
            all_chunks.append({
                "text": txt,
                "url": url,
                "heading": current_heading,
                "site_name": site_name,
                "crawl_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
            
        buffer = []

    def ensure_newline():
        if buffer and not buffer[-1].endswith("\n"):
            buffer.append("\n")

    def ensure_space():
        if buffer and not buffer[-1].endswith(" ") and not buffer[-1].endswith("\n"):
            buffer.append(" ")

    def traverse(node):
        nonlocal total_chars_processed
        
        if len(all_chunks) >= MAX_CHUNKS_PER_PAGE:
            return
        if total_chars_processed >= MAX_TEXT_PER_PAGE_CHARS:
            return

        if isinstance(node, NavigableString):
            text = str(node)
            if text.strip():
                # Normalize whitespace slightly but keep structure? 
                # For now, just append. split_paragraph handles internal spaces.
                # But we want to avoid "wordword".
                # If the previous node was a tag that didn't add space, we might need one.
                # But usually HTML ignores newlines.
                # Let's replace newlines with spaces in text nodes to be safe, 
                # unless inside pre? (Ignoring pre for now).
                clean_text = re.sub(r'\s+', ' ', text)
                buffer.append(clean_text)
                total_chars_processed += len(clean_text)
            return

        if isinstance(node, Tag):
            # Headings
            if node.name in ["h1","h2","h3","h4","h5","h6"]:
                flush_buffer(force=True)
                try:
                    level = int(node.name[1])
                    text = node.get_text(strip=True)
                    if text:
                        while heading_stack and heading_stack[-1][0] >= level:
                            heading_stack.pop()
                        heading_stack.append((level, text))
                except:
                    pass
                # We don't recurse into headings for content, just use them as delimiters
                return

            # Block Tags
            if node.name in BLOCK_TAGS:
                ensure_newline()
                flush_buffer(force=False) # Soft flush at start of block
                
                for child in node.children:
                    traverse(child)
                    
                ensure_newline()
                flush_buffer(force=False) # Soft flush at end of block
                return

            # List Items
            if node.name == "li":
                ensure_newline()
                # Try to detect if ordered or unordered?
                # Parent should be ul or ol.
                marker = "- "
                if node.parent and node.parent.name == "ol":
                    # Find index
                    # This is expensive, maybe just use "1. " or "- " always?
                    # Let's just use "- " for simplicity or try to be smart.
                    # Being smart might be slow.
                    marker = "- " 
                buffer.append(marker)
                
                for child in node.children:
                    traverse(child)
                return
            
            # Definition List Items
            if node.name == "dt":
                ensure_newline()
                buffer.append("**") # Bold term
                for child in node.children:
                    traverse(child)
                buffer.append("** ")
                return
            if node.name == "dd":
                ensure_space()
                for child in node.children:
                    traverse(child)
                return

            # Table Rows/Cells
            if node.name == "tr":
                ensure_newline()
                for child in node.children:
                    traverse(child)
                return
            
            if node.name in ["td", "th"]:
                ensure_space()
                for child in node.children:
                    traverse(child)
                buffer.append(" | ") # Visual separator for table cells
                return

            if node.name == "br":
                buffer.append("\n")
                return

            # Inline / Other
            for child in node.children:
                traverse(child)

    # Start traversal from body
    if soup.body:
        traverse(soup.body)
    else:
        traverse(soup)
        
    # Final flush
    flush_buffer(force=True)

    return all_chunks


def db_store(chunks, collection_name):
    if not chunks:
        return
        
    # Extract text for embedding
    docs = [c["text"] for c in chunks]
    
    vectors = embedding_fn.encode_documents(docs)

    for i in range(len(chunks)):
        chunks[i]["vector"] = vectors[i]

    res = client.insert(collection_name=collection_name, data=chunks)
    print(f"Inserted {len(chunks)} chunks into {collection_name}. Res: {res}")

def fetch_earthquake_feed(url):
    print(f"Fetching earthquake feed: {url}")
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Failed to fetch feed: {resp.status_code}")
            return []
        
        data = resp.json()
        features = data.get('features', [])
        chunks = []
        
        for f in features:
            props = f['properties']
            # Annotate
            try:
                time_str = datetime.datetime.fromtimestamp(props['time'] / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                text = f"Magnitude {props['mag']} earthquake at {props['place']} on {time_str}. Status: {props['status']}."
                origin = props['url']
                
                chunks.append({
                    "text": text,
                    "url": origin,
                    "heading": "Earthquake Report",
                    "site_name": "USGS Earthquake Feed",
                    "crawl_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
                })
            except Exception as e:
                print(f"Error processing feature: {e}")
                continue
                 
        return chunks
    except Exception as e:
        print(f"Error fetching feed: {e}")
        return []


def init_collection():
    
    # 1. PDF Collection - Removed
    # if not client.has_collection(collection_name=COLLECTION_PDF):
    #     ...


    # 2. Web Collection
    if not client.has_collection(collection_name=COLLECTION_WEB):
        # Define BM25 Function
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR), # BM25 Output
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="crawl_date", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="site_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=512),
        ]

        schema = CollectionSchema(fields, description="Web Scraped Data", functions=[bm25_function])
        client.create_collection(
            collection_name=COLLECTION_WEB,
            schema=schema,
        )

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        client.create_index(collection_name=COLLECTION_WEB, index_params=index_params)
        print(f"Collection '{COLLECTION_WEB}' created.")
        
        # Initial load of month feed
        # print("Performing initial load of month feed...")
        # chunks = fetch_earthquake_feed(MONTH_FEED)
        # if chunks:
        #     db_store(chunks, COLLECTION_WEB)

    else:
        print(f"Collection '{COLLECTION_WEB}' already exists.")

    # 3. Processed URLs Collection (for persistence/deduplication)
    if not client.has_collection(collection_name=processed_urls_collection):
        url_fields = [
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048, is_primary=True),
            FieldSchema(name="status", dtype=DataType.INT64), # 0 = Pending, 1 = Processed
        ]
        url_schema = CollectionSchema(url_fields, description="Track processed URLs with status")
        client.create_collection(
            collection_name=processed_urls_collection,
            schema=url_schema
        )
        print(f"Collection '{processed_urls_collection}' created with status field.")
    else:
        print(f"Collection '{processed_urls_collection}' already exists.")


if __name__ == '__main__':

    init_collection()
    


    u =["https://www.earthquakecountry.org/",
        "https://www.usgs.gov/programs/earthquake-hazards/faqs-category",
        "https://myshake.berkeley.edu",
        "https://seismo.berkeley.edu",
        "https://www.ready.gov/earthquakes",
        "https://www.redcross.org/get-help/how-to-prepare-for-emergencies/types-of-emergencies/earthquake.html",
        "https://www.caloes.ca.gov/",
        "https://www.gdacs.org/",
        "https://www.ifrc.org/earthquake",
        "https://www.usgs.gov/programs/earthquake-hazards/science/20-largest-earthquakes-world-1900"
        ]
    
    scrape(u)
