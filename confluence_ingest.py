import os
import logging
from bs4 import BeautifulSoup
from atlassian import Confluence
from dotenv import load_dotenv
import datetime

# Import existing pipeline components from standardscraper
# ensuring we reuse the initialized client and logic.
from standardscraper import (
    process_scrape_html,
    init_collection,
    get_confluence_metadata,
    update_confluence_metadata,
    delete_page_chunks
)

load_dotenv()

# Configuration
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "https://confluence.calsol.dev/")
CONF_AT = os.getenv("CONF_AT")

# Initialize Confluence Client
# Using token authentication as per user context
if CONF_AT:
    confluence = Confluence(
        url=CONFLUENCE_URL,
        token=CONF_AT
    )
else:
    print("Warning: CONF_AT not set in environment.")
    confluence = None


import json


STOP_FILE = "confluence_sync_stop.json"

def set_stop_flag(value: bool):
    try:
        with open(STOP_FILE, "w") as f:
            json.dump({"stop": value}, f)
    except Exception as e:
        print(f"Failed to set stop flag: {e}")

def should_stop():
    if not os.path.exists(STOP_FILE):
        return False
    try:
        with open(STOP_FILE, "r") as f:
            data = json.load(f)
            return data.get("stop", False)
    except:
        return False

def clear_stop_flag():
    set_stop_flag(False)

def sync_confluence_space(space_key, status_func=print):
    """
    Syncs pages from a specific Confluence space using CQL.
    Scans from Newest to Oldest, updating new/changed pages and skipping known ones.
    Supports cooperative cancellation via stop flag.
    """
    if not confluence:
        status_func("Confluence client not initialized.")
        return

    # Check initially
    if should_stop():
        status_func("Stopped gracefully.")
        return

    status_func(f"Scanning space {space_key} for new or updated pages...")
    
    # Capture current time (UTC) for logging or future use
    now_ts = datetime.datetime.now(datetime.timezone.utc)
    
    # CQL: All pages in space
    cql = f'space = "{space_key}" AND type = "page"'
    
    # Add ordering: Newest first
    cql += ' ORDER BY lastModified DESC'
    
    start = 0
    limit = 50
    total_processed = 0
    
    while True:
        # Check cancellation before batch
        if should_stop():
            status_func("Stopped gracefully.")
            return

        try:
            # Execute CQL Search
            results = confluence.cql(
                cql, 
                start=start, 
                limit=limit, 
                expand='version,body.storage,ancestors'
            )
            
            pages = results.get('results', [])
            if not pages:
                break
                
            status_func(f"Processing batch of {len(pages)} pages (Start: {start})...")
            
            # Optimization: Pre-fetch metadata for this batch to check versions
            batch_pids = []
            
            normalized_pages = []
            for p in pages:
                if 'content' in p and isinstance(p['content'], dict):
                    normalized_pages.append(p['content'])
                else:
                    normalized_pages.append(p)
            
            for p in normalized_pages:
                if 'id' in p:
                    batch_pids.append(p['id'])
            
            metadata_map = get_confluence_metadata(batch_pids)
            
            changes_processed_in_batch = 0
            
            for page in normalized_pages:
                # Check cancellation inside batch loop for faster response
                if should_stop():
                    status_func("Stopped gracefully.")
                    return

                # 1. Safe ID Access
                pid = page.get('id')
                if not pid:
                     continue
                     
                try:
                    # 2. Version Check (Optimization)
                    current_version_number = page.get('version', {}).get('number', 0)
                    stored_meta = metadata_map.get(pid)
                    
                    if stored_meta:
                        stored_version = stored_meta.get('version', -1)
                        if stored_version >= current_version_number:
                            # Optimization: Skip ingestion if version matches
                            continue
                            
                    ingest_result = ingest_confluence_page(page, space_key)
                    
                    if ingest_result:
                        update_confluence_metadata([{
                            "page_id": pid,
                            "version": current_version_number,
                            "url": ingest_result['url'],
                            "last_seen": now_ts.isoformat(),
                            "space_key": space_key
                        }])
                        status_func(f"Synced: {ingest_result['url']}")
                        changes_processed_in_batch += 1
                        
                except Exception as e:
                    status_func(f"Failed to ingest page {pid}: {e} (Page keys: {list(page.keys()) if isinstance(page, dict) else 'Not Dict'})")

            total_processed += changes_processed_in_batch

            if len(pages) < limit:
                break
                
            start += len(pages)
            
        except Exception as e:
            status_func(f"Error executing CQL at start={start}: {e}")
            break

    status_func(f"Completed Sync for {space_key}. Processed: {total_processed}")


def ingest_confluence_page(page, space_key):
    """
    Process a single page.
    Returns dict with metadata on success, None on failure.
    """
    # 1. Construct URL
    _links = page.get('_links', {})
    webui = _links.get('webui') or _links.get('self')
    
    if not webui:
        return None

    base_url = CONFLUENCE_URL.rstrip('/')
    if not webui.startswith('/'):
        webui = '/' + webui
    
    url = base_url + webui
    
    # 2. Extract Content
    body = page.get('body', {})
    storage = body.get('storage', {})
    html_content = storage.get('value', '')
    
    # If content is missing, try fetching the full page explicitly
    if not html_content and confluence:
        try:
            # print(f"Fetching full content for {page.get('id')}...") 
            full_page = confluence.get_page_by_id(
                page.get('id'), 
                expand='body.storage'
            )
            body = full_page.get('body', {})
            storage = body.get('storage', {})
            html_content = storage.get('value', '')
        except Exception as e:
            print(f"Failed to fetch full page content for {url}: {e}")

    if not html_content:
        return {'url': url} # Mark as seen even if empty

    soup = BeautifulSoup(html_content, 'html.parser')

    # Inject Title
    page_title = page.get('title', '')
    if page_title:
        if not soup.title:
            title_tag = soup.new_tag("title")
            title_tag.string = page_title
            if soup.head:
                soup.head.append(title_tag)
            else:
                head_tag = soup.new_tag("head")
                head_tag.append(title_tag)
                soup.insert(0, head_tag)

    # 3. Process
    try:
        # Prevent duplicates: clear existing chunks for this URL
        delete_page_chunks(url)
        
        process_scrape_html(soup, url)
        # Note: We rely on the caller to update the metadata table
        return {'url': url}
    except Exception as e:
        print(f"Error chunking {url}: {e}")
        return None

def update_confluence():
    """
    Entry point for UI Button.
    """
    init_collection() # Ensure DB exists
    
    # Define Spaces to Sync
    # This could be config driven
    SPACES = ["CG"] 
    
    # If user provided a specific space in Main (generic usage), we might want to respect that too.
    # For now, let's just sync the spaces we know or ask env.
    
    # Assuming "CG" for now as per previous default
    target_space = "CG"
    # CLI Mode: Force sync (now standard)
    sync_confluence_space(target_space)

if __name__ == "__main__":
    try:
        update_confluence()
    except KeyboardInterrupt:
        print("\nSync stopped.")


def get_page_content_by_id(page_id):
    """
    Fetches the full storage format of a page by its ID.
    Returns the raw HTML content string or None if failed.
    """
    if not confluence:
        print("Confluence client not initialized.")
        return None
        
    try:
        page = confluence.get_page_by_id(page_id, expand='body.storage')
        return page.get('body', {}).get('storage', {}).get('value', '')
    except Exception as e:
        print(f"Error fetching page {page_id}: {e}")
        return None

def get_page_content_by_url(url):
    """
    Fetches full page content using URL.
    1. Looks up page_id from CONFLUENCE_PAGES_COLLECTION using the URL.
    2. Calls get_page_content_by_id.
    """
    if not confluence:
        return None
        
    # Check if we can extract ID from URL regex (fast path)
    # CalSol URLs: .../pages/12345/Title
    import re
    match = re.search(r'/pages/(\d+)/', url)
    if match:
        page_id = match.group(1)
        return get_page_content_by_id(page_id)

    # Slow path: Query Metadata DB
    # Note: standardscraper.client is available via import
    from standardscraper import client, CONFLUENCE_PAGES_COLLECTION
    
    try:
        res = client.query(
            collection_name=CONFLUENCE_PAGES_COLLECTION,
            filter=f'url == "{url}"',
            output_fields=["page_id"]
        )
        if res and len(res) > 0:
            page_id = res[0]['page_id']
            return get_page_content_by_id(page_id)
        else:
            print(f"URL not found in metadata: {url}")
            return None
    except Exception as e:
        print(f"Metadata lookup failed for {url}: {e}")
        return None
