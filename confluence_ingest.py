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

STATE_FILE = "confluence_sync_state.json"

def load_sync_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_sync_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Failed to save sync state: {e}")

def sync_confluence_space(space_key, status_func=print, incremental=True):
    """
    Syncs pages from a specific Confluence space using CQL.
    Uses 'lastModified' timestamp to perform incremental updates.
    """
    if not confluence:
        status_func("Confluence client not initialized.")
        return

    status_func(f"Starting Sync for Space: {space_key}")
    
    # 1. Determine Sync Start Time
    # If incremental is False, we ignore state to force full sync.
    last_sync = None
    if incremental:
        state = load_sync_state()
        last_sync = state.get(space_key)
    else:
        status_func("Mode: Full Sync (Forced)")
    
    # Capture current time (UTC) for the next sync state
    # Using a slight buffer (e.g., -1 minute) might be safer for clock skew, but precise is okay.
    # Confluence API usually expects "yyyy-mm-dd" or full timestamps.
    # Let's use the format Confluence CQL expects: "yyyy/MM/dd HH:mm" or similar?
    # Actually, CQL supports "2019-02-04 14:00" format.
    
    now_ts = datetime.datetime.now(datetime.timezone.utc)
    
    cql = f'space = "{space_key}" AND type = "page"'
    
    if last_sync:
        status_func(f"Incremental Sync: Looking for changes since {last_sync}")
        # Make sure last_sync is in a format CQL likes. 
        # State stores ISO format, CQL likes "2006/01/02 15:04" or "2006-01-02 15:04"
        # We'll just trust the passed format or clean it.
        # Simplest is: lastModified > "2024-01-01 12:00"
        
        # Ensure 'T' is replaced by space if needed, though some APIs accept T.
        safe_date = last_sync.replace("T", " ")[:16] # "YYYY-MM-DD HH:MM"
        cql += f' AND lastModified > "{safe_date}"'
    else:
        status_func("Full Sync: Fetching all pages.")
    
    # Add ordering
    cql += ' ORDER BY lastModified DESC'
    
    start = 0
    limit = 50
    total_processed = 0
    
    while True:
        try:
            # Execute CQL Search
            # expand='version' to check vs local, 'body.storage' to ingest
            # Since CQL implies we ONLY got changed pages (if incremental),
            # we can assume we need to process them.
            # However, for robustness, we still fetch them.
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
            # Collect IDs safely
            batch_pids = []
            
            # Normalize pages list to be a list of the actual page dictionaries
            # CQL results often wrap the page in "content" or just return search fields.
            # Based on logs: keys are ['content', 'title', ...]. So the real page data might be in 'content'.
            
            normalized_pages = []
            for p in pages:
                if 'content' in p and isinstance(p['content'], dict):
                    # It's a search result wrapping the content
                    normalized_pages.append(p['content'])
                else:
                    # It's potentially already the page object (or we hope so)
                    normalized_pages.append(p)
            
            for p in normalized_pages:
                if 'id' in p:
                    batch_pids.append(p['id'])
            
            metadata_map = get_confluence_metadata(batch_pids)
            
            changes_processed_in_batch = 0
            
            for page in normalized_pages:
                # 1. Safe ID Access
                pid = page.get('id')
                if not pid:
                     # This might happen if normalized_page is still not quite right, but let's log it.
                     # status_func(f"Skipping malformed (no id): keys={list(page.keys())}")
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
            
    # Update State on success
    # We store the timestamp we started with.
    # Note: If pages were modified DURING the sync, we might miss them next time 
    # if we set time to NOW. But typically acceptable risk.
    if incremental:
        state = load_sync_state() # reload to be safe
    else:
        state = load_sync_state() # load existing or new
        
    state[space_key] = now_ts.strftime("%Y-%m-%d %H:%M")
    save_sync_state(state)

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
    # CLI Mode: Force incremental=False
    sync_confluence_space(target_space, incremental=False)

if __name__ == "__main__":
    try:
        update_confluence()
    except KeyboardInterrupt:
        print("\nSync stopped.")
