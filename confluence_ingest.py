import os
import logging
from bs4 import BeautifulSoup
from atlassian import Confluence
from dotenv import load_dotenv

# Import existing pipeline components from standardscraper
# ensuring we reuse the initialized client and logic.
from standardscraper import (
    process_scrape_html,
    is_url_processed,
    mark_url_processed,
    init_collection
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

def ingest_confluence_space(space_key):
    """
    Ingests all pages from a specific Confluence space.
    
    Args:
        space_key (str): The Key of the Confluence space (e.g., 'DS', 'ENG').
    """
    if not confluence:
        print("Confluence client not initialized.")
        return

    print(f"Starting ingestion for Space: {space_key}")
    
    start = 0
    limit = 50
    
    while True:
        try:
            # Fetch pages with body.storage, version, and ancestors expanded
            pages = confluence.get_all_pages_from_space(
                space_key,
                start=start,
                limit=limit,
                expand='body.storage,version,ancestors',
                content_type='page'
            )
            
            # If no results, we are done
            if not pages:
                print("No more pages found.")
                break
                
            print(f"Fetched batch of {len(pages)} pages (start={start})...")
            
            for page in pages:
                try:
                    ingest_confluence_page(page)
                except Exception as e:
                    print(f"Failed to ingest page ID {page.get('id')}: {e}")
            
            # Use strict pagination check
            # If we received fewer items than the limit, we've reached the end.
            if len(pages) < limit:
                break
                
            start += len(pages)
            
        except Exception as e:
            print(f"Error processing batch at start={start}: {e}")
            # Decide whether to break or continue; breaking is safer to avoid infinite loops on error
            break

    print(f"Completed ingestion for Space: {space_key}")


def ingest_confluence_page(page):
    """
    Helper function to process a single Confluence page object.
    
    1. Construct unique URL.
    2. Check if already processed.
    3. Convert storage format to BeautifulSoup HTML.
    4. Pass to process_scrape_html.
    5. Mark as processed.
    """
    # 1. Construct Unique ID (URL)
    _links = page.get('_links', {})
    webui = _links.get('webui') or _links.get('self')
    
    if not webui:
        print(f"Skipping page {page.get('id')}: No webui link found.")
        return

    # Handle relative URLs
    base_url = CONFLUENCE_URL.rstrip('/')
    if not webui.startswith('/'):
        webui = '/' + webui
    
    url = base_url + webui
    
    # 2. Check if already processed
    if is_url_processed(url):
        print(f"Skipping already processed: {url}")
        return

    # 3. Extract and Convert Content
    body_storage = page.get('body', {}).get('storage', {})
    html_content = body_storage.get('value', '')
    
    if not html_content:
        print(f"Page {url} has no body content. Marking as processed.")
        mark_url_processed(url)
        return

    soup = BeautifulSoup(html_content, 'html.parser')

    # Optional: Inject Title if missing in body (Confluence storage doesn't rely on <title> tags)
    # This helps chunk_soup indentify the context.
    page_title = page.get('title', '')
    if page_title:
        # Check if title tag exists
        if not soup.title:
            title_tag = soup.new_tag("title")
            title_tag.string = page_title
            
            if soup.head:
                soup.head.append(title_tag)
            else:
                # Create head if needed
                head_tag = soup.new_tag("head")
                head_tag.append(title_tag)
                soup.insert(0, head_tag)

    # 4. Pass to existing pipeline
    # process_scrape_html expects (soup, url)
    try:
        process_scrape_html(soup, url)
        
        # 5. Mark as processed 
        # Only mark if successful
        mark_url_processed(url)
        print(f"Successfully processed: {url}")
        
    except Exception as e:
        print(f"Error in process_scrape_html for {url}: {e}")

if __name__ == "__main__":
    # Example usage
    # Replace 'DS' with your target Confluence Space Key
    try:
        init_collection() # Ensure DB exists
        space_to_ingest = "CG"  
        if confluence:
            ingest_confluence_space(space_to_ingest)
    except KeyboardInterrupt:
        print("\nIngestion stopped.")
