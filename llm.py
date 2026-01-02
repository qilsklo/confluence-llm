import os
import datetime
import json
from dotenv import load_dotenv
import google.genai as genai
import streamlit as st
import standardscraper
from pymilvus import AnnSearchRequest, RRFRanker
from confluence_ingest import sync_confluence_space

# Load environment variables
load_dotenv()

# --- PROMPT DEFINITION ---
CONFLUENCE_PROMPT = """You are a helpful assistant for the CalSol Solar Car Team.
Your goal is to answer questions using the provided context from Confluence, but you may use internal knowledge to supplement low-quality context.

--- INSTRUCTIONS ---
1. Use the provided Context to answer the user's question.
2. If the answer is not in the context, state that you do not have enough information.
3. internal knowledge can be used to explain concepts, but context is the primary source of truth.
4. If you find a relevant page in the context, mention its title or provide its URL in your answer.
Never directly tell the user about "provided context" explicitly. You may use the context in your answer, but the user is unaware that this context exists.

--- CONTEXT ---
{context_text}

--- CHAT HISTORY ---
{history_text}

User: {user_query}
Assistant:
"""

def get_api_key():
    # Try to get from environment variable first
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    
    # Check if running in Streamlit
    try:
        if st.runtime.exists():
            # Try to get from Streamlit secrets
            if "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]

            # If not found, ask user via sidebar
            if "api_key" not in st.session_state:
                st.session_state.api_key = ""
            
            key = st.sidebar.text_input("Enter Gemini API Key", type="password", value=st.session_state.api_key)
            if key:
                st.session_state.api_key = key
                return key
            
            st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
            st.stop()
            return None
    except Exception:
        pass # Not in streamlit or runtime not ready

    # Fallback to CLI input
    print("GEMINI_API_KEY not found in .env")
    key = input("Please enter your Gemini API Key: ").strip()
    if not key:
        print("API Key is required to proceed.")
        exit(1)
    return key


def search_knowledge_base(query, limit=10):
    """
    Performs hybrid search (BM25 + Vector) across the Web collection.
    """
    try:
        # 1. Prepare Requests
        query_dense = standardscraper.embedding_fn.encode_queries([query])[0]
        
        # Dense Request (Vector)
        req_dense = AnnSearchRequest(
            data=[query_dense],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=limit * 2 
        )
        
        # Sparse Request (BM25)
        req_sparse = AnnSearchRequest(
            data=[query], # Text for BM25 function
            anns_field="sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=limit * 2
        )
        
        reqs = [req_dense, req_sparse]
        ranker = RRFRanker(k=60)
        
        results = []
        
        # Search Web Collection (Confluence Pages)
        try:
            # Check if collection is empty to avoid Milvus crash
            stats = standardscraper.client.get_collection_stats(standardscraper.COLLECTION_WEB)
            if stats['row_count'] > 0:
                res_web = standardscraper.client.hybrid_search(
                    collection_name=standardscraper.COLLECTION_WEB,
                    reqs=reqs,
                    ranker=ranker,
                    limit=limit,
                    output_fields=["text", "url", "site_name", "heading", "crawl_date"]
                )
                
                if res_web and len(res_web) > 0:
                    seen_signatures = set()
                    for hit in res_web[0]:
                        entity = hit['entity']
                        content = entity.get('text', '')
                        url = entity.get('url', '#')
                        
                        # Deduplicate based on URL and Content
                        if (url, content) in seen_signatures:
                            continue
                        seen_signatures.add((url, content))

                        results.append({
                            "type": "WEB",
                            "content": content,
                            "site_name": entity.get('site_name', 'Confluence'),
                            "heading": entity.get('heading', ''),
                            "url": url,
                            "date": entity.get('crawl_date', ''),
                            "score": hit['distance']
                        })
                        
        except Exception as e:
            print(f"Hybrid Search failed: {e}")
            
        return results

    except Exception as e:
        print(f"Search failed: {e}")
        return []

def query_rag(user_query, history, api_key, verbose=False):
    # Initialize Client with the new SDK syntax
    client = genai.Client(api_key=api_key)
    
    # 1. Retrieve Context
    raw_docs = search_knowledge_base(user_query, limit=10)
    
    # 2. Format Context
    formatted_docs = []
    for i, doc in enumerate(raw_docs):
        citation = f"(Source: {doc['heading']} | {doc['url']})"
        formatted_docs.append(f"[DOC {i+1}] {citation}\n{doc['content']}")
    
    context_text = "\n\n".join(formatted_docs)
    
    if not context_text:
        context_text = "No relevant documents found in the database."

    if verbose:
        print(f"\n[DEBUG] Context passed to LLM:\n{context_text}\n[DEBUG] End Context")

    # 3. Format History
    history_text = ""
    for role, msg in history:
        history_text += f"{role}: {msg}\n"
    
    # 4. Construct Prompt
    prompt = CONFLUENCE_PROMPT.format(
        context_text=context_text, 
        history_text=history_text, 
        user_query=user_query
    )

    # 5. Generate Response
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error calling LLM: {e}"

def main_streamlit():
    st.set_page_config(page_title="Confluence Assistant", page_icon="�️")
    st.title("☀️ CalSol Confluence Assistant")

    api_key = get_api_key()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about CalSol..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                # Convert Streamlit history to the format expected by query_rag
                history_tuples = [(msg["role"].capitalize(), msg["content"]) for msg in st.session_state.messages[:-1]]
                
                response = query_rag(prompt, history_tuples, api_key)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar Admin Section
    with st.sidebar:
        st.divider()
        st.header("Admin Controls")
        
        # Space Key Input (optional, could be fixed)
        space_key = st.text_input("Which space should we sync?", value="CG")
        
        if st.button("Update From Confluence"):
            if not space_key:
                st.error("Please provide a Space Key.")
            else:
                st.info(f"Starting sync for space: {space_key}...")
                log_container = st.empty()
                logs = []
                
                def streamlit_logger(msg):
                    # Append new log line
                    logs.append(msg)
                    # Update container with all logs (simulating scroll)
                    # Keeping last 10 lines for cleanliness or scrollable container
                    log_text = "\n".join(logs[-10:]) 
                    log_container.code(log_text, language='text')
                    # Also print to console for debugging
                    print(msg)
                
                try:
                    with st.spinner("Syncing... This may take a while."):
                        sync_confluence_space(space_key, status_func=streamlit_logger, incremental=True)
                    st.success("Sync Completed!")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

def main_cli():
    api_key = get_api_key()
    history = [] 
    
    print("CalSol Confluence Assistant (Type 'quit' to exit)")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ['quit', 'exit']:
            break
        
        answer = query_rag(query, history, api_key, verbose=True)
        print("\nResponse:")
        print(answer)
        
        # Update history
        history.append(("User", query))
        history.append(("Assistant", answer))
        
        # Keep history manageable
        if len(history) > 10:
            history = history[-10:]

if __name__ == "__main__":
    try:
        if st.runtime.exists():
            main_streamlit()
        else:
            main_cli()
    except Exception:
        main_cli()
