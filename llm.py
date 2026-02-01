import os
import datetime
import json
from dotenv import load_dotenv
import google.genai as genai
import streamlit as st
import standardscraper
from pymilvus import AnnSearchRequest, RRFRanker
from confluence_ingest import sync_confluence_space
import confluence_ingest

# Load environment variables
load_dotenv()

# --- PROMPT DEFINITION ---
# --- PROMPT DEFINITION ---
# --- PROMPTS ---

DECOMPOSITION_PROMPT = """You are a senior engineer breaking down a complex user question into specific research steps.
User Query: "{user_query}"

Goal: Identify the key pieces of information needed to answer this fully.
Output: A Python list of strings, where each string is a specific search query. Limit to 3-5 high-value queries.

Example:
User: "What powers the car?"
Output: ["electrical power distribution system", "battery pack specifications", "main power components list", "solar array power rating"]

Output:
"""

ENGINEER_PROMPT = """You are the Chief Engineer of the CalSol Solar Car Team.
Your goal is to explain the system to a new member based *strictly* on the provided documentation.
You are teaching them how the car works.

--- INSTRUCTIONS ---
1.  **Synthesize**: Combine information from multiple sources to build a complete picture.
2.  **Inference**: You are allowed to infer system design from partial information, but you must label it. (e.g., "While not explicitly stated, the presence of X suggests Y...")
3.  **Honesty**: If a critical piece of information is missing, explicitly state what is unknown.
4.  **Citations**: Cite the title of the page you are referencing.
5.  **Tone**: Professional, technical, educational, and authoritative.

--- CONTEXT ---
{context_text}

--- CHAT HISTORY ---
{history_text}

User: {user_query}
Chief Engineer:
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


def generate_search_plan(query, api_key):
    """
    Decomposes the user query into sub-questions using a lightweight LLM call.
    """
    client = genai.Client(api_key=api_key)
    prompt = DECOMPOSITION_PROMPT.format(user_query=query)
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        text = response.text.strip()
        
        # Parse list from text (handle various formats)
        import ast
        import re
        
        # Try to find a list-like structure
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            list_str = match.group(0)
            try:
                plan = ast.literal_eval(list_str)
                if isinstance(plan, list):
                    return plan
            except:
                pass
                
        # Fallback: split by newlines if it looks like a list
        lines = text.split('\n')
        plan = [l.strip('- ').strip() for l in lines if l.strip()]
        if plan:
            return plan

        return [query]
    except Exception as e:
        print(f"Decomposition failed: {e}")
        return [query]

def search_knowledge_base(queries, limit=5):
    """
    Performs hybrid search (BM25 + Vector) across the Web collection for MULTIPLE queries.
    Aggregates results.
    """
    if isinstance(queries, str):
        queries = [queries]
        
    all_results = {} # url -> doc_dict (deduplication by URL)
    
    for query in queries:
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
            
            # Search Web Collection
            stats = standardscraper.client.get_collection_stats(standardscraper.COLLECTION_WEB)
            if stats['row_count'] > 0:
                res_web = standardscraper.client.hybrid_search(
                    collection_name=standardscraper.COLLECTION_WEB,
                    reqs=reqs,
                    ranker=ranker,
                    limit=limit,
                    output_fields=["text", "url", "site_name", "heading", "id"] # Added 'id'
                )
                
                if res_web and len(res_web) > 0:
                    for hit in res_web[0]:
                        entity = hit['entity']
                        url = entity.get('url', '#')
                        heading = entity.get('heading', 'No Title')
                        
                        if url not in all_results:
                            all_results[url] = {
                                "heading": heading,
                                "url": url,
                                "content": entity.get('text', ''),
                                "id": entity.get('id', None),
                                "score": hit['distance']
                            }
                        else:
                            # Keep highest score? Or just append content if different?
                            # For now, simplistic dedup.
                            pass

        except Exception as e:
            print(f"Search failed for '{query}': {e}")

    return list(all_results.values())

def query_rag(user_query, history, api_key, verbose=False):
    client = genai.Client(api_key=api_key)
    
    # 1. Decomposition
    if verbose: print("Thinking about research plan...")
    plan = generate_search_plan(user_query, api_key)
    if verbose: print(f"Research Plan: {plan}")
    
    # 2. Deep Retrieval
    raw_docs = search_knowledge_base(plan, limit=5)
    
    # 3. Context Expansion (Fetch Full Content)
    # This simulates "Senior Engineer" reading the whole doc
    expanded_context = []
    
    # Lazy import to avoid circular dependency issues at top level if any
    from confluence_ingest import get_page_content_by_url
    from bs4 import BeautifulSoup
    
    for doc in raw_docs:
        # Check if we can/should fetch full content
        # For now, let's fetch full content for ALL top hits to ensure comprehensive synthesis
        full_content = None
        if doc.get('url'):
            if verbose: print(f"Reading full page: {doc['heading']}")
            raw_html = get_page_content_by_url(doc['url'])
            if raw_html:
                # Basic text extraction
                soup = BeautifulSoup(raw_html, 'html.parser')
                full_content = soup.get_text(separator='\n')
        
        content_to_use = full_content if full_content else doc['content']
        
        citation = f"SOURCE: {doc['heading']} ({doc['url']})"
        expanded_context.append(f"{citation}\n---\n{content_to_use}\n---")
    
    context_text = "\n\n".join(expanded_context)
    
    if not context_text:
        context_text = "No relevant documents found in the database."

    if verbose:
        print(f"\n[DEBUG] Context Length: {len(context_text)} chars")

    # 4. Format History
    history_text = ""
    for role, msg in history:
        history_text += f"{role}: {msg}\n"
    
    # 5. Synthesis
    prompt = ENGINEER_PROMPT.format(
        context_text=context_text, 
        history_text=history_text, 
        user_query=user_query
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error calling LLM: {e}"

def main_streamlit():
    st.set_page_config(page_title="Confluence Assistant", page_icon="☀️")
    st.title("☀️ CalSol Confluence Assistant")

    # --- AUTHENTICATION ---
    from google_auth_oauthlib.flow import Flow
    import requests

    def check_auth():
        # Check if user is already authenticated
        if "google_auth_token" in st.session_state:
            return True

        # Check for OAuth callback
        if "code" in st.query_params:
            try:
                code = st.query_params["code"]
                
                # Create flow instance to exchange code for token
                flow = Flow.from_client_config(
                    {
                        "web": {
                            "client_id": st.secrets["google"]["client_id"],
                            "client_secret": st.secrets["google"]["client_secret"],
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                        }
                    },
                    scopes=["https://www.googleapis.com/auth/userinfo.email", "openid"],
                    redirect_uri=st.secrets["google"]["redirect_uri"],
                )
                
                flow.fetch_token(code=code)
                credentials = flow.credentials
                
                # Get User Info
                user_info = requests.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {credentials.token}"}
                ).json()
                
                email = user_info.get("email", "")
                
                # Check Domain
                if not email.endswith("@berkeley.edu"):
                    st.error("Access restricted to @berkeley.edu emails only.", icon="🚫")
                    st.stop()
                
                # Save to session state
                st.session_state["google_auth_token"] = credentials.token
                st.session_state["user_email"] = email
                st.session_state["user_name"] = user_info.get("name", "User")
                
                # Clear query params to prevent re-submission and clean URL
                st.query_params.clear()
                st.rerun()

            except Exception as e:
                st.error(f"Authentication failed: {e}")
                st.stop()

        return False

    if not check_auth():
        st.info("Please log in with your Berkeley Google account to access the assistant.", icon="🔒")
        
        # Create Flow for Login Button
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": st.secrets["google"]["client_id"],
                    "client_secret": st.secrets["google"]["client_secret"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=["https://www.googleapis.com/auth/userinfo.email", "openid"],
            redirect_uri=st.secrets["google"]["redirect_uri"],
        )
        
        auth_url, _ = flow.authorization_url(prompt='consent')
        
        st.link_button("Log in with Google", auth_url)
        st.stop()
    
    st.sidebar.markdown(f"Logged in as **{st.session_state.get('user_name')}** ({st.session_state.get('user_email')})")
    if st.sidebar.button("Log out"):
        for key in ["google_auth_token", "user_email", "user_name"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    # ----------------------

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
            with st.spinner("Thinking & Searching..."):
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
        space_key = st.selectbox("Which space should we sync?", ["CG"])
        
        col1, col2 = st.columns(2)
        with col1:
             if st.button("Start Sync"):
                 confluence_ingest.clear_stop_flag()
                 st.session_state.run_sync = True
        
        with col2:
             if st.button("Stop Sync"):
                 confluence_ingest.set_stop_flag(True)
                 st.session_state.run_sync = False
                 st.warning("Stopping sync...")

        if st.session_state.get("run_sync"):
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
                        sync_confluence_space(space_key, status_func=streamlit_logger)
                    st.success("Sync Completed!")
                except Exception as e:
                    st.error(f"Sync failed: {e}")
                finally:
                    # Reset state after completion or failure so it doesn't auto-run on reload
                    st.session_state.run_sync = False
        st.link_button("Submit a bug report", "https://forms.gle/TETjF28UvvzLsTVNA")
        
        st.markdown(
            """
            <div style='text-align: center; color: #888888; margin-top: 50px; font-size: 0.8em;'>
                By Soham Kulkarni (<a href='mailto:sohamkulkarni@berkeley.edu' style='color: #888888; text-decoration: none;'>sohamkulkarni@berkeley.edu</a>)
            </div>
            """,
            unsafe_allow_html=True
        )

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
    if st.runtime.exists():
        main_streamlit()
    else:
        main_cli()
