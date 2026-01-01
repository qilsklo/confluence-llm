
import os
import datetime
import re
import json
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import standardscraper

import prompts
from pymilvus import AnnSearchRequest, RRFRanker

# Load environment variables
load_dotenv()

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

import earthquake_data


def get_search_params(user_query, api_key):
    """
    Uses LLM to extract search parameters (date, magnitude, etc.) from the query.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prompt = prompts.SEARCH_PARAM_PROMPT.format(
        current_datetime=current_time,
        user_query=user_query
    )
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        params = json.loads(response.text)
        return params
    except Exception as e:
        print(f"Error extracting params: {e}")
        return {}


def search_knowledge_base(query, limit=15):
    """
    Performs hybrid search (BM25 + Vector) across both PDF and Web collections.
    """
    try:
        # 1. Prepare Requests
        query_dense = standardscraper.embedding_fn.encode_queries([query])[0]
        
        # Dense Request (Vector)
        req_dense = AnnSearchRequest(
            data=[query_dense],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=limit * 2 # Get more candidates
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
        
        # 2. Search PDF Collection
        try:
            # Check if collection is empty to avoid Milvus crash on sparse search
            stats = standardscraper.client.get_collection_stats(standardscraper.COLLECTION_PDF)
            if stats['row_count'] > 0:
                res_pdf = standardscraper.client.hybrid_search(
                    collection_name=standardscraper.COLLECTION_PDF,
                    reqs=reqs,
                    ranker=ranker,
                    limit=limit,
                    output_fields=["text", "title", "page_num", "author", "publication_year"]
                )
                
                if res_pdf and len(res_pdf) > 0:
                    for hit in res_pdf[0]:
                        entity = hit['entity']
                        results.append({
                            "type": "PDF",
                            "content": entity.get('text', ''),
                            "title": entity.get('title', 'Unknown Title'),
                            "page_num": entity.get('page_num', '?'),
                            "author": entity.get('author', ''),
                            "year": entity.get('publication_year', ''),
                            "score": hit['distance']
                        })
            
        except Exception as e:
            print(f"PDF Hybrid Search failed: {e}")

        # 3. Search Web Collection
        try:
            # Check if collection is empty
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
                    for hit in res_web[0]:
                        entity = hit['entity']
                        results.append({
                            "type": "WEB",
                            "content": entity.get('text', ''),
                            "site_name": entity.get('site_name', 'Unknown Site'),
                            "heading": entity.get('heading', ''),
                            "url": entity.get('url', '#'),
                            "crawl_date": entity.get('crawl_date', ''),
                            "score": hit['distance']
                        })
        except Exception as e:
            print(f"Web Hybrid Search failed: {e}")
            
        # 4. Sort Combined Results
        # Since scores are RRF scores (0-1), we can compare them directly?
        # RRF scores are comparable across collections if k is same.
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    except Exception as e:
        print(f"Hybrid search failed: {e}")
        return []

def query_rag(user_query, history, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # 1. Fetch latest data (simple polling simulation: fetch on query)
    try:
        new_events = earthquake_data.manager.fetch_feed("all_day")
        earthquake_data.manager.process_features(new_events)
    except Exception as e:
        print(f"Error updating feed: {e}")

    # 2. Get Search Params
    search_params = get_search_params(user_query, api_key)
    start_date = search_params.get("start_date")
    end_date = search_params.get("end_date")
    semantic_query = search_params.get("semantic_query")
    user_coordinates = search_params.get("user_coordinates")
    mode = search_params.get("mode", "event") # Default to event
    
    # Default to UC Berkeley if no location specified
    if not user_coordinates:
        user_coordinates = [37.8715, -122.2730]
        
    if not semantic_query:
        semantic_query = user_query

    # 3. Get Context from EarthquakeManager
    min_mag = search_params.get("min_magnitude")
    if min_mag is None:
        mag_match = re.search(r'\b(?:magnitude|mag|m)\s*(\d+(?:\.\d+)?)', user_query, re.IGNORECASE)
        if mag_match:
            try:
                min_mag = float(mag_match.group(1))
            except ValueError:
                pass
    
    context_data = earthquake_data.manager.get_context_for_llm(
        min_magnitude=min_mag,
        start_date=start_date,
        end_date=end_date,
        user_lat=user_coordinates[0] if user_coordinates else None,
        user_lon=user_coordinates[1] if user_coordinates else None
    )
    
    # Inject reference location so LLM knows what "distance_to_user_km" is relative to
    context_data["reference_location"] = search_params.get("user_location") or "your location"
    event_context_json = json.dumps(context_data, indent=2)
    
    # 4. Get Safety Docs via Vector Search
    # Increase limit to get a mix of PDF and Web results
    raw_docs = search_knowledge_base(semantic_query, limit=15)
    
    # Format Context based on Mode
    if mode == "concept":
        # Structured format for citation generation
        safety_context_text = json.dumps(raw_docs, indent=2)
    else:
        # Simple string format for other modes (backward compatibility)
        formatted_docs = []
        for i, doc in enumerate(raw_docs):
            if doc['type'] == 'PDF':
                citation = f"(PDF: {doc['title']}, p.{doc['page_num']})"
            else:
                citation = f"(WEB: {doc['site_name']} - {doc['heading']}) [Link]({doc['url']})"
            formatted_docs.append(f"[DOC {i+1}] {citation}\n{doc['content']}")
        safety_context_text = "\n\n".join(formatted_docs)

    # Combine Contexts
    full_context = f"""
--- EVENT DATA (Authoritative) ---
{event_context_json}

--- SAFETY DOCS (Reference) ---
{safety_context_text}
"""
    
    # 5. Generate Answer
    history_text = ""
    for role, msg in history:
        history_text += f"{role}: {msg}\n"
    history_text += f"User: {user_query}"

    # Select Prompt based on Mode
    if mode == "concept":
        prompt = prompts.CONCEPT_ANSWER_PROMPT.format(
            context_text=full_context, 
            history_text=history_text, 
            user_query=user_query
        )
    else:
        # Default to Event/Safety prompt
        prompt = prompts.RAG_ANSWER_PROMPT.format(
            context_text=full_context, 
            history_text=history_text, 
            user_query=user_query
        )

    try:
        response = model.generate_content(prompt)
        return mode + "\n\n" + response.text
    except Exception as e:
        return f"Error calling LLM: {e}"

def main_streamlit():
    st.set_page_config(page_title="Earthquake Assistant", page_icon="🌍")
    st.title("🌍 Earthquake Assistant")

    api_key = get_api_key()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about earthquakes..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Convert Streamlit history to the format expected by query_rag (list of tuples)
                # Capitalize roles to match the expected format in query_rag
                history_tuples = [(msg["role"].capitalize(), msg["content"]) for msg in st.session_state.messages[:-1]]
                
                response = query_rag(prompt, history_tuples, api_key)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main_cli():
    api_key = get_api_key()
    history = [] # List of (role, message) tuples
    
    print("Earthquake Assistant (Type 'quit' to exit)")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ['quit', 'exit']:
            break
        
        answer = query_rag(query, history, api_key)
        print("\nResponse:")
        print(answer)
        
        # Update history
        history.append(("User", query))
        history.append(("Assistant", answer))
        
        # Keep history manageable
        if len(history) > 10:
            history = history[-10:]

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        if st.runtime.exists():
            main_streamlit()
        else:
            main_cli()
    except Exception:
        main_cli()
