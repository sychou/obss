from alive_progress import alive_bar
from dotenv import load_dotenv
import glob
import json
import numpy as np
from openai import OpenAI
import os
import pandas as pd
from rank_bm25 import BM25Okapi
import sqlite3
import sys
import tiktoken

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
TOKEN_LIMIT = int(os.getenv('TOKEN_LIMIT', 8000))
EMBEDDINGS_DB = os.getenv('EMBEDDINGS_DB', 'data/embeddings.db')
DIRS = json.loads(os.getenv('DIRS', '[]'))

def validate_api_key():
    """Validate OpenAI API key by making a test request"""
    print("🔑 Validating OpenAI API key...", end=" ", flush=True)
    try:
        # Make a minimal test request
        client.embeddings.create(
            input="test",
            model="text-embedding-3-small"
        )
        print("✅")
        return True
    except Exception as e:
        print("❌")
        print(f"\n❌ API key validation failed: {e}")
        print("\n💡 Please check:")
        print("   - Your API key is correct in the .env file")
        print("   - Your API key has sufficient credits")
        print("   - You have access to the embedding models")
        return False

def get_embedding(text, model='text-embedding-3-small'):
    """Get embedding for text using OpenAI API"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def init_db(conn) -> None:

    # Create an empty DataFrame with specified columns and data types
    df = pd.DataFrame(columns=['file', 'part', 'modified', 'embedding'])

    # Set the data types for each column
    df['file'] = df['file'].astype('str')
    df['part'] = df['part'].astype('int64')
    df['modified'] = df['modified'].astype('float64')
    df['embedding'] = df['embedding'].astype('str')

    df.to_sql('vault', conn, index=False)

def load_df() -> pd.DataFrame:
    # Load embeddings database and clean up stale entries
    conn = sqlite3.connect(EMBEDDINGS_DB)
    print("🔍 Checking for stale entries...")
    
    try:
        # Check if the 'vault' table exists and init if needed
        existing_tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vault';").fetchall()
        if not existing_tables:
            init_db(conn)

        # Read the 'vault' table into a DataFrame
        df = pd.read_sql('select * from vault', conn)
    finally:
        conn.close()

    if df.empty:
        return df
        
    # Convert the 'modified' column to a timestamp
    df['modified'] = df['modified'].astype('float64')

    # Remove files that have been deleted or modified
    valid_mask = []
    removed_files = []
    
    with alive_bar(len(df), title='Checking files') as bar:
        for _, row in df.iterrows():
            if not os.path.exists(row['file']) or os.path.getmtime(row['file']) > row['modified']:
                removed_files.append(row['file'])
                valid_mask.append(False)
            else:
                valid_mask.append(True)
            bar()
    
    if removed_files:
        print(f"🗑️  Removing {len(removed_files)} stale entries")
        with alive_bar(len(removed_files), title='Removing stale') as bar:
            for file in removed_files:
                filename = os.path.basename(file)
                print(f"   - {filename}")
                bar()
    else:
        print("✅ No stale entries found")
    
    df = df[valid_mask].reset_index(drop=True)

    # Convert JSON strings back to numpy arrays
    if not df.empty:
        df["embedding"] = df['embedding'].apply(json.loads).apply(np.array)

    return df


def save_df(df) -> None:
    # Save DataFrame to SQLite database
    df_copy = df.copy()
    
    # Convert arrays to JSON strings for SQLite storage
    df_copy['embedding'] = df_copy['embedding'].apply(lambda x: json.dumps(x.tolist() if hasattr(x, 'tolist') else x))
    df_copy['modified'] = df_copy['modified'].astype(str)
    
    conn = sqlite3.connect(EMBEDDINGS_DB)
    try:
        df_copy.to_sql('vault', conn, if_exists='replace', index=False)
    finally:
        conn.close()


def get_chunks(text):
    # Split text into token chunks that fit within TOKEN_LIMIT
    chunks = []
    current_chunk = []
    encoding = tiktoken.get_encoding('cl100k_base')
    encoded = encoding.encode(text)
    for i, token in enumerate(encoded):
        current_chunk.append(token)
        if (
            len(current_chunk) >= TOKEN_LIMIT
            or i == len(encoded) - 1
        ):
            chunks.append(current_chunk)
            current_chunk = []
    return chunks


def build():
    # Build embeddings for new markdown files
    print("🔍 Scanning for markdown files...")
    encoding = tiktoken.get_encoding('cl100k_base')

    print("📊 Loading existing database...")
    existing_df = load_df()
    print(f"✅ {len(existing_df)} files already indexed")

    # Find all markdown files in configured directories
    print("🔎 Searching configured directories...")
    files = []
    for dir in DIRS:
        dir_files = glob.glob(f"{dir}/**/*.md", recursive=True)
        print(f"   Found {len(dir_files)} files in {dir}")
        files.extend(dir_files)
    print(f"📝 {len(files)} total markdown files found")

    # Filter out already indexed files
    existing_files = set(existing_df['file']) if not existing_df.empty else set()
    files = list(set(files) - existing_files)
    print(f"🆕 {len(files)} files need embeddings")

    if not files:
        print("✨ All files are already indexed!")
        return

    file_chunks = []
    file_parts = []
    mod_timestamps = []
    embeddings = []

    print(f"🚀 Processing {len(files)} files and generating embeddings...")
    with alive_bar(len(files), title='Processing files') as file_bar:
        for f in files:
            filename = os.path.basename(f)
            file_bar.text(f'Processing: {filename}')
            try:
                with open(f, "r", encoding='utf-8') as file:
                    mtime = os.path.getmtime(f)
                    chunks = get_chunks(file.read())
                    
                    for j, chunk in enumerate(chunks):
                        content = encoding.decode(chunk)
                        if len(chunks) > 1:
                            file_bar.text(f'Processing: {filename} - chunk {j+1}/{len(chunks)}')
                        embeddings.append(get_embedding(content))
                        file_chunks.append(f)
                        file_parts.append(j)
                        mod_timestamps.append(mtime)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
            file_bar()

    if not embeddings:
        print("⚠️  No embeddings generated")
        return

    print(f"📋 Creating DataFrame with {len(embeddings)} embeddings...")
    df = pd.DataFrame({
        'file': file_chunks,
        'part': file_parts,
        'modified': mod_timestamps,
        'embedding': embeddings,
    })

    if not existing_df.empty:
        print("🔗 Merging with existing data...")
        df = pd.concat([existing_df, df], ignore_index=True)

    print("💾 Saving to database...")
    save_df(df)
    print(f"✅ Database updated! Total entries: {len(df)}")


def enhance_query(query):
    """Enhance query using GPT for better search results"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use mini for cost efficiency
            messages=[
                {
                    "role": "system", 
                    "content": "You are a query enhancement assistant. Expand the user's search query to include related terms, synonyms, and alternative phrasings that would help find relevant content in a knowledge base. Return only the enhanced query, no explanations."
                },
                {
                    "role": "user", 
                    "content": f"Enhance this search query for better results: {query}"
                }
            ],
            temperature=0.3,
            max_tokens=100
        )
        content = response.choices[0].message.content
        enhanced = content.strip() if content else query
        return enhanced if enhanced else query
    except Exception as e:
        print(f"⚠️ Query enhancement failed: {e}")
        return query

def hybrid_search(df, query, semantic_weight=0.7, bm25_weight=0.3, top_k=10, similarity_threshold=0.1):
    """Combine semantic search with BM25 keyword search"""
    if df.empty:
        return pd.DataFrame()
    
    # Enhance the query first
    enhanced_query = enhance_query(query)
    if enhanced_query != query:
        print(f"🔍 Enhanced query: '{enhanced_query}'")
    
    # Semantic search
    query_embedding = get_embedding(enhanced_query)
    semantic_scores = df['embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    
    # Prepare text corpus for BM25
    corpus = []
    for _, row in df.iterrows():
        try:
            with open(row['file'], 'r', encoding='utf-8') as file:
                content = file.read()
                # Simple tokenization for BM25
                tokens = content.lower().split()
                corpus.append(tokens)
        except Exception:
            corpus.append([])  # Empty document if can't read
    
    # BM25 search
    if corpus:
        bm25 = BM25Okapi(corpus)
        query_tokens = enhanced_query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores.tolist()) if len(bm25_scores) > 0 and max(bm25_scores.tolist()) > 0 else 1
        bm25_scores = [score / max_bm25 for score in bm25_scores]
    else:
        bm25_scores = [0] * len(df)
    
    # Combine scores
    df = df.copy()
    df['semantic_score'] = semantic_scores
    df['bm25_score'] = bm25_scores
    df['hybrid_score'] = (semantic_weight * df['semantic_score'] + 
                         bm25_weight * df['bm25_score'])
    df['similarity'] = df['hybrid_score']  # Keep compatibility
    
    # Filter by threshold and return top results
    filtered_df = df[df['hybrid_score'] >= similarity_threshold]
    results = filtered_df.sort_values("hybrid_score", ascending=False).head(top_k)
    
    return results

def search(df, query):
    """Main search function - now uses hybrid search"""
    return hybrid_search(df, query)

def get_chat_response(query, search_results, chat_history=None):
    """Generate a helpful response using GPT-4o based on search results and chat history"""
    if search_results.empty:
        return "I couldn't find any relevant information in your knowledge base. Please try a different search term or add more content to your indexed directories."
    
    # Build context from search results
    context_parts = []
    for i, (_, row) in enumerate(search_results.iterrows(), 1):
        try:
            with open(row['file'], 'r', encoding='utf-8') as file:
                content = file.read()
                filename = os.path.basename(row['file'])
                context_parts.append(f"Source {i} ({filename}, similarity: {row['similarity']:.3f}):\n{content}\n")
        except Exception as e:
            print(f"⚠️  Could not read {row['file']}: {e}")
    
    if not context_parts:
        return "I found relevant files but couldn't read their contents. Please check file permissions."
    
    context = "\n---\n".join(context_parts)
    
    # Create the prompt for GPT-4o
    system_prompt = """You are a helpful AI assistant that answers questions based on the user's personal knowledge base from their Obsidian vault. 

Provide comprehensive, well-structured answers using the provided context. When referencing information, mention which source it comes from. If the context doesn't fully answer the question, clearly state what information is missing.

Maintain conversation continuity by referencing previous exchanges when relevant. Format your response in a clear, readable way with proper markdown formatting when helpful."""
    
    # Build messages with chat history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history if available
    if chat_history:
        messages.extend(chat_history)
    
    # Add current query with context
    user_prompt = f"""Question: {query}

Relevant context from knowledge base:
{context}

Please provide a helpful answer based on this context."""
    
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,  # type: ignore
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error generating a response: {e}"

def print_separator(char="═", length=80):
    """Print a visual separator"""
    print(char * length)

def print_header(text, char="═"):
    """Print a formatted header"""
    print(f"\n{char * 3} {text} {char * 3}")

def show_progress(message):
    """Show progress indicator while processing"""
    import time
    import itertools
    import threading
    
    spinner = itertools.cycle(['▘', '▝', '▗', '▖'])
    stop_spinner = threading.Event()
    
    def spin():
        while not stop_spinner.is_set():
            print(f'\r{message} {next(spinner)}', end='', flush=True)
            time.sleep(0.2)
    
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.daemon = True
    spinner_thread.start()
    
    return stop_spinner

def print_ai_response(content, conversation_count):
    """Print AI response with clear formatting"""
    print(f"\n🤖 AI Response #{conversation_count}:")
    print("─" * 50)
    print(content)
    print("─" * 50)

def main():
    print_separator()
    print("🤖 OBSIDIAN SEMANTIC SEARCH SYSTEM")
    print_separator()
    
    # Validate API key before proceeding
    if not validate_api_key():
        sys.exit(1)
    
    build()
    
    print("\n🔄 Loading search index...")
    df = load_df()
    print(f"📚 Ready! Loaded {len(df)} indexed chunks")
    
    print_header("CHAT INTERFACE")
    print("🎯 Ask me anything about your knowledge base!")
    print("💡 Tips: Be specific in your questions for better results")
    print("⌨️  Commands: /exit to quit • /new to start fresh conversation\n")
    
    # Initialize chat history
    chat_history = []
    conversation_count = 0
    
    while True:
        user_input = input("💬 You: ").strip()
        
        if user_input.lower() in ['/exit', '/quit', '/q']:
            print_separator("─")
            print("👋 Goodbye!")
            sys.exit()
            
        if user_input.lower() in ['/new', '/reset']:
            chat_history = []
            conversation_count = 0
            print_separator("─")
            print("🆕 Started new conversation")
            print_separator("─")
            continue
            
        if not user_input:
            continue
            
        conversation_count += 1
        
        print_separator("─")
        print(f"🔎 Searching your knowledge base...")
        search_results = search(df, user_input)
        
        # Show progress while generating response
        progress_stop = show_progress("🤖 Generating response")
        ai_response = get_chat_response(user_input, search_results, chat_history)
        progress_stop.set()
        print('\r' + ' ' * 50 + '\r', end='')  # Clear progress line
        
        print_ai_response(ai_response, conversation_count)
        
        if not search_results.empty:
            print(f"\n📚 Sources: {len(search_results)} files consulted")
            for i, (_, row) in enumerate(search_results.iterrows(), 1):
                filename = os.path.basename(row['file'])
                print(f"   {i}. {filename} (similarity: {row['similarity']:.3f})")
        
        # Add to chat history (keep last 6 exchanges to manage context)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": ai_response})
        
        # Trim history to prevent context overflow
        if len(chat_history) > 12:  # Keep last 6 user-assistant pairs
            chat_history = chat_history[-12:]
        
        print_separator("─")
        print(f"💭 Conversation #{conversation_count} • Type /new to reset chat\n")

if __name__ == '__main__':
    main()
