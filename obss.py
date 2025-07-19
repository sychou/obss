from alive_progress import alive_bar
from dotenv import load_dotenv
import glob
import json
import numpy as np
from openai import OpenAI
import os
import pandas as pd
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
        test_response = client.embeddings.create(
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


def search(df, query):
    # Search for similar content using cosine similarity
    if df.empty:
        return pd.DataFrame()
        
    query_embedding = get_embedding(query)
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    results = (
        df.sort_values("similarity", ascending=False)
        .head(5)
    )
    return results

def get_chat_response(query, search_results):
    """Generate a helpful response using GPT-4o based on search results"""
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

Format your response in a clear, readable way with proper markdown formatting when helpful."""
    
    user_prompt = f"""Question: {query}

Relevant context from knowledge base:
{context}

Please provide a helpful answer based on this context."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error generating a response: {e}"

def main():
    print("🤖 Obsidian Semantic Search System")
    print("=" * 35)
    
    # Validate API key before proceeding
    if not validate_api_key():
        sys.exit(1)
    
    build()
    
    print("\n🔄 Loading search index...")
    df = load_df()
    print(f"📚 Ready! Loaded {len(df)} indexed chunks")
    print("\n🎯 Ask me anything about your knowledge base!")
    print("💡 Tips: Be specific in your questions for better results")
    print("⌨️  Commands: /exit to quit\n")
    
    while True:
        user_input = input("💬 Ask me: ")
        
        if user_input.lower() in ['/exit', '/quit', '/q']:
            print("👋 Goodbye!")
            sys.exit()
            
        print(f"\n🔎 Searching your knowledge base...")
        search_results = search(df, user_input)
        
        print("🤖 Generating response...\n")
        ai_response = get_chat_response(user_input, search_results)
        
        print("📝 **Response:**")
        print(ai_response)
        
        if not search_results.empty:
            print(f"\n📚 **Sources consulted:** {len(search_results)} files")
            for i, (_, row) in enumerate(search_results.iterrows(), 1):
                filename = os.path.basename(row['file'])
                print(f"   {i}. {filename} (similarity: {row['similarity']:.3f})")
        
        print("\n" + "─" * 60)

if __name__ == '__main__':
    main()
