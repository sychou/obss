# OBSS - Obsidian Semantic Search System

An AI-powered semantic search system for your Obsidian vault that combines OpenAI embeddings with GPT-4o to provide intelligent, conversational answers to your questions.

## Features

- 🔍 **Semantic Search**: Find relevant content across your entire - Obsidian vault using OpenAI embeddings
- 🤖 **AI-Powered Responses**: Get comprehensive answers powered by - GPT-4o based on your knowledge base
- 💬 **Conversational Chat**: Follow-up questions with maintained - context across conversations
- 📚 **Multi-file Context**: Searches across multiple files and - provides source references
- ⚡ **Incremental Indexing**: Only processes new or modified files - to save time and API costs
- 🎯 **Smart Chunking**: Handles large files by splitting them into - optimal token chunks
- ✨ **Enhanced UI**: Visually distinct responses with borders, - headers, and clear formatting
- 🔄 **Session Management**: Start fresh conversations anytime with - slash commands

## Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/sychou/obss.git
   cd obss
   ```

2. **Install dependencies using uv (recommended)**
   ```bash
   uv sync
   ```
   
   Or with pip:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_KEY=your_openai_api_key_here
   TOKEN_LIMIT=8000
   EMBEDDINGS_DB=data/embeddings.db
   DIRS=["./Users/your-name/path/to/obsidian/vault"]
   ```

4. **Configure your Obsidian directories**
   Update the `DIRS` environment variable with paths to your Obsidian vault folders containing markdown files.

## Usage

Run the system:
```bash
python obss.py
```

The system will:
1. Validate your OpenAI API key
2. Scan for markdown files in configured directories
3. Build embeddings for new files (this may take time on first run)
4. Start the interactive chat interface

### Interactive Commands

- Ask any question about your knowledge base
- Ask follow-up questions - the system maintains conversation context
- Use `/new` or `/reset` to start a fresh conversation
- Use `/exit`, `/quit`, or `/q` to quit
- The AI will search your vault and provide contextual answers with source references

### Example Session

```
════════════════════════════════════════════════════════════════════════════════
🤖 OBSIDIAN SEMANTIC SEARCH SYSTEM
════════════════════════════════════════════════════════════════════════════════
🔑 Validating OpenAI API key... ✅
🔍 Scanning for markdown files...
📊 Loading existing database...
✅ 245 files already indexed
🆕 12 files need embeddings
🚀 Processing 12 files and generating embeddings...
📚 Ready! Loaded 1,847 indexed chunks

═══ CHAT INTERFACE ═══
🎯 Ask me anything about your knowledge base!
💡 Tips: Be specific in your questions for better results
⌨️  Commands: /exit to quit • /new to start fresh conversation

💬 You: What are the key principles of effective note-taking?

────────────────────────────────────────────────────────────────────────────────
🔎 Searching your knowledge base...
🤖 Generating response ⠠

🤖 AI Response #1:
──────────────────────────────────────────────────────
Based on your notes, here are the key principles of effective note-taking:

1. **Active Processing**: Don't just copy information verbatim. Rephrase 
   concepts in your own words to ensure understanding...

2. **Structure and Organization**: Use consistent formatting and hierarchy 
   to make notes scannable and retrievable...

3. **Regular Review**: Schedule periodic reviews of your notes to reinforce 
   learning and identify knowledge gaps...
──────────────────────────────────────────────────────

📚 Sources: 3 files consulted
   1. Note-taking Methods.md (similarity: 0.892)
   2. Learning Principles.md (similarity: 0.834)
   3. Knowledge Management.md (similarity: 0.801)

────────────────────────────────────────────────────────────────────────────────
💭 Conversation #1 • Type /new to reset chat

💬 You: Can you elaborate on the active processing part?

────────────────────────────────────────────────────────────────────────────────
🔎 Searching your knowledge base...
🤖 Generating response ⠂

🤖 AI Response #2:
──────────────────────────────────────────────────────
Absolutely! Building on what I mentioned about active processing in your 
note-taking, here are the key aspects:

**Paraphrasing**: Instead of copying text directly, rewrite concepts using 
your own vocabulary. This forces deeper engagement with the material...

**Question Generation**: As you take notes, formulate questions about the 
content. This helps identify areas that need further exploration...
──────────────────────────────────────────────────────
```

## Configuration

### Environment Variables

- `OPENAI_KEY`: Your OpenAI API key
- `TOKEN_LIMIT`: Maximum tokens per chunk (default: 8000)
- `EMBEDDINGS_DB`: Path to SQLite database for embeddings storage
- `DIRS`: JSON array of directories to scan for markdown files

### Dependencies

The project uses modern Python dependency management with `pyproject.toml`:

- `openai`: OpenAI API client
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `tiktoken`: Token counting and encoding
- `alive-progress`: Progress bars
- `python-dotenv`: Environment variable loading

## How It Works

1. **File Discovery**: Recursively scans configured directories for `.md` files
2. **Change Detection**: Tracks file modification times to avoid reprocessing unchanged files
3. **Text Chunking**: Splits large files into token-limited chunks for optimal embedding
4. **Embedding Generation**: Creates vector embeddings using OpenAI's `text-embedding-3-small` model
5. **Semantic Search**: Finds most similar content using cosine similarity
6. **Conversational AI**: Uses GPT-4o with chat history to generate contextual responses
7. **Context Management**: Maintains conversation history (last 6 exchanges) for follow-up questions

## Requirements

- Python 3.12+
- OpenAI API key with access to embedding and chat models
- Obsidian vault with markdown files


