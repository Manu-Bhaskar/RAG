# Local Corrective RAG Assistant 🧠

A fully local Retrieval-Augmented Generation (RAG) system with Corrective Retrieval (CRAG) for intelligent document Q&A using LLMs. This project combines **LangChain**, **ChromaDB**, **Ollama**, and **Streamlit** to create a privacy-focused AI assistant that runs entirely on your machine.

## 🎯 Features

- **Local-First Architecture**: No API calls, all processing happens locally
- **Corrective RAG (CRAG)**: Intelligently grades retrieved context relevance and adjusts responses accordingly
- **Multiple Interfaces**:
  - 🌐 **Web UI** (Streamlit) - user-friendly document upload and chat
  - 💻 **CLI Interface** - command-line chatbot with CRAG
- **Flexible Document Support**: Upload PDFs and TXT files
- **ArXiv Integration**: Automatically download ML/AI papers for your knowledge base
- **Performance Metrics**: BLEU score evaluation for generated answers
- **Vector Database**: ChromaDB for efficient semantic search
- **Advanced Embeddings**: SentenceTransformer models for high-quality embeddings
- **LLM Flexibility**: Powered by Ollama for local LLM inference

## 📋 Architecture

### Components

1. **Document Ingestion** (`ingest.py`)
   - Loads PDFs and TXT files from `data/docs`
   - Chunks documents using RecursiveCharacterTextSplitter
   - Creates vector embeddings using SentenceTransformer
   - Stores in ChromaDB for fast retrieval

2. **ArXiv Paper Downloader** (`py.py`)
   - Automatically downloads ML/DL/AI papers from ArXiv
   - Targets 100+ papers on machine learning, transformers, neural networks
   - Populates the knowledge base automatically

3. **Web Interface** (`app.py`)
   - Streamlit-based UI for document management
   - File upload for PDFs and TXT files
   - Chat interface for asking questions
   - Real-time BLEU score evaluation
   - Results logging to `qa_bleu_log.json`

4. **CLI Chatbot** (`chat.py`)
   - Command-line CRAG implementation
   - Context relevance grading
   - Fallback responses when context is weak
   - Interactive chat loop

### How CRAG Works

The Corrective RAG system evaluates retrieved context relevance:

1. **Retrieve**: Query the vector DB to get relevant documents
2. **Grade**: Use LLM to evaluate if retrieved context is relevant to the question
3. **Generate**: If relevant → answer from context; If not → generate fallback response
4. **Evaluate**: Calculate BLEU score for answer quality

This ensures responses are grounded in retrieved documents and gracefully handles insufficient context.

## 📦 Requirements

- **Python 3.8+**
- **Ollama** (for local LLM inference) - [Download](https://ollama.ai)
- **Llama 3.2** model (pulled in Ollama)
- At least 8GB RAM recommended
- GPU optional but recommended for faster inference

## 🚀 Installation

### 1. Clone/Setup Project
```bash
cd d:\projects\RAG
```

### 2. Create Virtual Environment (if not already created)
```bash
python -m venv rag_env
```

### 3. Activate Virtual Environment
**Windows:**
```bash
rag_env\Scripts\activate
```
**macOS/Linux:**
```bash
source rag_env/bin/activate
```

### 4. Install Dependencies
```bash
pip install streamlit langchain langchain-community chromadb sentence-transformers nltk ollama arxiv
```

### 5. Setup Ollama
- Install Ollama from [ollama.ai](https://ollama.ai)
- Pull the llama3.2 model:
```bash
ollama pull llama3.2
```
- Start Ollama service (usually runs on `localhost:11434`)

## ⚙️ Configuration

Key settings in each file:

```python
# Embedding model (all-MiniLM-L6-v2 is lightweight & fast)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM model (requires Ollama)
LLM_MODEL = "llama3.2"

# Document storage
DATA_PATH = "./data/docs"      # Where documents are stored
DB_PATH = "./vector_db"         # Where ChromaDB persists

# Chunk settings (in ingest.py)
chunk_size = 500                # Document chunk size
chunk_overlap = 50              # Overlap between chunks
```

## 📖 Usage

### Option 1: Web Interface (Streamlit)
```bash
streamlit run app.py
```
- Opens in browser at `http://localhost:8501`
- Upload PDFs/TXT files in sidebar
- Click "Add to Vector Database" to ingest
- Ask questions in the main chat area

### Option 2: CLI Chatbot
```bash
python chat.py
```
- Interactive command-line interface
- Type questions and get immediate responses
- Type `exit` or `quit` to exit

### Option 3: Download ArXiv Papers
Populate your knowledge base with ML papers:
```bash
python py.py
```
- Downloads up to 100 papers on ML/DL/AI topics
- Saves to `data/docs`
- Then run `python ingest.py` to add them to the vector DB

### Option 4: Create/Update Vector Database
After adding new documents:
```bash
python ingest.py
```
- Processes all PDFs/TXT in `data/docs`
- Recreates vector DB
- Note: Clears existing DB before rebuilding

## 📁 Project Structure

```
RAG/
├── app.py                    # Streamlit web interface
├── chat.py                   # CLI CRAG chatbot
├── ingest.py                 # Vector DB creation
├── py.py                     # ArXiv paper downloader
├── data/
│   ├── docs/                 # Document storage (PDFs/TXT)
│   └── wiki_stem_corpus.csv  # Pre-loaded corpus
├── vector_db/                # ChromaDB persistence
│   └── chroma.sqlite3        # Vector database file
├── qa_bleu_log.json          # Q&A evaluation results
└── rag_env/                  # Python virtual environment
```

## 🔄 Workflow

### First Time Setup
1. Install dependencies
2. Setup Ollama and pull llama3.2
3. Run `py.py` to download ArXiv papers (optional)
4. Run `ingest.py` to create vector database
5. Run `app.py` (web) or `chat.py` (CLI)

### Regular Usage
1. Upload new documents via web UI or add to `data/docs`
2. Run `ingest.py` to update vector database
3. Start chatting via `app.py` or `chat.py`

## 📊 Evaluation

The system logs Q&A pairs with BLEU scores to `qa_bleu_log.json`:

```json
[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is a subset of AI...",
    "bleu_score": 0.45
  }
]
```

BLEU scores help evaluate answer quality (0-1 scale).

## ⚡ Performance Tips

- **Smaller chunks**: Use `chunk_size=250` for more precise retrieval
- **More overlap**: Increase `chunk_overlap=100` to keep context
- **GPU Acceleration**: Set `CUDA_VISIBLE_DEVICES` if using GPU with Ollama
- **Batch Ingestion**: Large document sets are added in batches of 1000

## 🔍 Troubleshooting

**Issue**: Ollama connection error
- Ensure Ollama is running: `ollama serve`
- Check Ollama is accessible at `localhost:11434`

**Issue**: Out of memory
- Reduce `chunk_size` or use lighter model
- Process documents in smaller batches

**Issue**: Slow responses
- Enable GPU support in Ollama
- Use a faster LLM model (e.g., `mistral` or `neural-chat`)

**Issue**: Poor answer quality
- Add more relevant documents to knowledge base
- Adjust prompt templates in `chat.py` or `app.py`
- Lower BLEU score threshold for better results

## 🛠️ Customization

- **Change LLM**: Modify `LLM_MODEL` to use different Ollama models
- **Change Embeddings**: Use different SentenceTransformer models
- **Adjust Prompts**: Edit prompt templates in `chat.py` for different behavior
- **Chunk Settings**: Tune `chunk_size` and `chunk_overlap` for your documents

## 📝 Notes

- All processing happens locally - no data sent to external APIs
- Requires sufficient disk space for vector DB and documents
- Ollama must be running in the background
- First run may be slow as models are loaded into memory

## 🤝 Dependencies

- **LangChain**: LLM orchestration and RAG pipeline
- **ChromaDB**: Vector database and similarity search
- **SentenceTransformers**: Semantic embeddings
- **Ollama**: Local LLM inference engine
- **Streamlit**: Web UI framework
- **NLTK**: Natural language processing (BLEU scoring)
- **ArXiv**: Paper downloading

## 📄 License

This is a local RAG system project.

## 🎓 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Models](https://ollama.ai/library)
- [RAG Concepts](https://python.langchain.com/docs/use_cases/question_answering/)