🧠 Orchestrated RAG
High-Accuracy Document Q&A System | CPU-Friendly | Multi-LLM Support
Orchestrated RAG is a production-grade Retrieval-Augmented Generation system designed for maximum accuracy in answering questions over your documents. It comes with a sleek dark-mode web UI featuring real-time streaming answers, full pipeline visibility, and persistent conversation history.

Python 3.10+ FastAPI LlamaIndex License: MIT

📁 Project Structure
rag_project/
├── docs/                ← Drop your documents here (.pdf, .txt, .md, .docx, .csv)
├── qdrant_db/           ← Auto-created Qdrant vector store (persists on disk)
├── graph_cache/         ← Cache for future GraphRAG features
├── chat_history.json    ← Persistent conversation history (auto-created)
│
├── config.py            ← Configuration (models, retrieval params, API keys)
├── .env                 ← Your API keys (not committed to git)
│
├── ingest.py            ← Document ingestion and indexing pipeline
├── pipeline.py          ← Core RAG orchestration logic (9-stage pipeline)
├── query.py             ← Terminal-based interactive Q&A interface
├── server.py            ← FastAPI web server with SSE streaming
├── app.html             ← Glassmorphism dark-mode web UI
├── tests.py             ← Comprehensive test suite
│
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
✨ Key Features
RAG Pipeline
Hybrid Retrieval — Combines vector search (Qdrant) + BM25 keyword search
Contextual Enrichment — LLM-generated context per chunk (35–49% retrieval improvement)
Query Decomposition — Splits complex queries into sub-questions
HyDE Expansion — Hypothetical Document Embeddings for better recall
RAG-Fusion — Reciprocal rank fusion across multiple retrieval strategies
Cohere Reranking — Neural reranking for top-k selection
Cross-Encoder Compression — Sentence-level context filtering
CRAG Gate — Intelligent web search fallback when docs are insufficient
Self-Critique — Hallucination detection and correction
Web UI
Real-Time Streaming — Token-by-token answer delivery via SSE
Pipeline Diagnostics — Live stage-by-stage progress indicators
Conversation History — Persistent chat sessions with rename/delete
Source Attribution — Inline source cards with file, page, and relevance scores
Dark Glassmorphism Design — Premium UI with Inter font, subtle animations
Responsive — Works on desktop and mobile
Developer Experience
Local Embeddings — BAAI/bge-large-en-v1.5 (CPU-only, no API required)
CPU-Friendly — Optimized for consumer hardware (i5 13th gen or equivalent)
Multi-LLM Support — Groq (fast routing), OpenRouter / NVIDIA (final generation)
Structured Parsing — Docling for table-aware document processing
🚀 Quick Start
1. Install Dependencies
pip install -r requirements.txt
Optional — For advanced document parsing (tables, structured layouts):

pip install docling llama-index-readers-docling llama-index-node-parser-docling
2. Set Up API Keys
Get free API keys from these services:

Service	URL	Purpose	Free Tier
Groq	https://console.groq.com	Fast LLM for routing/HyDE/CRAG	30 RPM
OpenRouter	https://openrouter.ai	Final answer generation	Pay-per-use
Cohere	https://dashboard.cohere.com	Document reranking	10k/month
Tavily	https://app.tavily.com	Web search fallback	1000/month
Create a .env file in the project root:

GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
COHERE_API_KEY=your_cohere_key_here
TAVILY_API_KEY=your_tavily_key_here
3. Add Documents
Place your documents in the docs/ folder. Supported formats:

PDF (.pdf)
Word documents (.docx)
Text files (.txt)
Markdown (.md)
CSV (.csv)
4. Ingest Documents
python ingest.py
This parses documents, generates contextual enrichments, and builds the vector index.

5. Launch
Web UI (recommended):

python server.py
Open http://localhost:8000 in your browser.

Terminal mode:

python query.py
Interactive Q&A with rich formatting in the console.

🏗️ Pipeline Architecture
User Query
    │
    ▼
┌─ Query Decomposition ─────────┐  Split complex queries into sub-questions
└────────────┬───────────────────┘
             ▼
┌─ HyDE Expansion ──────────────┐  Generate hypothetical answers for better retrieval
└────────────┬───────────────────┘
             ▼
┌─ Hybrid Retrieval ─────────────┐
│   ├── Vector Search (Qdrant)   │  → top 15 chunks per query
│   └── BM25 Keyword Search      │  → top 10 chunks per query
└────────────┬───────────────────┘
             ▼
┌─ RAG-Fusion ───────────────────┐  Reciprocal rank fusion + source diversity
└────────────┬───────────────────┘
             ▼
┌─ Cohere Rerank ────────────────┐  → top 8 most relevant chunks
└────────────┬───────────────────┘
             ▼
┌─ Context Compression ──────────┐  Cross-encoder sentence filtering
└────────────┬───────────────────┘
             ▼
┌─ CRAG Gate ────────────────────┐  "Sufficient context?"
│   └── No → Tavily web search   │  Merge web results with docs
└────────────┬───────────────────┘
             ▼
┌─ LLM Generation ──────────────┐  Stream answer token-by-token
└────────────┬───────────────────┘
             ▼
┌─ Self-Critique ────────────────┐  Hallucination detection & correction
└────────────┬───────────────────┘
             ▼
         Final Answer
🧪 Testing
Run the full test suite:

python tests.py
Run a specific test:

python tests.py --test P-02
🔧 Configuration
All settings live in config.py:

Setting	Default	Description
GROQ_MODEL	llama-3.1-8b-instant	Fast LLM for HyDE, CRAG, critique
OPENROUTER_MODEL	nvidia/nemotron-3-super-120b-a12b	Final answer generation model
EMBED_MODEL	BAAI/bge-large-en-v1.5	Local CPU embeddings
COHERE_RERANK	rerank-english-v3.0	Reranking model
VECTOR_TOP_K	15	Chunks per vector search
BM25_TOP_K	10	Chunks per BM25 search
RERANK_TOP_N	8	Chunks surviving reranking
CHUNK_SIZE	512	Fallback chunk size
VERBOSE	True	Show pipeline step details
📊 Performance
Metric	Details
Accuracy	Contextual enrichment reduces retrieval failures 35–49%
Speed	~5–10s per query (CPU); ingestion ~2–5 min for large docs
Cost	Leverages free tiers of Groq, Cohere, Tavily
Hardware	Runs on i5 13th gen or equivalent (no GPU required)
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Make your changes
Run tests: python tests.py
Submit a pull request
📄 License
MIT License — see LICENSE file for details.

Built with LlamaIndex, Qdrant, FastAPI, and modern RAG techniques.
