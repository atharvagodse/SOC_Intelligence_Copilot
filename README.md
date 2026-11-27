# 🛡️ Enterprise SOC Intelligence Copilot

> An AI-powered cybersecurity knowledge assistant that helps SOC analysts investigate security alerts using Retrieval-Augmented Generation (RAG).

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-orange)](https://www.trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-local-purple)](https://ollama.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)](https://streamlit.io)

---

## 🎯 What is this project?

This application answers cybersecurity questions grounded in **your own uploaded security documents** — NIST frameworks, MITRE ATT&CK guides, incident response playbooks, threat intelligence reports — rather than relying solely on the LLM's pre-trained knowledge.

**RAG in a nutshell:**
```
Your Question → Find relevant document chunks → Feed chunks + question to LLM → Grounded answer
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 DOCUMENT INGESTION PIPELINE                  │
│  Upload PDF/DOCX/TXT → Extract Text → Clean → Chunk         │
│  → Generate Embeddings → Store in ChromaDB                   │
└─────────────────────────────────────────────────────────────┘
                              ↕ vector_db/
┌─────────────────────────────────────────────────────────────┐
│                    RAG QUERY PIPELINE                         │
│  User Question → Embed Query → Retrieve Top-K Chunks         │
│  → Build Prompt → Ollama LLM → Grounded Answer + Sources    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SOC-Intelligence-Copilot/
├── app/
│   └── streamlit_app.py        # Streamlit UI (Phase 9)
├── data/
│   ├── raw/                    # Source security documents
│   └── processed/              # Cleaned text cache
├── vector_db/                  # ChromaDB persistent store
├── src/
│   ├── config.py               # Centralised settings (Phase 1)
│   ├── embeddings.py           # Embedding model wrapper (Phase 4)
│   ├── vector_store.py         # ChromaDB interface (Phase 5)
│   ├── retrieval.py            # Semantic search (Phase 6)
│   ├── generation.py           # Ollama LLM calls (Phase 7)
│   ├── prompts.py              # Prompt templates (Phase 8)
│   └── ingestion/
│       ├── loaders.py          # PDF/DOCX/TXT loaders (Phase 2)
│       └── chunking.py         # Text splitters (Phase 3)
├── evaluation/
│   └── test_questions.json     # RAG evaluation dataset (Phase 11)
├── tests/
│   └── test_retrieval.py       # Pytest test suite
├── .env.example                # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.12.x | Required |
| pip | latest | Included with Python |
| Ollama | latest | For local LLM inference |

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SOC-Intelligence-Copilot
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Hardware note:** This installs `sentence-transformers` which downloads the
> `all-MiniLM-L6-v2` model (~90 MB) on first run. It is CPU-compatible and
> works with as little as 500 MB of RAM.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to change defaults (optional for local dev)
```

### 5. Install and start Ollama

```bash
# Download from https://ollama.ai/download
# Then pull the lightweight model (630 MB, requires ~2 GB RAM):
ollama pull llama3.2:1b

# Verify Ollama is running:
ollama list
```

### 6. Run the application

```bash
streamlit run app/streamlit_app.py
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## 🗺️ Development Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Project setup & environment | ✅ Complete |
| **2** | Document loading (PDF, DOCX, TXT) | ✅ Complete |
| **3** | Text cleaning & chunking | ✅ Complete |
| **4** | Embedding generation | ✅ Complete |
| **5** | ChromaDB integration | ✅ Complete |
| **6** | Semantic search & retrieval | 🔜 Next |
| **7** | Ollama LLM integration | ⏳ Planned |
| **8** | Complete RAG pipeline | ⏳ Planned |
| **9** | Streamlit interface | ⏳ Planned |
| **10** | Citations & conversation history | ⏳ Planned |
| **11** | RAG evaluation metrics | ⏳ Planned |
| **12** | Security hardening & Docker | ⏳ Planned |

---

## 🔒 Security Notes

- All uploaded documents are validated (type + size) before processing.
- Prompt injection via document content is mitigated through sanitised context injection.
- System prompts are never exposed to the UI.
- Sensitive document contents are never logged.
- The assistant provides **advisory guidance only** and does not execute security actions.

---

## 🤝 Contributing

This project is built incrementally as a learning portfolio. Each phase is independently testable before the next begins.

---

## 📜 Licence

MIT Licence — see [LICENSE](LICENSE) for details.
