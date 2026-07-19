# Colombia Financial RAG 📊

RAG (Retrieval-Augmented Generation) system for querying annual reports of Colombian companies listed on the Bolsa de Valores de Colombia (BVC).

## Companies covered

| Company | Report |
|---------|--------|
| 📘 Bancolombia | Annual Report 2024 |
| 🛢️ Ecopetrol | Annual Report 2024 |
| 🏦 Grupo Sura | Annual Report 2024 |
| 🏗️ Cemargos | Annual Report 2024 |
| ⚡ Celsia | Annual Report 2023 |

## Architecture

```
PDF Reports (5 companies)
        ↓ pdfplumber
    718 chunks (3-page sliding window, overlap=1)
        ↓
   ┌────────────────────────────────┐
   │        Search Engine           │
   │  ┌──────────┐ ┌─────────────┐ │
   │  │ Keyword  │ │   Vector    │ │
   │  │ minsearch│ │ ONNX SBERT  │ │
   │  └────┬─────┘ └──────┬──────┘ │
   │       └──────┬────────┘        │
   │          Hybrid RRF            │
   └──────────────┬─────────────────┘
                  ↓
         Top-5 relevant chunks
                  ↓
        Groq (llama-3.3-70b-versatile)
                  ↓
            Answer + Sources
                  ↓
         SQLite (conversation history)
```

## Tech stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | `all-MiniLM-L6-v2` via ONNX (no GPU required) |
| Keyword search | `minsearch` (TF-IDF) |
| Vector search | NumPy cosine similarity |
| Hybrid search | Reciprocal Rank Fusion (RRF) |
| PDF parsing | `pdfplumber` |
| UI | Streamlit |
| Database | SQLite |

## Setup

```bash
# Clone the repo
git clone https://github.com/graciano90210/colombia-financial-rag.git
cd colombia-financial-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Download the ONNX model (first time only)
# Place all-MiniLM-L6-v2 under models/Xenova/all-MiniLM-L6-v2/

# Add PDF reports under data/reports/ and run ingestion
python src/ingest.py

# Launch the app
streamlit run app.py
```

## Features

- **3 search modes**: keyword, vector, and hybrid (RRF) — selectable from the UI
- **Company filter**: query all companies or a specific one
- **Source transparency**: see exactly which PDF pages were used to generate the answer
- **Conversation history**: all queries saved to SQLite and shown in the UI
- **Token usage tracking**: input and output tokens displayed per query

## Sample queries

- `¿Cuál fue la utilidad neta de Bancolombia en 2024?`
- `¿Cuáles son los principales riesgos que enfrenta Ecopetrol?`
- `¿Cuál es la estrategia de transición energética de Celsia?`
- `¿Cómo evolucionó el EBITDA de Cemargos en 2024?`
- `¿Cuál es la política de dividendos de Grupo Sura?`
