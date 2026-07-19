import json
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from search import SearchEngine
from rag import RAGPipeline
from db import init_db, save_conversation, get_recent_conversations

# ─── Configuración de página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Colombia Financial RAG",
    page_icon="📊",
    layout="wide",
)

# ─── Estilos ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Fondo principal más oscuro */
.stApp {
    background-color: #0a0f1e;
    color: #e0e6f0;
}

/* Sidebar oscuro */
[data-testid="stSidebar"] {
    background-color: #080d1a;
    border-right: 1px solid #00d4ff33;
}

/* Título principal con glow */
h1 {
    color: #00d4ff !important;
    text-shadow: 0 0 20px #00d4ff88;
}

h2, h3 {
    color: #00b8d9 !important;
}

/* Input de texto */
[data-testid="stTextInput"] input {
    background-color: #0d1530 !important;
    border: 1px solid #00d4ff !important;
    border-radius: 8px !important;
    color: #e0e6f0 !important;
    box-shadow: 0 0 8px #00d4ff44;
}
[data-testid="stTextInput"] input:focus {
    box-shadow: 0 0 16px #00d4ffaa !important;
}

/* Botón primario */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #003d66, #0066aa) !important;
    border: 1px solid #00d4ff !important;
    color: #00d4ff !important;
    font-weight: bold;
    box-shadow: 0 0 12px #00d4ff66;
    border-radius: 8px !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 0 20px #00d4ffaa !important;
    background: linear-gradient(135deg, #004d80, #0080cc) !important;
}

/* Cards / Expanders */
[data-testid="stExpander"] {
    background-color: #0d1530 !important;
    border: 1px solid #00d4ff55 !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px #00d4ff22;
}
[data-testid="stExpander"]:hover {
    border-color: #00d4ff !important;
    box-shadow: 0 0 18px #00d4ff44;
}

/* Métricas */
[data-testid="stMetric"] {
    background-color: #0d1530;
    border: 1px solid #00d4ff44;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 0 8px #00d4ff22;
}
[data-testid="stMetricLabel"] {
    color: #00d4ff !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background-color: #0d1530 !important;
    border: 1px solid #00d4ff55 !important;
    color: #e0e6f0 !important;
}

/* Divider */
hr {
    border-color: #00d4ff33 !important;
}

/* Info box */
[data-testid="stAlert"] {
    background-color: #0d1530 !important;
    border: 1px solid #00d4ff44 !important;
    border-radius: 8px !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #00d4ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Carga de modelos (se hace una sola vez) ──────────────────────────────────

@st.cache_resource
def load_pipeline():
    init_db()
    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    engine = SearchEngine(chunks)
    rag = RAGPipeline(engine)
    return rag


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuración")

    company = st.selectbox(
        "🏢 Empresa",
        options=["Todas", "Bancolombia", "Ecopetrol", "Grupo Sura", "Cemargos", "Celsia"],
    )
    company_filter = None if company == "Todas" else company

    search_mode = st.selectbox(
        "🔍 Modo de búsqueda",
        options=["hybrid", "keyword", "vector"],
        format_func=lambda x: {
            "hybrid": "Híbrido (RRF) — Recomendado",
            "keyword": "Keyword (TF-IDF)",
            "vector": "Vectorial (semántico)",
        }[x],
    )

    st.divider()
    st.markdown("""
    ### ℹ️ Acerca de
    RAG sobre reportes anuales de empresas BVC:
    - 📘 Bancolombia 2024
    - 🛢️ Ecopetrol 2024
    - 🏦 Grupo Sura 2024
    - 🏗️ Cemargos 2024
    - ⚡ Celsia 2023

    **Modelo:** llama-3.3-70b-versatile (Groq)
    **Embeddings:** all-MiniLM-L6-v2 (ONNX)
    """)

# ─── Main ─────────────────────────────────────────────────────────────────────

st.title("📊 Colombia Financial RAG")
st.caption("Consulta los reportes anuales de las principales empresas de la BVC")

# Input
question = st.text_input(
    "Escribe tu pregunta:",
    placeholder="¿Cuál fue el EBITDA de Ecopetrol en 2024?",
)

ask_btn = st.button("Preguntar", type="primary", disabled=not question)

# Respuesta
if ask_btn and question:
    rag = load_pipeline()

    with st.spinner("Buscando en los reportes..."):
        result = rag.ask(question, company=company_filter, search_mode=search_mode)

    save_conversation(
        question=question,
        answer=result["answer"],
        company=company_filter,
        search_mode=search_mode,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )

    st.subheader("Respuesta")
    st.markdown(result["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Empresa", company)
    col2.metric("Tokens entrada", result["input_tokens"])
    col3.metric("Tokens salida", result["output_tokens"])

    with st.expander(f"📄 Ver {len(result['chunks'])} fragmentos usados"):
        for i, chunk in enumerate(result["chunks"], 1):
            st.markdown(f"**Fragmento {i} — {chunk['company']} (p. {chunk['page_start']}-{chunk['page_end']})**")
            st.text(chunk["text"][:600] + "...")
            st.divider()

# Historial
st.divider()
st.subheader("📜 Historial de conversaciones")

history = get_recent_conversations(limit=10)
if not history:
    st.info("Aún no hay conversaciones guardadas.")
else:
    for row in history:
        with st.expander(f"[{row['timestamp'][:19]}] {row['company'] or 'Todas'} — {row['question'][:60]}"):
            st.markdown(f"**Pregunta:** {row['question']}")
            st.markdown(f"**Respuesta:** {row['answer']}")
            st.caption(f"Modo: {row['search_mode']} | Tokens: {row['input_tokens']} entrada / {row['output_tokens']} salida")
