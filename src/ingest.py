import json
import pdfplumber
from pathlib import Path

CHUNKS_FILE = Path("data/chunks.json")

REPORTS_DIR = Path("data/reports")

COMPANIES = {
    "bancolombia_2024.pdf": "Bancolombia",
    "ecopetrol_2024.pdf":   "Ecopetrol",
    "grupo_sura_2024.pdf":  "Grupo Sura",
    "cemargos_2024.pdf":    "Cemargos",
    "celsia_2023.pdf":      "Celsia",
}


def extract_text_from_pdf(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page": i + 1,
                    "text": text.strip(),
                })
    return pages


def chunk_pages(pages, company, chunk_size=3, overlap=1):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(pages), step):
        group = pages[i: i + chunk_size]
        combined_text = "\n\n".join(p["text"] for p in group)
        chunks.append({
            "company":    company,
            "page_start": group[0]["page"],
            "page_end":   group[-1]["page"],
            "text":       combined_text,
        })
    return chunks


def load_documents():
    all_chunks = []
    for filename, company in COMPANIES.items():
        pdf_path = REPORTS_DIR / filename
        if not pdf_path.exists():
            print(f"No encontrado: {filename}")
            continue
        print(f"Procesando {company}...")
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages, company)
        print(f"  {len(pages)} páginas → {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks):
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks guardados en {CHUNKS_FILE}")


def load_chunks():
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Chunks cargados desde caché: {len(chunks)} chunks")
        return chunks
    print("Caché no encontrada, procesando PDFs...")
    chunks = load_documents()
    save_chunks(chunks)
    return chunks


if __name__ == "__main__":
    chunks = load_documents()
    save_chunks(chunks)
    print(f"\nEjemplo chunk 1:")
    print(f"  Empresa:  {chunks[0]['company']}")
    print(f"  Páginas:  {chunks[0]['page_start']}-{chunks[0]['page_end']}")
    print(f"  Texto:    {chunks[0]['text'][:300]}")
