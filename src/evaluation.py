import json
import os
import random
from pathlib import Path

from groq import Groq
from tqdm import tqdm


GROUND_TRUTH_FILE = Path("data/ground_truth.json")


# ─── Generación de preguntas de evaluación ───────────────────────────────────

def generate_questions(chunk, client, n=5):
    prompt = f"""Eres un evaluador de sistemas de recuperación de información financiera.

Dado el siguiente fragmento de un reporte anual corporativo, genera {n} preguntas concretas
cuya respuesta se encuentre directamente en el texto.

Fragmento:
[{chunk['company']} | Páginas {chunk['page_start']}-{chunk['page_end']}]
{chunk['text'][:1200]}

Devuelve ÚNICAMENTE una lista JSON de strings con las preguntas, sin explicaciones.
Ejemplo: ["¿Pregunta 1?", "¿Pregunta 2?"]"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    # Extraer el JSON de la respuesta
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    return json.loads(text[start:end])


def build_ground_truth(chunks, n_chunks=60, questions_per_chunk=3):
    """Genera preguntas de evaluación a partir de chunks aleatorios."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    sample = random.sample(chunks, min(n_chunks, len(chunks)))
    ground_truth = []

    for chunk in tqdm(sample, desc="Generando preguntas"):
        try:
            questions = generate_questions(chunk, client, n=questions_per_chunk)
            for q in questions:
                ground_truth.append({
                    "question": q,
                    "company":    chunk["company"],
                    "page_start": chunk["page_start"],
                    "page_end":   chunk["page_end"],
                })
        except Exception as e:
            print(f"  Error en {chunk['company']} p{chunk['page_start']}: {e}")

    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"\nGround truth guardado: {len(ground_truth)} preguntas en {GROUND_TRUTH_FILE}")
    return ground_truth


# ─── Métricas ────────────────────────────────────────────────────────────────

def hit_rate(results, relevant_page_start):
    """¿Apareció el chunk correcto en los resultados?"""
    return any(r["page_start"] == relevant_page_start for r in results)


def mrr(results, relevant_page_start):
    """Reciprocal Rank: 1/posición del chunk correcto (0 si no aparece)."""
    for i, r in enumerate(results, 1):
        if r["page_start"] == relevant_page_start:
            return 1.0 / i
    return 0.0


def evaluate(search_engine, ground_truth, search_mode="hybrid", top_k=5):
    hit_rates = []
    mrr_scores = []

    for item in tqdm(ground_truth, desc=f"Evaluando {search_mode}"):
        query   = item["question"]
        company = item["company"]
        page    = item["page_start"]

        if search_mode == "keyword":
            results = search_engine.keyword_search(query, company=company, top_k=top_k)
        elif search_mode == "vector":
            results = search_engine.vector_search(query, company=company, top_k=top_k)
        else:
            results = search_engine.hybrid_search(query, company=company, top_k=top_k)

        hit_rates.append(hit_rate(results, page))
        mrr_scores.append(mrr(results, page))

    avg_hr  = sum(hit_rates)  / len(hit_rates)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    return {"hit_rate": round(avg_hr, 4), "mrr": round(avg_mrr, 4)}


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from dotenv import load_dotenv
    from search import SearchEngine

    load_dotenv()

    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    # Generar ground truth si no existe
    if not GROUND_TRUTH_FILE.exists():
        print("Generando ground truth con Groq (puede tardar ~3 min)...")
        ground_truth = build_ground_truth(chunks, n_chunks=60, questions_per_chunk=3)
    else:
        with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
            ground_truth = json.load(f)
        print(f"Ground truth cargado: {len(ground_truth)} preguntas")

    print("\nConstruyendo motor de búsqueda...")
    engine = SearchEngine(chunks)

    print("\n=== Resultados de Evaluación ===")
    for mode in ["keyword", "vector", "hybrid"]:
        metrics = evaluate(engine, ground_truth, search_mode=mode)
        print(f"  {mode:8s} → Hit Rate: {metrics['hit_rate']:.4f} | MRR: {metrics['mrr']:.4f}")
