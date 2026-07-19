import os
from groq import Groq


SYSTEM_PROMPT = """Eres un analista financiero experto en empresas colombianas listadas en la Bolsa de Valores de Colombia (BVC).
Tienes acceso a los reportes anuales de: Bancolombia, Ecopetrol, Grupo Sura, Cemargos y Celsia.
Responde en español, de forma clara y precisa, basándote únicamente en la información de los fragmentos proporcionados.
Si la información no está en los fragmentos, dilo explícitamente."""

PROMPT_TEMPLATE = """Fragmentos del reporte anual de {company}:

{context}

---
Pregunta: {question}

Responde basándote únicamente en los fragmentos anteriores."""


MAX_CHUNK_CHARS = 1500


def build_prompt(question, chunks, company=None):
    company_label = company if company else "las empresas"
    context = "\n\n---\n\n".join(
        f"[{c['company']} | Páginas {c['page_start']}-{c['page_end']}]\n{c['text'][:MAX_CHUNK_CHARS]}"
        for c in chunks
    )
    return PROMPT_TEMPLATE.format(
        company=company_label,
        context=context,
        question=question,
    )


class RAGPipeline:
    def __init__(self, search_engine, model="llama-3.3-70b-versatile"):
        self.search_engine = search_engine
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def ask(self, question, company=None, search_mode="hybrid", top_k=5):
        # 1. Búsqueda
        if search_mode == "keyword":
            chunks = self.search_engine.keyword_search(question, company=company, top_k=top_k)
        elif search_mode == "vector":
            chunks = self.search_engine.vector_search(question, company=company, top_k=top_k)
        else:
            chunks = self.search_engine.hybrid_search(question, company=company, top_k=top_k)

        # 2. Construir prompt
        prompt = build_prompt(question, chunks, company=company)

        # 3. LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content
        usage = response.usage

        return {
            "answer":        answer,
            "chunks":        chunks,
            "input_tokens":  usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    from search import SearchEngine

    load_dotenv()

    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Cargando motor de búsqueda...")
    engine = SearchEngine(chunks)

    rag = RAGPipeline(engine)

    question = "¿Cuál fue la utilidad neta de Bancolombia en 2024?"
    print(f"\nPregunta: {question}\n")

    result = rag.ask(question, company="Bancolombia")
    print("Respuesta:")
    print(result["answer"])
    print(f"\nTokens usados: {result['input_tokens']} entrada / {result['output_tokens']} salida")
