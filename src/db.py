import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path("data/conversations.db")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            company       TEXT,
            question      TEXT    NOT NULL,
            answer        TEXT    NOT NULL,
            search_mode   TEXT    NOT NULL,
            input_tokens  INTEGER,
            output_tokens INTEGER
        )
    """)
    con.commit()
    con.close()


def save_conversation(question, answer, company=None, search_mode="hybrid",
                      input_tokens=None, output_tokens=None):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO conversations
            (timestamp, company, question, answer, search_mode, input_tokens, output_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        company,
        question,
        answer,
        search_mode,
        input_tokens,
        output_tokens,
    ))
    con.commit()
    con.close()


def get_recent_conversations(limit=20):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT * FROM conversations
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    save_conversation(
        question="¿Cuál fue la utilidad neta de Bancolombia en 2024?",
        answer="Prueba de guardado en base de datos.",
        company="Bancolombia",
        search_mode="hybrid",
        input_tokens=2307,
        output_tokens=72,
    )
    print("Conversaciones guardadas:")
    for row in get_recent_conversations():
        print(f"  [{row['timestamp']}] {row['company']} — {row['question'][:50]}")
