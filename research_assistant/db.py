import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .config import DATABASE_PATH, ensure_directories

def connect():
    ensure_directories()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    with connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            file_path TEXT NOT NULL,
            pages INTEGER NOT NULL DEFAULT 0,
            chunks INTEGER NOT NULL DEFAULT 0,
            indexed_at TEXT NOT NULL
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS research_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id)
            )""")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_messages
            ON conversation_messages(conversation_id, id)
            """)

def upsert_document(filename: str, file_path: str, pages: int, chunks: int):
    initialize_db()
    with connect() as conn:
        conn.execute("""
        INSERT INTO documents(filename,file_path,pages,chunks,indexed_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(filename) DO UPDATE SET
          file_path=excluded.file_path, pages=excluded.pages,
          chunks=excluded.chunks, indexed_at=excluded.indexed_at
        """, (filename, file_path, pages, chunks,
              datetime.now(timezone.utc).isoformat()))

def list_documents():
    initialize_db()
    with connect() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM documents ORDER BY filename"
        ).fetchall()]

def save_research(question: str, answer: str):
    initialize_db()
    with connect() as conn:
        conn.execute(
            "INSERT INTO research_history(question,answer,created_at) VALUES(?,?,?)",
            (question, answer, datetime.now(timezone.utc).isoformat())
        )

def get_history(limit: int = 20):
    initialize_db()
    with connect() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM research_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]


def create_conversation(title: str = "New Research Conversation") -> str:
    """Create a new conversation and return its unique ID."""
    conversation_id = str(uuid.uuid4())

    initialize_db()

    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            INSERT INTO conversations
            (conversation_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                title,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat()
            )
        )

    return conversation_id


def save_conversation_message(
    conversation_id: str,
    role: str,
    content: str
):
    """Save a user or assistant message."""
    if role not in ("user", "assistant"):
        raise ValueError("Role must be user or assistant.")

    initialize_db()

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_messages
            (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                datetime.now(timezone.utc).isoformat()
            )
        )

        conn.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE conversation_id = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                conversation_id
            )
        )


def get_conversation_messages(
    conversation_id: str,
    limit: int = 20
) -> list[dict]:
    """Retrieve the most recent messages in chronological order."""
    initialize_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit)
        ).fetchall()

    return [dict(row) for row in reversed(rows)]


def list_conversations(limit: int = 20) -> list[dict]:
    """List recent conversation sessions."""
    initialize_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    return [dict(row) for row in rows]
