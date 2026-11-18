import sqlite3
from research_assistant import db

def test_database_initializes_and_saves_history(tmp_path, monkeypatch):
    test_db = tmp_path / "test.sqlite"
    monkeypatch.setattr(db, "DATABASE_PATH", test_db)
    db.initialize_db()
    db.save_research("What is RAG?", "RAG retrieves relevant context.")
    history = db.get_history()
    assert len(history) == 1
    assert history[0]["question"] == "What is RAG?"
