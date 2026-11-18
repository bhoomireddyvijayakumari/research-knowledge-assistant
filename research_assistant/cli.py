import argparse
import json
from pathlib import Path
from .config import DOCUMENTS_DIR, ensure_directories
from .db import initialize_db, list_documents, get_history
from .ingest import ingest_pdf, ingest_directory
from .agent import ask

def build_parser():
    parser = argparse.ArgumentParser(prog="research-assistant",
        description="Research across your indexed PDF knowledge base.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Initialize local database and folders")
    add = sub.add_parser("ingest", help="Index a PDF or all PDFs in documents folder")
    add.add_argument("path", nargs="?", help="PDF path; omit to ingest data/documents/")
    sub.add_parser("list", help="List indexed PDFs")
    query = sub.add_parser("ask", help="Ask a question across indexed PDFs")
    query.add_argument("question", help="Question in quotation marks")
    hist = sub.add_parser("history", help="Show recent research questions")
    hist.add_argument("--limit", type=int, default=10)
    return parser

def main():
    args = build_parser().parse_args()
    ensure_directories()
    if args.command == "init":
        initialize_db()
        print(json.dumps({"initialized": True, "documents_dir": str(DOCUMENTS_DIR)}, indent=2))
    elif args.command == "ingest":
        result = ingest_pdf(args.path) if args.path else ingest_directory()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "list":
        print(json.dumps(list_documents(), indent=2, ensure_ascii=False))
    elif args.command == "ask":
        print(ask(args.question))
    elif args.command == "history":
        print(json.dumps(get_history(args.limit), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
