from research_assistant.cli import build_parser

def test_parser_accepts_quoted_question():
    args = build_parser().parse_args(["ask", "Summarize the uploaded papers"])
    assert args.command == "ask"
    assert args.question == "Summarize the uploaded papers"
