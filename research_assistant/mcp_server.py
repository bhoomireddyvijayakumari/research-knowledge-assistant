import json
from mcp.server.fastmcp import FastMCP
from .retrieval import search_documents, get_document_list

mcp = FastMCP("research-knowledge-tools")

@mcp.tool()
def search_knowledge_base(query: str, k: int = 5) -> str:
    """Search indexed PDFs for relevant passages. Returns source filename, page and excerpt."""
    return json.dumps(search_documents(query, k), ensure_ascii=False)

@mcp.tool()
def list_indexed_documents() -> str:
    """List PDFs currently indexed in the research knowledge base."""
    return json.dumps(get_document_list(), ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport="stdio")
