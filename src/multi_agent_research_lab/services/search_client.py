"""Search client abstraction for ResearcherAgent."""

from duckduckgo_search import DDGS
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.core.errors import AgentExecutionError


class SearchClient:
    """DuckDuckGo search client implementation."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query using DuckDuckGo."""
        try:
            results = DDGS().text(query, max_results=max_results)
            docs = []
            if results:
                for r in results:
                    docs.append(SourceDocument(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", "")
                    ))
            return docs
        except Exception as e:
            raise AgentExecutionError(f"Search API error: {str(e)}")
