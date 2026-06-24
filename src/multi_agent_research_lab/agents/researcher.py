"""Researcher agent skeleton."""

import json
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self):
        self.llm = LLMClient()
        self.search_client = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        prompt = f"""You are a Researcher Agent.
User Request: {state.request.query}

Generate a search query to find relevant information.
Output exactly a JSON object with a key "query" containing the search term. Example: {{"query": "search term"}}"""
        
        response = self.llm.complete("You are a helpful researcher.", prompt)
        search_query = state.request.query
        try:
            text = response.content.strip()
            if text.startswith("```json"):
                text = text[7:-3]
            elif text.startswith("```"):
                text = text[3:-3]
            data = json.loads(text.strip())
            search_query = data.get("query", state.request.query)
        except Exception:
            pass

        docs = self.search_client.search(search_query)
        state.sources.extend(docs)

        doc_texts = "\n".join([f"Title: {d.title}\nContent: {d.snippet}" for d in docs])
        summary_prompt = f"""Summarize these findings relevant to the query: '{state.request.query}'.
Findings:
{doc_texts}"""
        
        summary_response = self.llm.complete("You are a researcher summarizing findings.", summary_prompt)
        if state.research_notes:
            state.research_notes += "\n\n" + summary_response.content
        else:
            state.research_notes = summary_response.content

        state.add_trace_event("researcher_ran", {"search_query": search_query, "docs_found": len(docs)})
        return state
