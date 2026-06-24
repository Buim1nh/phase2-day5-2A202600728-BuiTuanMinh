"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self):
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        prompt = f"""You are a Writer Agent.
Original Request: {state.request.query}
Analysis Notes:
{state.analysis_notes}
Research Notes:
{state.research_notes}

Synthesize a clear, comprehensive final response with citations to the provided information.
Provide a high-quality written output.
"""
        response = self.llm.complete("You are an expert technical writer.", prompt)
        state.final_answer = response.content
        state.add_trace_event("writer_ran", {})
        return state
