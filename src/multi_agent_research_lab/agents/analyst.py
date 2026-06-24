"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self):
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        prompt = f"""You are an Analyst Agent.
Original Request: {state.request.query}
Research Notes:
{state.research_notes}

Extract key claims, compare viewpoints, and synthesize structured insights.
Focus on providing a deep analysis of the information.
"""
        response = self.llm.complete("You are a rigorous analyst.", prompt)
        state.analysis_notes = response.content
        state.add_trace_event("analyst_ran", {})
        return state
