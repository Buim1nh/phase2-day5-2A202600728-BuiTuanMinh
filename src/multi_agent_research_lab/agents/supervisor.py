"""Supervisor / router skeleton."""

import os
import json
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self):
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        max_iter = int(os.getenv("MAX_ITERATIONS", "6"))
        if state.iteration >= max_iter:
            state.errors.append("Max iterations reached")
            state.record_route("done")
            return state

        if state.final_answer:
            state.record_route("done")
            return state

        prompt = f"""You are a Supervisor Agent orchestrating a research task.
User Request: {state.request.query}
Current Notes:
- Research Notes: {state.research_notes}
- Analysis Notes: {state.analysis_notes}
- Final Answer: {state.final_answer}

Decide the next step. Options:
- "researcher": if we need more information or the research notes are empty.
- "analyst": if we have enough research notes but no analysis yet, or if analysis is insufficient.
- "writer": if we have good analysis and are ready to write the final answer.
- "done": if everything is fully complete.

Output exactly a JSON object with a single key "next" containing the chosen option. Example: {{"next": "researcher"}}"""
        
        response = self.llm.complete("You are a helpful supervisor.", prompt)
        
        try:
            text = response.content.strip()
            if text.startswith("```json"):
                text = text[7:-3]
            elif text.startswith("```"):
                text = text[3:-3]
            data = json.loads(text.strip())
            next_route = data.get("next", "done")
        except Exception:
            if not state.research_notes:
                next_route = "researcher"
            elif not state.analysis_notes:
                next_route = "analyst"
            elif not state.final_answer:
                next_route = "writer"
            else:
                next_route = "done"

        if next_route not in ["researcher", "analyst", "writer", "done"]:
            next_route = "done"

        state.record_route(next_route)
        state.add_trace_event("supervisor_decision", {"next": next_route})
        return state
