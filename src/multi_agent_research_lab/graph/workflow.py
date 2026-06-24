"""LangGraph workflow skeleton."""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.writer import WriterAgent


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.graph = self.build()

    def build(self) -> CompiledStateGraph:
        """Create a LangGraph graph."""
        workflow = StateGraph(ResearchState)

        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("researcher", self.researcher.run)
        workflow.add_node("analyst", self.analyst.run)
        workflow.add_node("writer", self.writer.run)

        workflow.set_entry_point("supervisor")

        def supervisor_router(state: ResearchState) -> str:
            route = state.route_history[-1] if state.route_history else "done"
            if route == "done":
                return END
            return route

        workflow.add_conditional_edges(
            "supervisor",
            supervisor_router,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END
            }
        )

        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")

        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        result = self.graph.invoke(state)
        if isinstance(result, dict):
            return ResearchState(**result)
        return result  # type: ignore
