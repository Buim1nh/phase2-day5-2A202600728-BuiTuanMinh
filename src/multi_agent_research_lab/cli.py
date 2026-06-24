"""Command-line entrypoint for the lab starter."""

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report


app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""
    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    
    llm = LLMClient()
    response = llm.complete("You are a helpful assistant.", query)
    
    state.final_answer = response.content
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
        console.print(result.model_dump_json(indent=2))
    except Exception as exc:
        console.print(Panel.fit(str(exc), title="Error", style="red"))
        raise typer.Exit(code=2) from exc


@app.command("benchmark")
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run benchmark comparing baseline vs multi-agent."""
    _init()
    
    def baseline_runner(q: str) -> ResearchState:
        request = ResearchQuery(query=q)
        state = ResearchState(request=request)
        llm = LLMClient()
        response = llm.complete("You are a helpful assistant. Provide a comprehensive answer.", q)
        state.final_answer = response.content
        return state

    def ma_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow()
        return workflow.run(state)

    console.print("[blue]Running baseline...[/blue]")
    base_state, base_metrics = run_benchmark("Baseline", query, baseline_runner)
    
    console.print("[blue]Running multi-agent...[/blue]")
    ma_state, ma_metrics = run_benchmark("Multi-Agent", query, ma_runner)

    report_content = render_markdown_report([base_metrics, ma_metrics])
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    console.print("[green]Benchmark complete. Report saved to reports/benchmark_report.md[/green]")


if __name__ == "__main__":
    app()
