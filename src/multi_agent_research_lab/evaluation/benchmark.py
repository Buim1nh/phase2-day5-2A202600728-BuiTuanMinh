"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return a placeholder metric object."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    cost = 0.0
    quality = 0.0 # Could use LLM-as-a-judge here
    
    metrics = BenchmarkMetrics(
        run_name=run_name, 
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=f"Iterations: {state.iteration}"
    )
    return state, metrics
