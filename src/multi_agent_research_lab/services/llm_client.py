"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

from multi_agent_research_lab.core.errors import AgentExecutionError

load_dotenv()


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client implementation using DeepSeek."""

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion using DeepSeek API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0
            )
        except Exception as e:
            raise AgentExecutionError(f"LLM API error: {str(e)}")
