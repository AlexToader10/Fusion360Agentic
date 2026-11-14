import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
WEB_SEARCH_TOOL = {"type": "openrouter:web_search"}


def model_for(task: str) -> str:
    """OPENROUTER_MODEL_<TASK> when set, otherwise the general OPENROUTER_MODEL."""
    return os.getenv(f"OPENROUTER_MODEL_{task.upper()}") or os.environ["OPENROUTER_MODEL"]


class OpenRouterClient:
    """Wrapper over OpenRouter's OpenAI-compatible endpoint, bound to the model of one task."""

    def __init__(self, task: str):
        self.model = model_for(task)
        self.client = OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])

    def text(self, system: str, prompt: str, web_search: bool = False) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, prompt),
            **({"tools": [WEB_SEARCH_TOOL]} if web_search else {}),
        )
        return completion.choices[0].message.content

    def parse[T: BaseModel](self, system: str, prompt: str, output_model: type[T]) -> T:
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=self._messages(system, prompt),
            response_format=output_model,
        )
        return completion.choices[0].message.parsed

    @staticmethod
    def _messages(system: str, prompt: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
