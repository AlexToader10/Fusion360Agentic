from agent_backend.events import Reporter
from api_common.assets.common import PipelineRequest, PipelineState


class Stage:
    name: str  # the PipelineState field this stage fills in

    def run(self, state: PipelineState, report: Reporter | None = None) -> PipelineState:
        raise NotImplementedError


def count_rule(count: int | None, default: str, singular: str, plural: str) -> tuple[str, str]:
    """What to show in the prompt, plus a sentence that asks for that exact sample size.

    A model that is asked for "1 persona" politely and nothing else may return a whole cast. The explicit rule
    makes the requested size part of the contract, not a suggestion. Without a count, the user can still ask for
    a size in plain words in the additional info.
    """
    if count is None:
        return default, f"Return {default} {plural}, unless the additional info asks for a different number of {plural}."
    noun = singular if count == 1 else plural
    return str(count), f"Return exactly {count} {noun}, no more and no fewer."


def trim_count[T](items: list[T], count: int | None, report: Reporter, kind: str) -> list[T]:
    """Keep a request for an exact count even when the model returns a larger cast."""
    if count is None or len(items) <= count:
        return items
    report.status(f"Model returned {len(items)} {kind}; keeping the requested {count}.", returned=len(items), kept=count)
    return items[:count]


def reindex(items: list, prefix: str) -> None:
    """Give the kept items clean ids: A1..An, P1..Pn, Q1..Qn."""
    for number, item in enumerate(items, start=1):
        item.id = f"{prefix}{number}"


def business_context(request: PipelineRequest) -> str:
    parts = {
        "Business idea": request.business_idea,
        "Target audience": request.target_audience,
        "Additional info": request.additional_info,
        "Business plan": request.business_plan,
    }
    return "\n\n".join(f"{title}:\n{text}" for title, text in parts.items() if text)
