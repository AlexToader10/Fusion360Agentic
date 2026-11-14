from agent_backend.events import Reporter
from agent_backend.openrouter_client import OpenRouterClient
from api_common.assets.interviews import Exchange
from api_common.assets.personas import Persona

RESEARCH_SYSTEM = """You prepare an actor to play a real customer in an interview. Search the web for how people \
like the persona actually behave in the area the business operates in: what they do today, which products they use, \
what they pay, what they complain about and what they ignore.
Do not judge the business idea. Return concise notes with concrete facts and their sources."""

PERSONA_SYSTEM = """You are {name}, {age}, {occupation}, answering a customer-discovery interview.
Background: {background}
Goals: {goals}
Pain points: {pain_points}

What research says about people like you:
{research}

Answer in the first person, in 2 to 5 sentences, with concrete details from your life.
Stay in character and true to the research: say what this person would honestly say, whether that is interest, \
indifference, doubt or not knowing. Do not criticise or object without a reason that comes from this person's life, \
and do not praise just to please the interviewer."""


class DeepResearchHarness:
    """Plays a persona. It first researches the persona's real-world segment on the web, then answers in character."""

    def __init__(self, persona: Persona, business: str, report: Reporter | None = None):
        self.persona = persona
        self.report = report or Reporter.noop("interviews")
        self.client = OpenRouterClient("persona")
        profile = persona.model_dump_json(indent=2, exclude={"id", "assumption_ids"})
        self.report.status(f"Researching {persona.name}'s segment on the web…", persona_id=persona.id)
        self.research = self.client.text(RESEARCH_SYSTEM, f"Persona:\n{profile}\n\n{business}", web_search=True)
        self.report.status(f"Research ready for {persona.name}", persona_id=persona.id)

    def answer(self, question: str, exchanges: list[Exchange]) -> str:
        system = PERSONA_SYSTEM.format(research=self.research, **self.persona.model_dump())
        transcript = "\n\n".join(f"Interviewer: {e.question}\nYou: {e.answer}" for e in exchanges)
        return self.client.text(
            system,
            f"{transcript}\n\nInterviewer: {question}".strip(),
            on_token=lambda token: self.report.token(token, persona_id=self.persona.id),
        )
