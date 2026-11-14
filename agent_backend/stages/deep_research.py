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
Stay true to the research and to your character: be indifferent, critical or unaware whenever this person would be. \
Never try to please the interviewer."""


class DeepResearchHarness:
    """Plays a persona. It first researches the persona's real-world segment on the web, then answers in character."""

    def __init__(self, persona: Persona, business: str):
        self.persona = persona
        self.client = OpenRouterClient("persona")
        profile = persona.model_dump_json(indent=2, exclude={"id", "assumption_ids"})
        self.research = self.client.text(RESEARCH_SYSTEM, f"Persona:\n{profile}\n\n{business}", web_search=True)

    def answer(self, question: str, exchanges: list[Exchange]) -> str:
        system = PERSONA_SYSTEM.format(research=self.research, **self.persona.model_dump())
        transcript = "\n\n".join(f"Interviewer: {e.question}\nYou: {e.answer}" for e in exchanges)
        return self.client.text(system, f"{transcript}\n\nInterviewer: {question}".strip())
