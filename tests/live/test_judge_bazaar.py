"""Live adversarial benchmark of the questions bias judge: "The Bias Bazaar".

A handcrafted set of 20 questions about weeknight dinners: 10 leading questions from different bias families and 10
neutral, past-behaviour questions. The real judge sees all 20 in one call, and the 10 neutral ones in a second call.

Checks: recall on the biased questions, false positives on the neutral ones, every suggestion names a question id and
gives an actionable rewrite cue, restraint on the neutral-only set, and a confusion matrix for the run.

Budget: two judge calls, each made once by a module-scoped fixture.
"""

from __future__ import annotations

import re

import pytest

from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.questions import QuestionsStage
from api_common.assets.questions import Question, QuestionsOutput
from tests.helpers import extract_ids

pytestmark = [pytest.mark.live]

BIASED_TEXTS = [
    "Don't you hate how long weeknight dinner planning takes?",
    "Would you pay 15 EUR a week for a box that delivers pre-portioned dinner ingredients?",
    "Our app plans your whole week of dinners in two minutes. What do you think of it?",
    "Most parents say they give up on homemade dinners by Wednesday. Do you feel the same?",
    "How much time and money do you lose each week on dinner planning and grocery runs?",
    "Why is your wasteful takeaway habit costing your family so much?",
    "You'd agree that cooking from scratch on weeknights is exhausting, right?",
    "Will you use a meal kit every week once it arrives at your door?",
    "At only 49 EUR a month, can you see yourself saving the cost of takeaway?",
    "As an expert home cook, you must see how much a pre-portioned kit would help, don't you?",
]

NEUTRAL_TEXTS = [
    "Walk me through how you decided what to cook last Tuesday.",
    "Tell me about the last time dinner did not go as planned at home.",
    "What did you buy for dinner during the past week, and where did you buy it?",
    "Who usually does the grocery shopping in your household?",
    "How did you spend your time on weeknight evenings last month?",
    "Describe the steps you took the last time you cooked a meal from scratch.",
    "What happens when the fridge is nearly empty on a school night?",
    "How did the family handle dinner during your last busy week?",
    "Tell me about the grocery shops you visited in the last month.",
    "What did you cook the last time you had guests for dinner?",
]

BIASED_IDS = [f"Q{n}" for n in range(1, len(BIASED_TEXTS) + 1)]
NEUTRAL_IDS = [f"Q{n}" for n in range(len(BIASED_TEXTS) + 1, len(BIASED_TEXTS) + len(NEUTRAL_TEXTS) + 1)]

MIN_BIASED_RECALL = 8
MAX_NEUTRAL_FALSE_POSITIVES = 2
MAX_NEUTRAL_ONLY_SUGGESTIONS = 2
MIN_PRECISION = 0.8
MIN_RECALL = 0.8

REWRITE_CUE = re.compile(r"\b(ask|replace|remove|neutral|rephrase|instead|example|past)\w*\b", re.IGNORECASE)


def _build_questions(texts: list[str], ids: list[str]) -> QuestionsOutput:
    return QuestionsOutput(
        questions=[Question(id=ident, text=text, assumption_ids=["A1"]) for ident, text in zip(ids, texts, strict=True)]
    )


def _question_ids(suggestion: str) -> list[str]:
    return extract_ids(suggestion)["Q"]


def _flagged(suggestions: list[str]) -> set[str]:
    return {ident for suggestion in suggestions for ident in _question_ids(suggestion)}


def _format(suggestions: list[str]) -> str:
    return "\n".join(f"  - {suggestion}" for suggestion in suggestions) or "  (none)"


@pytest.fixture(scope="module")
def bazaar_questions() -> QuestionsOutput:
    return _build_questions(BIASED_TEXTS + NEUTRAL_TEXTS, BIASED_IDS + NEUTRAL_IDS)


@pytest.fixture(scope="module")
def bazaar_suggestions(bazaar_questions: QuestionsOutput) -> list[str]:
    return QuestionsStage._judge(OpenRouterClient("questions_judge"), bazaar_questions)


@pytest.fixture(scope="module")
def neutral_only_suggestions() -> list[str]:
    neutral = _build_questions(NEUTRAL_TEXTS, NEUTRAL_IDS)
    return QuestionsStage._judge(OpenRouterClient("questions_judge"), neutral)


def test_judge_recalls_biased_questions(bazaar_suggestions: list[str]) -> None:
    flagged = _flagged(bazaar_suggestions)
    caught = [ident for ident in BIASED_IDS if ident in flagged]
    missed = [ident for ident in BIASED_IDS if ident not in flagged]

    assert len(caught) >= MIN_BIASED_RECALL, (
        f"judge caught {len(caught)}/{len(BIASED_IDS)} biased questions, needs {MIN_BIASED_RECALL}. "
        f"Missed: {missed}. Suggestions:\n{_format(bazaar_suggestions)}"
    )


def test_judge_spares_neutral_questions(bazaar_suggestions: list[str]) -> None:
    flagged = _flagged(bazaar_suggestions)
    false_positives = [ident for ident in NEUTRAL_IDS if ident in flagged]

    assert len(false_positives) <= MAX_NEUTRAL_FALSE_POSITIVES, (
        f"judge flagged {len(false_positives)}/{len(NEUTRAL_IDS)} neutral questions as leading: {false_positives}. "
        f"Suggestions:\n{_format(bazaar_suggestions)}"
    )


def test_every_suggestion_names_a_question_id(bazaar_suggestions: list[str]) -> None:
    unnamed = [suggestion for suggestion in bazaar_suggestions if not _question_ids(suggestion)]

    assert not unnamed, f"judge suggestions name no question id: {unnamed}"


def test_every_suggestion_gives_an_actionable_rewrite_cue(bazaar_suggestions: list[str]) -> None:
    without_cue = [suggestion for suggestion in bazaar_suggestions if not REWRITE_CUE.search(suggestion)]

    assert not without_cue, (
        "judge suggestions say what is wrong but not how to fix it (expected words such as ask, replace, remove, "
        f"neutral, rephrase, instead, example or past): {without_cue}"
    )


def test_neutral_only_set_gets_little_or_no_feedback(neutral_only_suggestions: list[str]) -> None:
    assert len(neutral_only_suggestions) <= MAX_NEUTRAL_ONLY_SUGGESTIONS, (
        f"judge returned {len(neutral_only_suggestions)} suggestions for 10 neutral questions "
        f"(max {MAX_NEUTRAL_ONLY_SUGGESTIONS}). Suggestions:\n{_format(neutral_only_suggestions)}"
    )


def test_bias_bazaar_confusion_matrix(bazaar_suggestions: list[str]) -> None:
    flagged = _flagged(bazaar_suggestions)
    tp = sum(ident in flagged for ident in BIASED_IDS)
    fn = len(BIASED_IDS) - tp
    fp = sum(ident in flagged for ident in NEUTRAL_IDS)
    tn = len(NEUTRAL_IDS) - fp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    print("\nThe Bias Bazaar: questions judge confusion matrix (positive = leading)")
    print("                  judge flagged   judge passed")
    print(f"  leading (10)    TP={tp:<12d}  FN={fn}")
    print(f"  neutral (10)    FP={fp:<12d}  TN={tn}")
    print(f"  precision={precision:.2f}  recall={recall:.2f}")
    print("Suggestions:\n" + _format(bazaar_suggestions))

    assert precision >= MIN_PRECISION, f"precision {precision:.2f} below {MIN_PRECISION}: FP={fp}, TP={tp}"
    assert recall >= MIN_RECALL, f"recall {recall:.2f} below {MIN_RECALL}: FN={fn}, TP={tp}"
