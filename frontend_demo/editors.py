"""How the output of each finished stage is edited: what views.py shows, as form fields.

An editor draws the draft of its stage, a copy of the output as plain data, and returns the output to save. Every
field writes what it holds back into the draft, so nothing typed is lost when a card is added or removed.
"""

from collections.abc import Callable
from itertools import count

import streamlit as st

from api_common.assets.common import PipelineState
from frontend_demo.views import VERDICTS, chips, persona_names, statements, text

VERDICT_CHOICE = {  # the marks of the read view: check, cross and empty ring
    "validated": ":green[:material/check_circle:] Validated",
    "invalidated": ":red[:material/cancel:] Invalidated",
    "inconclusive": ":gray[:material/radio_button_unchecked:] Inconclusive",
}


def area(label: str, item: dict, name: str, key: str) -> None:
    """A text field that is as tall as its text."""
    item[name] = st.text_area(label, item.get(name, ""), key=f"{key}_{name}", height="content")


def about(id: str, title: str) -> str:
    """A field label that opens with the id the field is about, drawn like the chips of the read view."""
    return f":green-badge[{id}] **{text(title)}**"


def assumption_picker(label: str, item: dict, state: PipelineState) -> None:
    """The assumptions an item is about, picked by their statements instead of typed as ids."""
    titles = statements(state)
    item["assumption_ids"] = st.multiselect(
        label,
        list(titles),
        [id for id in item.get("assumption_ids", []) if id in titles],
        format_func=lambda id: f"{id} · {titles[id]}",
        key=f"{item['id']}_assumption_ids",
    )


def verdict(item: dict, note: str, titles: dict[str, str], key: str) -> None:
    """The verdict on one assumption as three choices, and under them the text that backs it."""
    id = item["assumption_id"]
    with st.container(border=True):
        item["verdict"] = st.radio(
            about(id, titles.get(id, "")),
            VERDICTS,
            VERDICTS.index(item["verdict"]),
            format_func=VERDICT_CHOICE.get,
            key=f"{key}_verdict",
            horizontal=True,
        )
        area(note.capitalize(), item, note, key)


def add(stage: str, items: list[dict]) -> None:
    """Append an empty item under the next free id: A6 for assumptions, P5 for personas, Q9 for questions."""
    taken = {item["id"] for item in items}
    ids = (f"{stage[0].upper()}{number}" for number in count(len(items) + 1))
    items.append({"id": next(id for id in ids if id not in taken)})


def written(item: dict) -> bool:
    """Whether a card holds any text besides its id."""
    return any(isinstance(value, str) and value.strip() for name, value in item.items() if name != "id")


def cards(stage: str, draft: dict, fields: Callable[[dict], None]) -> dict:
    """A stage that is one list: a card per item with a button that removes it, and a button that adds an empty card.

    The buttons submit the form like Save changes does, so what was typed so far is kept. A card left without any
    text is not saved.
    """
    items = draft[stage]
    for item in items:
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.html(chips([item["id"]]), width="content")
                st.form_submit_button(
                    "Remove",
                    key=f"remove_{item['id']}",
                    type="tertiary",
                    icon=":material/delete:",
                    on_click=items.remove,
                    args=(item,),
                )
            fields(item)
    label = f"Add {stage[:-1]}"  # the stage name without its plural s
    st.form_submit_button(label, key=f"add_{stage}", icon=":material/add:", on_click=add, args=(stage, items))
    return {stage: [item for item in items if written(item)]}


def assumptions(draft: dict, state: PipelineState) -> dict:
    def fields(item: dict) -> None:
        area("Assumption", item, "statement", item["id"])
        rationale, category = st.columns([3, 1])
        with rationale:
            area("Why the business depends on it", item, "rationale", item["id"])
        item["category"] = category.text_input("Category", item.get("category", ""), key=f"{item['id']}_category")

    return cards("assumptions", draft, fields)


def personas(draft: dict, state: PipelineState) -> dict:
    def fields(item: dict) -> None:
        key = item["id"]
        name, age, occupation = st.columns([3, 1, 3])
        item["name"] = name.text_input("Name", item.get("name", ""), key=f"{key}_name")
        item["age"] = age.number_input("Age", 0, 120, item.get("age", 35), key=f"{key}_age")
        item["occupation"] = occupation.text_input("Occupation", item.get("occupation", ""), key=f"{key}_occupation")
        area("Background", item, "background", key)
        area("Goals", item, "goals", key)
        area("Pain points", item, "pain_points", key)
        assumption_picker("Assumptions this persona can speak to", item, state)

    return cards("personas", draft, fields)


def questions(draft: dict, state: PipelineState) -> dict:
    def fields(item: dict) -> None:
        area("Question", item, "text", item["id"])
        assumption_picker("Assumptions it tests", item, state)

    return cards("questions", draft, fields)


def interviews(draft: dict, state: PipelineState) -> dict:
    names = persona_names(state)
    for index, interview in enumerate(draft["interviews"]):
        with st.expander(names.get(interview["persona_id"], interview["persona_id"])):
            for number, exchange in enumerate(interview["exchanges"]):
                label = about(exchange["question_id"], exchange["question"])
                area(label, exchange, "answer", f"interview_{index}_{number}")
    return draft


def validations(draft: dict, state: PipelineState) -> dict:
    names = persona_names(state)
    titles = statements(state)
    for index, result in enumerate(draft["validations"]):
        with st.expander(names.get(result["persona_id"], result["persona_id"])):
            for number, item in enumerate(result["verdicts"]):
                verdict(item, "evidence", titles, f"validation_{index}_{number}")
    return draft


def conclusions(draft: dict, state: PipelineState) -> dict:
    titles = statements(state)
    area("Summary", draft, "summary", "conclusions")
    for number, item in enumerate(draft["assumptions"]):
        verdict(item, "reasoning", titles, f"conclusion_{number}")
    lines = st.text_area(
        "Recommendations, one per line",
        "\n".join(draft["recommendations"]),
        key="conclusions_recommendations",
        height="content",
    )
    draft["recommendations"] = [line.strip() for line in lines.splitlines() if line.strip()]
    return draft


EDITORS = {
    "assumptions": assumptions,
    "personas": personas,
    "questions": questions,
    "interviews": interviews,
    "validations": validations,
    "conclusions": conclusions,
}
