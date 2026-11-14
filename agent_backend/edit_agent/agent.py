import json
import os
from pathlib import Path
from uuid import UUID

# The embedded agent keeps its state here instead of the user's own ~/.hermes
os.environ.setdefault("HERMES_HOME", str(Path(__file__).parent / ".hermes_home"))

from run_agent import AIAgent
from tools.registry import registry

from agent_backend.openrouter_client import BASE_URL, model_for
from agent_backend.runs import STAGE_NAMES, RunManager

TOOLSET = "pipeline_edit"

SYSTEM = f"""You edit a business-validation pipeline run that the user has put on hold.
The run has these stages, in order: {", ".join(STAGE_NAMES)}. Each finished stage has an output.
Read the run with get_run, then apply what the user asks with edit_stage.
edit_stage replaces the whole output of a stage, so send the complete output with only the requested change applied \
and every other field and id kept as it is.
Editing a stage clears the stages after it. They run again when the user resumes the run. Tell the user which \
stages will run again."""

RUN_ID = {"type": "string", "description": "UUID of the run"}


class EditAgent:
    """Hermes agent whose tools are the same RunManager functions the HTTP paths call."""

    def __init__(self, manager: RunManager):
        self.manager = manager
        self.histories: dict[UUID, list[dict]] = {}
        self._register("get_run", "Read the run: its status and the output of every finished stage.", {}, self._get_run)
        self._register(
            "edit_stage",
            "Replace the output of a finished stage of a held run.",
            {
                "stage": {"type": "string", "enum": STAGE_NAMES},
                "output": {"type": "object", "description": "The complete new output of the stage"},
            },
            self._edit_stage,
        )

    def chat(self, run_id: UUID, message: str) -> str:
        agent = AIAgent(
            model=model_for("edit_agent"),
            base_url=BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            provider="openrouter",
            enabled_toolsets=[TOOLSET],
            ephemeral_system_prompt=f"{SYSTEM}\n\nThe run id is {run_id}.",
            max_iterations=15,
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
        )
        result = agent.run_conversation(message, conversation_history=self.histories.get(run_id))
        self.histories[run_id] = result["messages"]
        return result["final_response"]

    def _get_run(self, args: dict) -> str:
        return self.manager.get(UUID(args["run_id"])).model_dump_json()

    def _edit_stage(self, args: dict) -> str:
        self.manager.edit(UUID(args["run_id"]), args["stage"], args["output"])
        return json.dumps({"edited": args["stage"]})

    @staticmethod
    def _register(name: str, description: str, properties: dict, function) -> None:
        def handler(args: dict, **_) -> str:
            try:
                return function(args)
            except Exception as error:  # Hermes tools report errors to the model instead of raising
                return json.dumps({"error": str(error)})

        parameters = {"type": "object", "properties": {"run_id": RUN_ID, **properties}, "required": ["run_id", *properties]}
        registry.register(
            name=name,
            toolset=TOOLSET,
            schema={"name": name, "description": description, "parameters": parameters},
            handler=handler,
        )
