from api_common.assets.common import PipelineRequest, PipelineState


class Stage:
    name: str  # the PipelineState field this stage fills in

    def run(self, state: PipelineState) -> PipelineState:
        raise NotImplementedError


def business_context(request: PipelineRequest) -> str:
    parts = {
        "Business idea": request.business_idea,
        "Target audience": request.target_audience,
        "Additional info": request.additional_info,
        "Business plan": request.business_plan,
    }
    return "\n\n".join(f"{title}:\n{text}" for title, text in parts.items() if text)
