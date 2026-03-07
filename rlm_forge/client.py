"""Client for connecting to a remote RLM-Forge environment."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.env_client import StepResult

from .models import RLMForgeAction, RLMForgeObservation, RLMForgeState


class RLMForgeClient(EnvClient[RLMForgeAction, RLMForgeObservation, RLMForgeState]):
    """Client for the RLM-Forge environment."""

    def _step_payload(self, action: RLMForgeAction) -> Dict[str, Any]:
        return {"code": action.code, "action_type": action.action_type}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[RLMForgeObservation]:
        obs = RLMForgeObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> RLMForgeState:
        return RLMForgeState(**payload)
