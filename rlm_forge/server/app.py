"""FastAPI server for RLM-Forge environment."""

from openenv.core.env_server import create_app

from ..models import RLMForgeAction, RLMForgeObservation
from .environment import RLMForgeEnvironment

# OpenEnv's HTTP server calls the factory per-request.
# Use a singleton so reset/step share the same environment instance.
_singleton_env = None


def _env_factory():
    global _singleton_env
    if _singleton_env is None:
        _singleton_env = RLMForgeEnvironment()
    return _singleton_env


app = create_app(
    _env_factory,
    RLMForgeAction,
    RLMForgeObservation,
    env_name="rlm_forge",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
