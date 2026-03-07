"""Pydantic models for RLM-Forge environment."""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class RLMForgeAction(Action):
    """Agent submits Python code to execute in the REPL."""

    code: str = Field(..., description="Python code to execute in the REPL environment")
    action_type: str = Field(
        default="execute",
        description="Type of action: 'execute' for code, 'final' to submit solution",
    )


class RLMForgeObservation(Observation):
    """What the agent sees after each step.

    Inherits from Observation base:
      done: bool = False
      reward: Optional[float] = None
      metadata: Dict[str, Any] = {}
    """

    stdout: str = Field(default="", description="Truncated stdout from code execution")
    stderr: str = Field(default="", description="Truncated stderr from code execution")
    success: bool = Field(default=True, description="Whether code executed without errors")
    iteration: int = Field(default=0, description="Current iteration number")
    max_iterations: int = Field(default=10, description="Maximum allowed iterations")
    repo_manifest: Optional[dict] = Field(
        default=None, description="Repository structure manifest"
    )
    task_description: Optional[str] = Field(
        default=None, description="The coding task to complete"
    )
    failing_tests: Optional[list[str]] = Field(
        default=None, description="List of currently failing test names"
    )
    available_functions: list[str] = Field(
        default_factory=list, description="Built-in functions available in the REPL"
    )
    test_results: Optional[dict] = Field(
        default=None, description="Detailed test results on completion"
    )


class RLMForgeState(State):
    """Internal environment state, not directly sent to agent.

    Inherits from State base:
      episode_id: Optional[str] = None
      step_count: int = 0
    """

    repo_url: str = ""
    repo_local_path: str = ""
    removed_feature_path: str = ""
    removed_feature_content: str = ""
    target_test_files: list[str] = Field(default_factory=list)
    baseline_test_count: int = 0
    files_written: dict[str, str] = Field(default_factory=dict)
    sub_agents_spawned: int = 0
    final_reward: Optional[float] = None
