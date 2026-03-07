"""Core RLM-Forge Environment implementation."""

import os
import random
import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from ..models import RLMForgeAction, RLMForgeObservation, RLMForgeState
from .feature_extractor import CURATED_PAIRS, FeatureExtractor
from .repo_manager import RepoManager
from .reward import RewardComputer
from .sandbox import REPLSandbox


class RLMForgeEnvironment(
    Environment[RLMForgeAction, RLMForgeObservation, RLMForgeState]
):
    """RLM-Forge: Recursive Language Model training environment for coding agents.

    Clones a Python repo, removes a source file with test coverage, and provides
    a multi-step REPL for the agent to explore and rebuild the feature.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        self.repo_manager = RepoManager()
        self.feature_extractor = FeatureExtractor()
        self.reward_computer = RewardComputer()
        self._state = RLMForgeState()
        self._sandbox: Optional[REPLSandbox] = None
        self._max_iterations = 10

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RLMForgeObservation:
        """Clone repo, remove feature, return initial observation."""
        # Clean up previous episode
        if self._state.repo_local_path:
            self.repo_manager.cleanup(self._state.repo_local_path)

        if seed is not None:
            random.seed(seed)

        # Select a curated pair
        pair = random.choice(CURATED_PAIRS)

        # AMENDMENT 2: Use pre-cloned repos if available, else clone from network
        pre_cloned_dir = os.environ.get("RLM_FORGE_PRE_CLONED_DIR", "")
        repo_name = pair["repo_url"].rstrip("/").split("/")[-1]
        pre_cloned_path = os.path.join(pre_cloned_dir, repo_name) if pre_cloned_dir else ""

        if pre_cloned_path and os.path.isdir(pre_cloned_path):
            repo_path = self.repo_manager.copy_pre_cloned(pre_cloned_path)
        else:
            repo_path = self.repo_manager.clone_repo(pair["repo_url"])

        # Install dependencies (best-effort)
        self.repo_manager.install_dependencies(repo_path)

        # Extract feature (remove source file)
        feature = self.feature_extractor.extract_feature(
            repo_path, pair["source_file"], pair["test_file"]
        )

        # Generate manifest
        manifest = self.repo_manager.generate_manifest(repo_path)

        # Create sandbox
        self._sandbox = REPLSandbox(repo_path)

        # Get initial failing test info
        initial_test_result = self._sandbox._run_tests(pair["test_file"])
        failing_tests = [
            f"FAILING: {pair['test_file']} "
            f"({initial_test_result.get('failed', '?')} failures, "
            f"{initial_test_result.get('errors', '?')} errors)"
        ]

        # Initialize state
        self._state = RLMForgeState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            repo_url=pair["repo_url"],
            repo_local_path=repo_path,
            removed_feature_path=pair["source_file"],
            removed_feature_content=feature.original_content,
            target_test_files=[pair["test_file"]],
            baseline_test_count=feature.num_tests,
        )

        return RLMForgeObservation(
            stdout="Environment initialized. Repository cloned and feature removed.",
            stderr="",
            success=True,
            iteration=0,
            max_iterations=self._max_iterations,
            repo_manifest=manifest,
            task_description=feature.task_description,
            failing_tests=failing_tests,
            available_functions=self._sandbox.available_functions,
            done=False,
            reward=None,
        )

    def step(
        self,
        action: RLMForgeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RLMForgeObservation:
        """Execute code in REPL, check for termination, compute reward if done."""
        if self._sandbox is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._state.step_count += 1

        # Check for explicit final action or iteration limit
        if action.action_type == "final":
            return self._finalize_episode()

        if self._state.step_count >= self._max_iterations:
            return self._finalize_episode()

        # Execute code in sandbox
        result = self._sandbox.execute(action.code)

        # Check if FINAL() was called in the code
        if result["final_called"]:
            return self._finalize_episode()

        return RLMForgeObservation(
            stdout=result["stdout"],
            stderr=result["stderr"],
            success=result["success"],
            iteration=self._state.step_count,
            max_iterations=self._max_iterations,
            available_functions=self._sandbox.available_functions,
            done=False,
            reward=None,
        )

    def _finalize_episode(self) -> RLMForgeObservation:
        """Compute reward and return final observation."""
        assert self._sandbox is not None

        reward_result = self.reward_computer.compute(
            repo_path=self._state.repo_local_path,
            target_test=self._state.target_test_files[0],
            files_written=self._sandbox.files_written,
            max_iterations=self._max_iterations,
            iterations_used=self._state.step_count,
            baseline_test_count=self._state.baseline_test_count,
        )

        self._state.final_reward = reward_result["total_reward"]
        self._state.files_written = self._sandbox.files_written
        self._state.sub_agents_spawned = self._sandbox._sub_agents_spawned

        return RLMForgeObservation(
            stdout=f"Episode complete. Reward: {reward_result['total_reward']:.3f}",
            stderr="",
            success=True,
            iteration=self._state.step_count,
            max_iterations=self._max_iterations,
            test_results=reward_result,
            done=True,
            reward=reward_result["total_reward"],
        )

    @property
    def state(self) -> RLMForgeState:
        return self._state

    def close(self):
        """No-op for HTTP singleton. Use cleanup() for explicit teardown."""
        # OpenEnv HTTP server calls close() after each request handler.
        # For singleton mode, we must NOT destroy state here.
        # Actual cleanup happens in reset() (previous episode) or explicit cleanup().
        pass

    def cleanup(self):
        """Explicit teardown: remove cloned repo."""
        if self._state.repo_local_path:
            self.repo_manager.cleanup(self._state.repo_local_path)
            self._state.repo_local_path = ""
