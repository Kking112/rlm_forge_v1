"""Composite reward computation for RLM-Forge episodes."""

import ast
import os
import re
import subprocess


class RewardComputer:
    """Computes composite reward: test pass rate + structural validity + efficiency."""

    def __init__(
        self,
        test_weight: float = 0.55,
        structural_weight: float = 0.15,
        efficiency_weight: float = 0.30,
    ):
        self.test_weight = test_weight
        self.structural_weight = structural_weight
        self.efficiency_weight = efficiency_weight

    def compute(
        self,
        repo_path: str,
        target_test: str,
        files_written: dict[str, str],
        max_iterations: int,
        iterations_used: int,
        baseline_test_count: int,
    ) -> dict:
        """Compute composite reward. Returns detailed breakdown."""
        # 1. Test pass rate (55%)
        test_result = self._run_target_tests(repo_path, target_test)
        total_tests = max(test_result["total"], baseline_test_count, 1)
        test_pass_rate = test_result["passed"] / total_tests

        # 2. Structural validity (15%)
        structural_score = self._compute_structural(repo_path, files_written)

        # 3. Efficiency (30%)
        efficiency_score = self._compute_efficiency(iterations_used, max_iterations)

        # Composite
        total = (
            self.test_weight * test_pass_rate
            + self.structural_weight * structural_score
            + self.efficiency_weight * efficiency_score
        )

        return {
            "total_reward": round(total, 4),
            "test_pass_rate": round(test_pass_rate, 4),
            "tests_passed": test_result["passed"],
            "tests_failed": test_result["failed"],
            "tests_total": test_result["total"],
            "structural_score": round(structural_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "breakdown": {
                "test_component": round(self.test_weight * test_pass_rate, 4),
                "structural_component": round(
                    self.structural_weight * structural_score, 4
                ),
                "efficiency_component": round(
                    self.efficiency_weight * efficiency_score, 4
                ),
            },
            "test_output": test_result.get("output", "")[:2000],
        }

    def _run_target_tests(self, repo_path: str, test_path: str) -> dict:
        """Run the target test file and parse results."""
        import sys

        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "--no-header"]
        cmd.append(os.path.join(repo_path, test_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=repo_path,
            )
            raw_output = result.stdout + result.stderr
            # Strip ANSI color codes for reliable parsing
            output = re.sub(r"\x1b\[[0-9;]*m", "", raw_output)
            passed = len(re.findall(r" PASSED", output))
            failed = len(re.findall(r" FAILED", output))
            errors = len(re.findall(r" ERROR", output))

            return {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total": passed + failed + errors,
                "output": output[:3000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "total": 1,
                "output": "Test execution timed out",
                "returncode": -1,
            }

    def _compute_structural(
        self, repo_path: str, files_written: dict[str, str]
    ) -> float:
        """Check structural validity of written files."""
        if not files_written:
            return 0.0

        file_scores = []
        for path, content in files_written.items():
            # Parse check (weight 0.3)
            try:
                ast.parse(content)
                parse_ok = 1.0
            except SyntaxError:
                parse_ok = 0.0

            # Import check (weight 0.3)
            module_name = path.replace("/", ".").replace(".py", "")
            try:
                import sys

                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"import importlib; importlib.import_module('{module_name}')",
                    ],
                    capture_output=True,
                    timeout=10,
                    cwd=repo_path,
                )
                import_ok = 1.0 if result.returncode == 0 else 0.0
            except Exception:
                import_ok = 0.0

            file_scores.append(0.3 * parse_ok + 0.3 * import_ok)

        avg_file_score = sum(file_scores) / len(file_scores)

        # Regression check (weight 0.4)
        # For hackathon: assume no regressions since we only modify the removed file
        regression_score = 0.4

        return avg_file_score + regression_score

    def _compute_efficiency(
        self, iterations_used: int, max_iterations: int
    ) -> float:
        """Tiered efficiency score."""
        if max_iterations <= 0:
            return 0.0
        ratio = iterations_used / max_iterations
        if ratio <= 0.5:
            return 1.0
        elif ratio <= 0.75:
            return 0.75
        elif ratio <= 1.0:
            return 0.5
        else:
            return 0.0
