"""Sandboxed Python REPL using exec() with persistent globals."""

import contextlib
import io
import os
import re
import subprocess


class REPLSandbox:
    """Sandboxed Python REPL with built-in tool functions for repo exploration."""

    def __init__(self, repo_path: str, max_output_chars: int = 5000):
        self.repo_path = os.path.realpath(repo_path)
        self.max_output_chars = max_output_chars
        self.files_written: dict[str, str] = {}
        self._final_called = False
        self._sub_agents_spawned = 0

        self.globals_dict: dict = {"__builtins__": __builtins__}
        self.globals_dict.update(
            {
                "read_file": self._read_file,
                "list_dir": self._list_dir,
                "search": self._search,
                "write_file": self._write_file,
                "run_tests": self._run_tests,
                "spawn_agent": self._spawn_agent,
                "FINAL": self._final,
            }
        )

    def execute(self, code: str) -> dict:
        """Execute code in the sandbox, return stdout/stderr/success."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                stderr_capture
            ):
                exec(code, self.globals_dict)
            success = True
        except Exception as e:
            stderr_capture.write(f"{type(e).__name__}: {e}\n")
            success = False

        stdout = stdout_capture.getvalue()[: self.max_output_chars]
        stderr = stderr_capture.getvalue()[: self.max_output_chars]

        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": success,
            "final_called": self._final_called,
        }

    def _validate_path(self, path: str) -> str:
        """Ensure path stays within repo. Returns the real absolute path."""
        full_path = os.path.join(self.repo_path, path)
        real_path = os.path.realpath(full_path)
        if not real_path.startswith(self.repo_path):
            raise PermissionError(f"Access denied: {path}")
        return real_path

    def _read_file(self, path: str) -> str:
        """Read a file from the repo. Path relative to repo root."""
        real_path = self._validate_path(path)
        with open(real_path, "r") as f:
            content = f.read()
        if len(content) > 10000:
            content = content[:10000] + "\n... [truncated]"
        return content

    def _list_dir(self, path: str = ".") -> list[str]:
        """List directory contents relative to repo root."""
        real_path = self._validate_path(path)
        entries = os.listdir(real_path)
        result = []
        for e in sorted(entries):
            full = os.path.join(real_path, e)
            suffix = "/" if os.path.isdir(full) else ""
            result.append(e + suffix)
        return result

    def _search(self, pattern: str, path: str = ".") -> list[str]:
        """Grep for pattern in repo files. Returns list of matches."""
        real_path = self._validate_path(path)
        results = []
        try:
            output = subprocess.run(
                ["grep", "-rn", "--include=*.py", pattern, real_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in output.stdout.strip().split("\n")[:50]:
                if line:
                    results.append(line.replace(self.repo_path + "/", ""))
        except (subprocess.TimeoutExpired, Exception):
            pass
        return results

    def _write_file(self, path: str, content: str) -> str:
        """Write a file to the repo. Records it for evaluation."""
        real_path = self._validate_path(path)
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        with open(real_path, "w") as f:
            f.write(content)
        self.files_written[path] = content
        return f"Written {len(content)} chars to {path}"

    def _run_tests(self, test_path: str | None = None) -> dict:
        """Run pytest on specified test file(s). Returns pass/fail summary."""
        import sys

        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "--no-header"]
        if test_path:
            cmd.append(os.path.join(self.repo_path, test_path))
        else:
            cmd.append(self.repo_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.repo_path,
            )
            raw_output = result.stdout + result.stderr
            # Strip ANSI color codes for reliable parsing
            output = re.sub(r"\x1b\[[0-9;]*m", "", raw_output)
            passed = len(re.findall(r" PASSED", output))
            failed = len(re.findall(r" FAILED", output))
            errors = len(re.findall(r" ERROR", output))
            output_truncated = output[: self.max_output_chars]

            return {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total": passed + failed + errors,
                "output": output_truncated,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "total": 1,
                "output": "Test execution timed out (60s limit)",
                "returncode": -1,
            }

    def _spawn_agent(self, scope: str, mission: str, budget: int = 5) -> dict:
        """Stateless sub-LM call. Gathers scoped context and returns structured report."""
        self._sub_agents_spawned += 1
        scope_path = os.path.join(self.repo_path, scope)

        if not os.path.exists(scope_path):
            return {
                "error": f"Scope path not found: {scope}",
                "summary": "",
                "files_examined": [],
            }

        # Build file listing for the scope
        files = []
        for root, dirs, filenames in os.walk(scope_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in filenames:
                if f.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, f), self.repo_path)
                    files.append(rel)

        # Read first few files to build context
        context_parts = []
        for fpath in files[:5]:
            try:
                content = self._read_file(fpath)
                context_parts.append(f"--- {fpath} ---\n{content[:2000]}")
            except Exception:
                pass

        report = {
            "summary": (
                f"Explored scope '{scope}' for mission: {mission}. "
                f"Found {len(files)} Python files."
            ),
            "files_examined": files[:10],
            "file_contents_preview": context_parts[:3],
            "mission": mission,
        }
        return report

    def _final(self) -> str:
        """Signal episode completion."""
        self._final_called = True
        return "Episode marked as complete. Evaluating..."

    @property
    def available_functions(self) -> list[str]:
        return [
            "read_file(path)",
            "list_dir(path='.')",
            "search(pattern, path='.')",
            "write_file(path, content)",
            "run_tests(test_path=None)",
            "spawn_agent(scope, mission, budget=5)",
            "FINAL()",
        ]
