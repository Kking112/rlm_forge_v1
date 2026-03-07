"""Repository cloning, dependency installation, and manifest generation."""

import os
import shutil
import subprocess
import sys
import tempfile


class RepoManager:
    """Manages repository cloning and lifecycle."""

    def __init__(self, cache_dir: str = "/tmp/rlm_forge_repos"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def clone_repo(self, repo_url: str) -> str:
        """Clone repo to a unique temp directory. Returns path."""
        work_dir = tempfile.mkdtemp(dir=self.cache_dir, prefix="rlm_")
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, work_dir],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return work_dir

    def copy_pre_cloned(self, pre_cloned_path: str) -> str:
        """Copy a pre-cloned repo directory for a fresh episode. Returns new path."""
        work_dir = tempfile.mkdtemp(dir=self.cache_dir, prefix="rlm_")
        # Remove the empty temp dir first, then copy
        shutil.rmtree(work_dir)
        shutil.copytree(pre_cloned_path, work_dir)
        return work_dir

    def install_dependencies(self, repo_path: str) -> bool:
        """Best-effort dependency installation using uv pip (falls back to pip)."""
        uv_path = shutil.which("uv")

        # Build install command: prefer uv pip, fall back to sys.executable -m pip
        def _pip_install(args: list[str]) -> bool:
            if uv_path:
                cmd = [uv_path, "pip", "install"] + args
            else:
                cmd = [sys.executable, "-m", "pip", "install"] + args
            try:
                subprocess.run(
                    cmd, capture_output=True, timeout=120, check=True
                )
                return True
            except Exception:
                return False

        # Try pyproject.toml / setup.py first
        has_pyproject = os.path.exists(os.path.join(repo_path, "pyproject.toml"))
        has_setup = os.path.exists(os.path.join(repo_path, "setup.py"))
        if has_pyproject or has_setup:
            if _pip_install(["-e", repo_path]):
                return True

        # Try requirements.txt
        req_file = os.path.join(repo_path, "requirements.txt")
        if os.path.exists(req_file):
            if _pip_install(["-r", req_file]):
                return True

        return False

    def generate_manifest(self, repo_path: str) -> dict:
        """Generate a high-level manifest of the repo structure."""
        manifest: dict = {"files": [], "total_files": 0, "total_loc": 0}

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and d != "__pycache__"
            ]
            for f in files:
                if f.endswith(".py"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, repo_path)
                    try:
                        with open(full_path) as fh:
                            loc = sum(1 for _ in fh)
                    except Exception:
                        loc = 0
                    manifest["files"].append({"path": rel_path, "loc": loc})
                    manifest["total_files"] += 1
                    manifest["total_loc"] += loc

        # Read README excerpt if available
        for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = os.path.join(repo_path, readme_name)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path) as f:
                        manifest["readme_excerpt"] = f.read()[:2000]
                except Exception:
                    pass
                break

        return manifest

    def cleanup(self, repo_path: str):
        """Remove cloned repo directory."""
        if repo_path and repo_path.startswith(self.cache_dir):
            shutil.rmtree(repo_path, ignore_errors=True)
