"""FastAPI server entry point for RLM-Forge environment.

This module provides the standardized OpenEnv server entry point.
It wraps the rlm_forge.server.app module for multi-mode deployment.

Usage:
    uv run server
    python -m server.app
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from rlm_forge.server.app import app  # noqa: F401


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
