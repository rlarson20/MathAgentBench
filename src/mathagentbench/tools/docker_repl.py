"""Docker-isolated Python REPL."""

import docker
from docker.errors import ImageNotFound
from typing import Any


class DockerREPL:
    """Executes Python code in isolated Docker container."""

    DEPS = ["sympy", "numpy", "scipy"]

    def __init__(self, timeout: int = 30, image: str = "python:3.13-slim"):
        self.timeout = timeout
        self.image = image
        self.client = docker.from_env()
        self._ensure_image()

    def _ensure_image(self) -> None:
        """Ensure Docker image is available."""
        try:
            self.client.images.get(self.image)
        except ImageNotFound:
            print(f"Pulling {self.image}...")
            self.client.images.pull(self.image)

    def execute(self, code: str) -> dict[str, Any]:
        """Execute code in container.

        Args:
            code: Python code to execute

        Returns:
            Dict with stdout, stderr, success flag
        """
        # TODO: Implement container execution
        # - Prepend: pip install -q sympy numpy scipy
        # - Run code with timeout
        # - Capture stdout/stderr
        # - Handle errors gracefully
        raise NotImplementedError
