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
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = docker.from_env()
            self._ensure_image()
        return self._client

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
        setup_script = f"""
import subprocess
import sys

# Install uv first
try:
    import uv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
    import uv

# Install dependencies
subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", "-q"] + {self.DEPS})

{code}
"""
        
        try:
            container = self.client.containers.run(
                self.image,
                command=["python", "-c", setup_script],
                detach=True,
                mem_limit="512m",
                network_disabled=True,
                remove=True,
            )

            result = container.wait(timeout=self.timeout)
            stdout = container.logs(stdout=True, stderr=False).decode()
            stderr = container.logs(stdout=False, stderr=True).decode()

            return {
                "success": result["StatusCode"] == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result["StatusCode"],
            }

        except docker.errors.ContainerError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": e.exit_status,
            }
        except docker.errors.ImageNotFound as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Docker image not found: {e}",
                "exit_code": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
            }
