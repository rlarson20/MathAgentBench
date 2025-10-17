"""Docker-isolated Python REPL."""

import docker
from docker.errors import ImageNotFound, ContainerError
from typing import Any


class DockerREPL:
    """Executes Python code in isolated Docker container."""

    DEPS: list[str] = ["sympy", "numpy", "scipy"]

    def __init__(
        self, timeout: int = 30, image: str = "math-agent-bench-sandbox:latest"
    ):
        self.timeout: int = timeout
        self.image: str = image
        self._client = docker.from_env()
        self._ensure_image(self._client, self.image)

    @staticmethod
    def _ensure_image(client: docker.DockerClient, image: str) -> None:
        """Ensure Docker image is available."""
        try:
            client.images.get(image)
        except ImageNotFound:
            print(f"Building {image}...")
            client.images.build(
                path=".",
                dockerfile="Sandbox.Dockerfile",
                tag=image,
                rm=True,
            )

    def execute(self, code: str) -> dict[str, Any]:
        """Execute code in container.

        Args:
            code: Python code to execute

        Returns:
            Dict with stdout, stderr, success flag
        """
        try:
            container = self._client.containers.run(
                self.image,
                command=["uv", "run", "python", "-c", code],
                detach=True,
                mem_limit="1024m",
                network_disabled=True,
            )
            result = container.wait(timeout=self.timeout)
            stdout = container.logs(stdout=True, stderr=False).decode()
            stderr = container.logs(stdout=False, stderr=True).decode()

            container.remove()

            return {
                "success": result["StatusCode"] == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result["StatusCode"],
            }

        except ContainerError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": e.exit_status,
            }
        except ImageNotFound as e:
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
