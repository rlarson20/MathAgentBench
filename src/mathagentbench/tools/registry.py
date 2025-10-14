"""Tool registry and descriptions."""

from .docker_repl import DockerREPL
from .symbolic import symbolic_math_handler


# Global tool registry
TOOLS = {
    "python": {
        "description": (
            "Execute Python code with sympy, numpy, scipy available. "
            "Returns the result of the last expression or print output. "
            "Use for numerical calculations, symbolic math via sympy."
        ),
        "parameters": {"code": "string - Python code to execute"},
        "handler": DockerREPL().execute,
    },
    "symbolic_math": {
        "description": (
            "Perform symbolic mathematics operations using sympy. "
            "Operations: solve, simplify, diff, integrate, expand, factor."
        ),
        "parameters": {
            "operation": "string - Operation to perform",
            "expression": "string - Mathematical expression",
            "variable": "string (optional) - Variable for operations like diff/integrate",
        },
        "handler": symbolic_math_handler,
    },
}


def get_tool(name: str):
    """Get tool handler by name."""
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOLS[name]["handler"]
