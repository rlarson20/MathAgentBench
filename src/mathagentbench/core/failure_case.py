from pydantic import BaseModel
from typing import Literal, Any


class FailureCase(BaseModel):
    problem_id: str
    failure_type: Literal["parse_error", "tool_error", "math_error", "timeout"]
    trace: list[dict[Any, Any]]  # TODO: clarify trace datatype
    error_msg: str
