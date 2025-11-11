"""ReAct agent implementation (one-shot)."""

from typing import Any

from .base import MathAgent, AgentResult
from ..core.problem import Problem


class ReActAgent(MathAgent):
    """ReAct agent: Thought → Action → Observation loop."""

    REACT_PROMPT = """You are a math problem solver. Solve this step-by-step.

   Available tools:
   {tool_descriptions}

   Problem: {question}

   Use this format:
   Thought: [your reasoning]
   Action: [tool_name]
   Action Input: {{"param": "value"}}

   Or to give final answer:
   Thought: I now know the final answer
   Final Answer: [answer]

   Begin!
   """

    category = "one-shot"

    def solve(self, problem: Problem, tools: dict[str, Any]) -> AgentResult:
        """Solve problem using ReAct loop.

        Process:
        1. Generate thought about next step
        2. Choose action (tool call or answer)
        3. Observe result
        4. Repeat until answer or max_steps
        """
        # TODO: Implement ReAct loop
        # - Build prompt with problem + tool descriptions
        # - Parse LLM response for thought/action
        # - Execute tool calls
        # - Track tokens/cost
        # - Return AgentResult
        raise NotImplementedError
        # Still need to flesh out
        trace = []
        total_tokens = 0
        total_cost = 0.0

        # make tool desc
        tool_desc = "\n".join(
            [f"{name}: {info['description']}" for name, info in tools.items()]
        )

        messages = [
            {
                "role": "user",
                "content": self.REACT_PROMPT.format(
                    tool_descriptions=tool_desc, question=problem.question
                ),
            }
        ]
        for step in range(self.max_steps):
            response = self.llm.complete(messages)
            content = response["content"]
            total_tokens += response["usage"].get("total_tokens", 0)

            if "Final Answer:" in content:
                answer = content.split("Final Answer:")[1].strip()
                trace.append(
                    {
                        "step": step,
                        "thought": content,
                        "action": "finish",
                        "observation": None,
                    }
                )
                break

            action, action_input = self._parse_action(content)

            observation = self._execute_tool(action, action_input, tools)

            trace.append(
                {
                    "step": step,
                    "thought": content,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                }
            )

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return AgentResult(
            answer=answer,
            trace=trace,
            cost=total_cost,
            tokens=total_tokens,
            metadata={"steps": len(trace)},
        )
