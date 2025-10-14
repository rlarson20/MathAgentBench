# To write: use httpx mock or respx
# test_openrouter_retry_on_rate_limit()
# test_openrouter_token_counting()
# test_openrouter_missing_api_key()
import respx
from httpx import Response

from mathagentbench.llm.client import OpenRouterClient


@respx.mock
def test_openrouter_complete_success():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [{"message": {"content": "42"}}],
                "usage": {"total_tokens": 10},
            },
        )
    )

    client = OpenRouterClient(api_key="test")
    result = client.complete([{"role": "user", "content": "Hi"}])
    assert result["content"] == "42"
