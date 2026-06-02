import os

from openai import OpenAI
from openai.types.chat import ChatCompletion


class OpenAIService:
    """Wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = model

    def add_user_message(self, messages: list, message):
        """Append a user message. If message is a list of tool result dicts,
        each is appended individually as a role=tool message."""
        if isinstance(message, list):
            # List of {"role": "tool", "tool_call_id": ..., "content": ...}
            messages.extend(message)
        else:
            messages.append({"role": "user", "content": message})

    def add_assistant_message(self, messages: list, response: ChatCompletion):
        """Append the assistant turn from a ChatCompletion response."""
        # Store the message object as a dict so it can be serialised back
        msg = response.choices[0].message
        messages.append(msg.model_dump())

    def text_from_message(self, response: ChatCompletion) -> str:
        """Extract plain text from a ChatCompletion response."""
        content = response.choices[0].message.content
        return content or ""

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=None,
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> ChatCompletion:
        # Build the message list, optionally prepending a system message
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": full_messages,
            "temperature": temperature,
        }

        if stop_sequences:
            params["stop"] = stop_sequences

        # Map thinking/budget_tokens → OpenAI's reasoning_effort.
        if thinking:
            if thinking_budget < 512:
                params["reasoning_effort"] = "low"
            elif thinking_budget <= 4096:
                params["reasoning_effort"] = "medium"
            else:
                params["reasoning_effort"] = "high"

        # Convert tools to OpenAI function-calling format
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]

        return self.client.chat.completions.create(**params)
