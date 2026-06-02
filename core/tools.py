import json
from typing import Optional, List
from mcp.types import CallToolResult, Tool, TextContent
from mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict]:
        """Gets all tools from the provided clients in OpenAI function-call format."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that has the specified tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    def _build_tool_result_message(
        cls,
        tool_call_id: str,
        content: str,
    ) -> dict:
        """Builds an OpenAI tool result message dict."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], response
    ) -> List[dict]:
        """Executes all tool calls from a ChatCompletion response and returns
        a list of role=tool message dicts."""
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        tool_result_messages: list[dict] = []
        for tc in tool_calls:
            tool_call_id = tc.id
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                tool_result_messages.append(
                    cls._build_tool_result_message(
                        tool_call_id, "Error: Could not find that tool"
                    )
                )
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                items = tool_output.content if tool_output else []
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_str = json.dumps(content_list)

                if tool_output and tool_output.isError:
                    content_str = f"Error: {content_str}"

                tool_result_messages.append(
                    cls._build_tool_result_message(tool_call_id, content_str)
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_messages.append(
                    cls._build_tool_result_message(
                        tool_call_id, json.dumps({"error": error_message})
                    )
                )

        return tool_result_messages
