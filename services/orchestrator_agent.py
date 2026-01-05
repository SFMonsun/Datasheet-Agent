"""
Orchestrator Agent - Main agent that coordinates with datasheet agents
"""
import anthropic
from typing import List, Dict, Any, Optional
from services.datasheet_agent import DatasheetAgent
import json


class OrchestratorAgent:
    """Main agent that helps with PCB design by coordinating datasheet agents"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.datasheet_agents: Dict[str, DatasheetAgent] = {}
        self.conversation_history: List[Dict[str, str]] = []

    async def register_datasheet_agent(self, datasheet_agent: DatasheetAgent):
        """Register a datasheet agent for a specific component"""
        self.datasheet_agents[datasheet_agent.datasheet_id] = datasheet_agent
        await datasheet_agent.initialize()

    def unregister_datasheet_agent(self, datasheet_id: str):
        """Remove a datasheet agent"""
        if datasheet_id in self.datasheet_agents:
            del self.datasheet_agents[datasheet_id]

    def _get_available_components_context(self) -> str:
        """Get list of available components for the orchestrator"""
        if not self.datasheet_agents:
            return "No components loaded yet."

        components = []
        for agent_id, agent in self.datasheet_agents.items():
            components.append(f"- {agent.filename} (ID: {agent_id})")

        return "Available components:\n" + "\n".join(components)

    async def _query_datasheet_agent(self, datasheet_id: str, question: str) -> str:
        """Query a specific datasheet agent"""
        if datasheet_id not in self.datasheet_agents:
            return f"Error: Datasheet agent {datasheet_id} not found."

        agent = self.datasheet_agents[datasheet_id]
        result = await agent.query(question)
        return f"From {agent.filename}:\n{result['response']}"

    async def chat(self, user_message: str, max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Main chat interface - orchestrator decides when to query datasheet agents

        Args:
            user_message: User's question or request
            max_tokens: Maximum tokens for response

        Returns:
            Dict with 'response', 'tool_calls', and 'metadata'
        """
        try:
            # Build system prompt with available tools
            system_prompt = f"""You are a PCB Design Assistant that helps engineers design printed circuit boards.

{self._get_available_components_context()}

When a user asks about specific components or needs detailed specifications:
1. Identify which datasheet(s) contain relevant information
2. Query the appropriate datasheet agent(s) using the query_datasheet tool
3. Synthesize the information to provide comprehensive PCB design guidance

You have access to specialized datasheet agents. When you need specific information from a datasheet, you should query the relevant agent.

Focus on:
- Component compatibility and integration
- Power requirements and distribution
- Signal integrity considerations
- Pin configurations and connections
- PCB layout recommendations
- Component placement suggestions

Be practical and engineering-focused in your responses."""

            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })

            # Define tools for Claude
            tools = [
                {
                    "name": "query_datasheet",
                    "description": "Query a specific datasheet agent to get detailed information about a component. Use this when you need specific technical details, specifications, or parameters from a datasheet.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "datasheet_id": {
                                "type": "string",
                                "description": "The ID of the datasheet to query (from available components list)"
                            },
                            "question": {
                                "type": "string",
                                "description": "The specific question to ask about this datasheet"
                            }
                        },
                        "required": ["datasheet_id", "question"]
                    }
                }
            ]

            # Initial API call
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tools,
                messages=self.conversation_history
            )

            # Process tool calls if any
            tool_results = []
            while response.stop_reason == "tool_use":
                # Extract tool calls
                tool_uses = [block for block in response.content if block.type == "tool_use"]

                # Execute each tool call
                for tool_use in tool_uses:
                    if tool_use.name == "query_datasheet":
                        datasheet_id = tool_use.input["datasheet_id"]
                        question = tool_use.input["question"]

                        # Query the datasheet agent
                        result = await self._query_datasheet_agent(datasheet_id, question)
                        tool_results.append({
                            "tool_use_id": tool_use.id,
                            "result": result
                        })

                # Add assistant response and tool results to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })

                self.conversation_history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tr["tool_use_id"],
                            "content": tr["result"]
                        }
                        for tr in tool_results
                    ]
                })

                # Continue conversation with tool results
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    system=system_prompt,
                    tools=tools,
                    messages=self.conversation_history
                )

                tool_results = []  # Reset for next iteration

            # Extract final text response
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text

            # Add final response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })

            return {
                'response': final_response,
                'metadata': {
                    'model': response.model,
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens
                    }
                }
            }

        except Exception as e:
            print(f"Error in orchestrator chat: {e}")
            return {
                'response': f"I encountered an error: {str(e)}",
                'metadata': {'error': str(e)}
            }

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
