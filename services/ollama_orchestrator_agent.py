"""Ollama Orchestrator Agent"""
import aiohttp
import asyncio
from typing import List, Dict, Any
from services.ollama_datasheet_agent import OllamaDatasheetAgent


class OllamaOrchestratorAgent:
    def __init__(self, model: str = "qwen2.5:3b", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self.datasheet_agents: Dict[str, OllamaDatasheetAgent] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.agent_interactions: List[Dict[str, Any]] = []
        self.rag_steps: List[Dict[str, Any]] = []  # Collect RAG steps from agents

    async def register_datasheet_agent(self, datasheet_agent: OllamaDatasheetAgent):
        """Register a new datasheet agent and record the registration interaction"""
        await datasheet_agent.initialize()
        self.datasheet_agents[datasheet_agent.datasheet_id] = datasheet_agent

        # Record the initial registration interaction
        comp_info = datasheet_agent.get_component_info()
        registration_message = f"I am responsible for {comp_info['name']} ({comp_info['type']}): {comp_info['description']}"

        print(f"[Orchestrator] Agent registered: {datasheet_agent.filename} - {comp_info['name']}")

        # Store this as a special registration interaction (will be shown differently in UI)
        if not hasattr(self, 'registration_interactions'):
            self.registration_interactions = []

        self.registration_interactions.append({
            'type': 'registration',
            'agent_id': datasheet_agent.datasheet_id,
            'filename': datasheet_agent.filename,
            'from': datasheet_agent.filename,
            'to': 'Orchestrator',
            'component_info': comp_info,
            'message': registration_message
        })

    def unregister_datasheet_agent(self, datasheet_id: str):
        if datasheet_id in self.datasheet_agents:
            del self.datasheet_agents[datasheet_id]

    def _get_available_components_context(self) -> str:
        """Get context about available components with their identified information"""
        if not self.datasheet_agents:
            return "No components loaded yet."
        components = []
        for agent_id, agent in self.datasheet_agents.items():
            comp_info = agent.get_component_info()
            components.append(f"- {comp_info['name']} ({comp_info['type']}): {comp_info['description']}")
        return "Available components:\n" + "\n".join(components)

    async def _query_datasheet_agent(self, datasheet_id: str, question: str) -> str:
        """Query a specific datasheet agent"""
        if datasheet_id not in self.datasheet_agents:
            return f"Error: Datasheet agent {datasheet_id} not found."

        agent = self.datasheet_agents[datasheet_id]

        # Record the query interaction
        self.agent_interactions.append({
            'type': 'query',
            'from': 'Orchestrator',
            'to': agent.filename,
            'question': question
        })

        result = await agent.query(question)

        # Collect RAG steps if available (from RAG-enhanced agents)
        if 'rag_steps' in result and result['rag_steps']:
            self.rag_steps.extend(result['rag_steps'])

        # Record the response interaction
        self.agent_interactions.append({
            'type': 'response',
            'from': agent.filename,
            'to': 'Orchestrator'
        })

        return f"From {agent.filename}:\n{result['response']}"

    async def _determine_relevant_datasheets(self, user_message: str) -> List[str]:
        """Determine which datasheets are relevant to the user's question using component info"""
        if not self.datasheet_agents:
            return []

        relevant_ids = []
        user_message_lower = user_message.lower()

        for agent_id, agent in self.datasheet_agents.items():
            comp_info = agent.get_component_info()

            # Check multiple criteria for relevance
            matches = []

            # Check component name (must be exact word match to avoid partial matches)
            comp_name_lower = comp_info['name'].lower()
            # Split by word boundaries to check for exact component name
            if comp_name_lower in user_message_lower:
                # Verify it's a word boundary match, not just substring
                import re
                if re.search(r'\b' + re.escape(comp_name_lower) + r'\b', user_message_lower):
                    matches.append(f"component name '{comp_info['name']}'")

            # Check filename (without extension) - exact word match
            filename_base = agent.filename.lower().replace('.pdf', '').replace('.txt', '').replace('.json', '')
            if re.search(r'\b' + re.escape(filename_base) + r'\b', user_message_lower):
                matches.append(f"filename '{filename_base}'")

            # Only add to relevant if we have matches
            if matches:
                relevant_ids.append(agent_id)
                print(f"[Orchestrator] Matched {agent.filename} ({comp_info['name']}) - matched: {', '.join(matches)}")

        # If we found specific matches, return only those
        if relevant_ids:
            return relevant_ids

        # If no specific match, don't query any agents - let the orchestrator answer generally
        print(f"[Orchestrator] No specific component match found, orchestrator will answer without querying agents")
        return []

    async def chat(self, user_message: str) -> Dict[str, Any]:
        try:
            self.agent_interactions = []
            self.rag_steps = []  # Clear previous RAG steps

            # Determine which datasheets are relevant
            relevant_datasheet_ids = await self._determine_relevant_datasheets(user_message)
            print(f"[Orchestrator] Found {len(relevant_datasheet_ids)} relevant datasheets")

            # Query relevant datasheet agents
            datasheet_responses = []
            if relevant_datasheet_ids:
                for datasheet_id in relevant_datasheet_ids:
                    print(f"[Orchestrator] Querying datasheet agent: {datasheet_id}")
                    response = await self._query_datasheet_agent(datasheet_id, user_message)
                    datasheet_responses.append(response)
                print(f"[Orchestrator] Collected {len(datasheet_responses)} responses from agents")
                print(f"[Orchestrator] Agent interactions recorded: {len(self.agent_interactions)}")

            # Build context with datasheet information
            datasheet_context = ""
            if datasheet_responses:
                datasheet_context = "\n\nInformation from datasheets:\n" + "\n\n".join(datasheet_responses)

            system_prompt = f"""You are a PCB Design Assistant.

{self._get_available_components_context()}

Help with PCB design questions. Be practical and engineering-focused.
{datasheet_context}"""

            prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'response': result.get('response', ''),
                            'agent_interactions': self.agent_interactions,
                            'rag_steps': self.rag_steps,
                            'metadata': {'model': self.model}
                        }
                    else:
                        return {
                            'response': f"Error: HTTP {response.status}",
                            'agent_interactions': [],
                            'metadata': {'error': True}
                        }

        except aiohttp.ClientError:
            return {
                'response': "Cannot connect to Ollama. Ensure it's running (ollama serve).",
                'agent_interactions': [],
                'metadata': {'error': 'connection_error'}
            }
        except Exception as e:
            return {
                'response': f"Error: {str(e)}",
                'agent_interactions': [],
                'metadata': {'error': str(e)}
            }

    def clear_history(self):
        self.conversation_history = []
