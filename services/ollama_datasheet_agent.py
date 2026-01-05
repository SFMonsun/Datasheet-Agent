"""
Ollama Datasheet Agent - Analyzes individual datasheets using Ollama
"""
import aiohttp
import json
from pathlib import Path
from typing import Optional, Dict, Any
import PyPDF2


class OllamaDatasheetAgent:
    """Agent responsible for analyzing a single datasheet using Ollama"""

    def __init__(self, datasheet_id: str, filename: str, filepath: Path, model: str = "qwen2.5:3b", ollama_url: str = "http://localhost:11434"):
        self.datasheet_id = datasheet_id
        self.filename = filename
        self.filepath = filepath
        self.model = model
        self.ollama_url = ollama_url
        self.datasheet_content = None
        self.component_info = None  # Will store component name, description, and type
        self.system_prompt = f"""You are a Datasheet Analysis Agent specialized in understanding electronic component datasheets.

Your datasheet: {filename}

Your responsibilities:
1. Fully understand all specifications, parameters, and characteristics in this datasheet
2. Answer questions accurately about this component
3. Provide relevant information for PCB design decisions
4. Highlight important electrical characteristics, pin configurations, and operating conditions

When asked a question:
- Be precise and cite specific values from the datasheet
- If information isn't in the datasheet, clearly state that
- Focus on practical engineering considerations
"""

    async def initialize(self) -> bool:
        """Load and process the datasheet PDF"""
        try:
            # Extract text from PDF
            self.datasheet_content = self._extract_pdf_text()
            content_length = len(self.datasheet_content) if self.datasheet_content else 0
            print(f"[DatasheetAgent] Initialized {self.filename}: extracted {content_length} characters")

            # Analyze and identify the component
            await self._identify_component()

            return True
        except Exception as e:
            print(f"[DatasheetAgent] Error initializing datasheet agent for {self.filename}: {e}")
            return False

    async def _identify_component(self) -> None:
        """Analyze datasheet to identify what component this is"""
        try:
            if not self.datasheet_content:
                self.component_info = {
                    'name': self.filename.replace('.pdf', ''),
                    'type': 'Unknown',
                    'description': 'No content available'
                }
                return

            # Use first 3000 characters for identification
            identification_content = self.datasheet_content[:3000]

            identification_prompt = f"""Analyze this datasheet excerpt and provide a brief identification.

Datasheet excerpt:
{identification_content}

Provide ONLY the following information in this exact format:
Component Name: [e.g., LM75B, INA219, etc.]
Component Type: [e.g., Temperature Sensor, Current Sensor, Microcontroller, etc.]
Brief Description: [One sentence describing what this component does]

Your response:"""

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": identification_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_ctx": 2048
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')

                        # Parse the response
                        lines = response_text.strip().split('\n')
                        name = self.filename.replace('.pdf', '')
                        comp_type = 'Unknown'
                        description = 'Electronic component'

                        for line in lines:
                            if 'Component Name:' in line:
                                name = line.split('Component Name:')[1].strip()
                            elif 'Component Type:' in line:
                                comp_type = line.split('Component Type:')[1].strip()
                            elif 'Brief Description:' in line or 'Description:' in line:
                                description = line.split('Description:')[1].strip()

                        self.component_info = {
                            'name': name,
                            'type': comp_type,
                            'description': description
                        }

                        print(f"[DatasheetAgent] Identified component: {name} - {comp_type}")
                    else:
                        # Fallback
                        self.component_info = {
                            'name': self.filename.replace('.pdf', ''),
                            'type': 'Unknown',
                            'description': 'Electronic component'
                        }
        except Exception as e:
            print(f"[DatasheetAgent] Error identifying component: {e}")
            self.component_info = {
                'name': self.filename.replace('.pdf', ''),
                'type': 'Unknown',
                'description': 'Electronic component'
            }

    def get_component_info(self) -> Dict[str, str]:
        """Get the identified component information"""
        return self.component_info if self.component_info else {
            'name': self.filename.replace('.pdf', ''),
            'type': 'Unknown',
            'description': 'Not yet analyzed'
        }

    def _extract_pdf_text(self) -> str:
        """Extract text content from PDF"""
        try:
            with open(self.filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []

                # Extract text from each page (limit to first 20 pages for local models)
                max_pages = min(20, len(pdf_reader.pages))
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    text_content.append(f"--- Page {page_num + 1} ---\n{page.extract_text()}")

                return "\n\n".join(text_content)
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query this datasheet agent with a specific question

        Args:
            question: The question to ask about this datasheet

        Returns:
            Dict with 'response' and 'metadata'
        """
        try:
            print(f"[DatasheetAgent] {self.filename} received query: {question}")
            if not self.datasheet_content:
                print(f"[DatasheetAgent] {self.filename} not initialized!")
                return {
                    'response': f"Datasheet {self.filename} has not been initialized yet.",
                    'metadata': {'error': 'not_initialized'}
                }

            # Create prompt with datasheet context (truncated for local models)
            max_context = 6000  # characters
            truncated_content = self.datasheet_content[:max_context]
            if len(self.datasheet_content) > max_context:
                truncated_content += "\n\n[Datasheet truncated for processing...]"

            prompt = f"""{self.system_prompt}

Datasheet Content:
{truncated_content}

Question: {question}

Please answer based on the datasheet content above."""

            # Call Ollama API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_ctx": 4096
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')

                        return {
                            'response': response_text,
                            'metadata': {
                                'datasheet_id': self.datasheet_id,
                                'filename': self.filename,
                                'model': self.model,
                                'eval_count': result.get('eval_count', 0),
                                'eval_duration': result.get('eval_duration', 0)
                            }
                        }
                    else:
                        return {
                            'response': f"Error calling Ollama: {response.status}",
                            'metadata': {'error': f"HTTP {response.status}"}
                        }

        except aiohttp.ClientError:
            return {
                'response': "Cannot connect to Ollama. Please ensure Ollama is running (ollama serve).",
                'metadata': {'error': 'connection_error'}
            }
        except Exception as e:
            print(f"Error querying Ollama datasheet agent: {e}")
            return {
                'response': f"Error querying datasheet: {str(e)}",
                'metadata': {'error': str(e)}
            }
