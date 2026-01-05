"""
Ollama Datasheet Agent with RAG - Enhanced version using Retrieval-Augmented Generation
"""
import aiohttp
from pathlib import Path
from typing import Dict, Any
from services.rag_service import RAGService


class OllamaDatasheetAgentRAG:
    """RAG-enhanced agent for analyzing datasheets using Ollama"""

    def __init__(self, datasheet_id: str, filename: str, filepath: Path, model: str = "qwen2.5:3b", ollama_url: str = "http://localhost:11434"):
        self.datasheet_id = datasheet_id
        self.filename = filename
        self.filepath = filepath
        self.model = model
        self.ollama_url = ollama_url
        self.component_info = None
        self.rag_service = None
        self.system_prompt = f"""You are a Datasheet Analysis Agent specialized in understanding electronic component datasheets.

Your datasheet: {filename}

Your responsibilities:
1. Fully understand all specifications, parameters, and characteristics in this datasheet
2. Answer questions accurately about this component with VERIFIABLE citations
3. Provide relevant information for PCB design decisions
4. Highlight important electrical characteristics, pin configurations, and operating conditions

CRITICAL CITATION RULES:
- ALWAYS add "(Page X)" after EVERY specific fact, value, or specification
- Multiple pages: Use "(Pages X, Y)" when information spans multiple pages
- Be precise and cite specific values from the datasheet
- If information isn't in the provided excerpts, clearly state that
- Focus on practical engineering considerations
"""

    async def initialize(self) -> bool:
        """Load and process the datasheet PDF using RAG"""
        try:
            print(f"[DatasheetAgentRAG] Initializing RAG pipeline for {self.filename}")

            # Initialize RAG service
            self.rag_service = RAGService(self.datasheet_id)

            # Extract PDF content
            content = self.rag_service.extract_pdf_content(self.filepath)

            # Chunk the content
            self.rag_service.chunk_content(content)

            # Create embeddings
            self.rag_service.create_embeddings()

            # Analyze and identify the component using first chunk
            await self._identify_component()

            print(f"[DatasheetAgentRAG] Initialization complete for {self.filename}")
            return True

        except Exception as e:
            print(f"[DatasheetAgentRAG] Error initializing: {e}")
            return False

    async def _identify_component(self) -> None:
        """Analyze datasheet to identify what component this is"""
        try:
            if not self.rag_service or not self.rag_service.chunks:
                self.component_info = {
                    'name': self.filename.replace('.pdf', ''),
                    'type': 'Unknown',
                    'description': 'No content available'
                }
                return

            # Use first few chunks for identification
            identification_content = "\n\n".join([
                chunk['content'] for chunk in self.rag_service.chunks[:3]
            ])[:3000]

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

                        print(f"[DatasheetAgentRAG] Identified component: {name} - {comp_type}")
                    else:
                        # Fallback
                        self.component_info = {
                            'name': self.filename.replace('.pdf', ''),
                            'type': 'Unknown',
                            'description': 'Electronic component'
                        }

        except Exception as e:
            print(f"[DatasheetAgentRAG] Error identifying component: {e}")
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

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query this datasheet agent with a specific question using RAG

        Args:
            question: The question to ask about this datasheet

        Returns:
            Dict with 'response', 'metadata', and 'rag_steps'
        """
        try:
            print(f"[DatasheetAgentRAG] {self.filename} received query: {question}")

            if not self.rag_service:
                print(f"[DatasheetAgentRAG] {self.filename} not initialized!")
                return {
                    'response': f"Datasheet {self.filename} has not been initialized yet.",
                    'metadata': {'error': 'not_initialized'},
                    'rag_steps': []
                }

            # Clear previous RAG steps
            self.rag_service.clear_rag_steps()

            # Retrieve relevant chunks using RAG
            relevant_chunks = self.rag_service.retrieve_relevant_chunks(question, top_k=5)

            if not relevant_chunks:
                return {
                    'response': "I couldn't find relevant information in the datasheet to answer this question.",
                    'metadata': {'error': 'no_relevant_chunks'},
                    'rag_steps': self.rag_service.get_rag_steps()
                }

            # Build context from relevant chunks with clear page markers
            context_parts = []
            page_references = []
            for i, chunk in enumerate(relevant_chunks):
                page_num = chunk['metadata'].get('page', 'N/A')
                context_parts.append(
                    f"[Source {i+1}: Page {page_num}]\n{chunk['content']}\n"
                )
                page_references.append(f"Source {i+1} = Page {page_num}")

            context = "\n".join(context_parts)
            references_summary = ", ".join(page_references)

            # Create prompt with retrieved context and citation instructions
            prompt = f"""{self.system_prompt}

Relevant datasheet excerpts:
{context}

Question: {question}

IMPORTANT INSTRUCTIONS:
1. Answer the question based on the datasheet excerpts above
2. ALWAYS cite your sources by adding "(Page X)" immediately after each piece of information
3. If information comes from multiple pages, cite all relevant pages
4. Use the page numbers shown in the [Source N: Page X] markers above
5. Example format: "The maximum voltage is 5.5V (Page 3). Operating temperature ranges from -40°C to 85°C (Page 5)."
6. If the information is not in the provided excerpts, clearly state that

Your answer with page citations:"""

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
                            "num_ctx": 8192  # Larger context for RAG
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')

                        # Add source references footer
                        unique_pages = sorted(set([chunk['metadata'].get('page', 'N/A') for chunk in relevant_chunks]))
                        pages_str = ', '.join([str(p) for p in unique_pages if p != 'N/A'])

                        if pages_str:
                            source_footer = f"\n\n---\n**Sources Referenced:** Pages {pages_str} in {self.filename}"
                            response_with_sources = response_text + source_footer
                        else:
                            response_with_sources = response_text

                        return {
                            'response': response_with_sources,
                            'metadata': {
                                'datasheet_id': self.datasheet_id,
                                'filename': self.filename,
                                'model': self.model,
                                'chunks_used': len(relevant_chunks),
                                'pages_referenced': unique_pages,
                                'eval_count': result.get('eval_count', 0),
                                'eval_duration': result.get('eval_duration', 0)
                            },
                            'rag_steps': self.rag_service.get_rag_steps()
                        }
                    else:
                        return {
                            'response': f"Error calling Ollama: {response.status}",
                            'metadata': {'error': f"HTTP {response.status}"},
                            'rag_steps': self.rag_service.get_rag_steps()
                        }

        except aiohttp.ClientError:
            return {
                'response': "Cannot connect to Ollama. Please ensure Ollama is running (ollama serve).",
                'metadata': {'error': 'connection_error'},
                'rag_steps': self.rag_service.get_rag_steps() if self.rag_service else []
            }
        except Exception as e:
            print(f"[DatasheetAgentRAG] Error querying: {e}")
            return {
                'response': f"Error querying datasheet: {str(e)}",
                'metadata': {'error': str(e)},
                'rag_steps': self.rag_service.get_rag_steps() if self.rag_service else []
            }
