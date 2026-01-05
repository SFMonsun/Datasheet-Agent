"""
Datasheet Agent - Analyzes individual datasheets using Claude API
"""
import anthropic
from pathlib import Path
from typing import Optional, Dict, Any
import PyPDF2


class DatasheetAgent:
    """Agent responsible for analyzing a single datasheet"""

    def __init__(self, datasheet_id: str, filename: str, filepath: Path, api_key: str):
        self.datasheet_id = datasheet_id
        self.filename = filename
        self.filepath = filepath
        self.client = anthropic.Anthropic(api_key=api_key)
        self.datasheet_content = None
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
            return True
        except Exception as e:
            print(f"Error initializing datasheet agent for {self.filename}: {e}")
            return False

    def _extract_pdf_text(self) -> str:
        """Extract text content from PDF"""
        try:
            with open(self.filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []

                # Extract text from each page (limit to first 50 pages to manage context)
                max_pages = min(50, len(pdf_reader.pages))
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    text_content.append(f"--- Page {page_num + 1} ---\n{page.extract_text()}")

                return "\n\n".join(text_content)
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""

    async def query(self, question: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Query this datasheet agent with a specific question

        Args:
            question: The question to ask about this datasheet
            max_tokens: Maximum tokens for the response

        Returns:
            Dict with 'response' and 'metadata'
        """
        try:
            if not self.datasheet_content:
                return {
                    'response': f"Datasheet {self.filename} has not been initialized yet.",
                    'metadata': {'error': 'not_initialized'}
                }

            # Create message with datasheet context
            message_content = f"""Datasheet Content:
{self.datasheet_content[:100000]}  # Limit context to ~100k chars

Question: {question}

Please answer based on the datasheet content above."""

            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Using Sonnet for cost efficiency
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": message_content}
                ]
            )

            response_text = message.content[0].text

            return {
                'response': response_text,
                'metadata': {
                    'datasheet_id': self.datasheet_id,
                    'filename': self.filename,
                    'model': message.model,
                    'usage': {
                        'input_tokens': message.usage.input_tokens,
                        'output_tokens': message.usage.output_tokens
                    }
                }
            }

        except Exception as e:
            print(f"Error querying datasheet agent: {e}")
            return {
                'response': f"Error querying datasheet: {str(e)}",
                'metadata': {'error': str(e)}
            }
