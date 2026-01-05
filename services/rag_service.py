"""
RAG (Retrieval-Augmented Generation) Service for Datasheet Processing
"""
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import re


class RAGService:
    """Handles document chunking, embedding, and retrieval for datasheets"""

    def __init__(self, datasheet_id: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG service for a specific datasheet

        Args:
            datasheet_id: Unique identifier for this datasheet
            embedding_model: Sentence transformer model name (local, no API needed)
        """
        self.datasheet_id = datasheet_id
        self.embedding_model_name = embedding_model
        self.embedding_model = None  # Lazy load
        self.chunks: List[Dict[str, Any]] = []
        self.rag_steps: List[Dict[str, Any]] = []  # Track RAG operations for visualization

        # Initialize ChromaDB (local, persistent storage)
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # Create or get collection for this datasheet
        try:
            self.collection = self.chroma_client.get_collection(name=f"datasheet_{datasheet_id}")
            print(f"[RAG] Loaded existing collection for {datasheet_id}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=f"datasheet_{datasheet_id}",
                metadata={"description": f"Datasheet embeddings for {datasheet_id}"}
            )
            print(f"[RAG] Created new collection for {datasheet_id}")

    def _load_embedding_model(self):
        """Lazy load the embedding model (downloads on first use)"""
        if self.embedding_model is None:
            print(f"[RAG] Loading embedding model: {self.embedding_model_name}")
            self.rag_steps.append({
                'step': 'model_loading',
                'description': f'Loading embedding model: {self.embedding_model_name}',
                'status': 'in_progress'
            })
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.rag_steps[-1]['status'] = 'completed'
            print(f"[RAG] Embedding model loaded")

    def extract_pdf_content(self, filepath: Path) -> Dict[str, Any]:
        """
        Extract structured content from PDF using pdfplumber

        Returns:
            Dict with text, tables, and metadata
        """
        print(f"[RAG] Extracting PDF content from {filepath}")
        self.rag_steps.append({
            'step': 'pdf_extraction',
            'description': f'Extracting content from PDF',
            'status': 'in_progress'
        })

        content = {
            'text': [],
            'tables': [],
            'metadata': {'total_pages': 0}
        }

        try:
            with pdfplumber.open(filepath) as pdf:
                content['metadata']['total_pages'] = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        content['text'].append({
                            'page': page_num + 1,
                            'content': text
                        })

                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            content['tables'].append({
                                'page': page_num + 1,
                                'table_index': table_idx,
                                'data': table
                            })

            self.rag_steps[-1]['status'] = 'completed'
            self.rag_steps[-1]['pages_extracted'] = content['metadata']['total_pages']
            self.rag_steps[-1]['tables_found'] = len(content['tables'])

            print(f"[RAG] Extracted {content['metadata']['total_pages']} pages, {len(content['tables'])} tables")

        except Exception as e:
            print(f"[RAG] Error extracting PDF: {e}")
            self.rag_steps[-1]['status'] = 'error'
            self.rag_steps[-1]['error'] = str(e)

        return content

    def chunk_content(self, content: Dict[str, Any], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Chunk the extracted content into meaningful segments

        Args:
            content: Extracted PDF content
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of chunks with metadata
        """
        print(f"[RAG] Chunking content (size={chunk_size}, overlap={overlap})")
        self.rag_steps.append({
            'step': 'chunking',
            'description': f'Chunking text into {chunk_size} char segments',
            'status': 'in_progress'
        })

        chunks = []
        chunk_id = 0

        # Chunk text content
        for page_data in content['text']:
            page_num = page_data['page']
            text = page_data['content']

            # Split by paragraphs first (double newline)
            paragraphs = re.split(r'\n\s*\n', text)

            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # If adding this paragraph exceeds chunk size, save current chunk
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunks.append({
                        'id': f"chunk_{chunk_id}",
                        'content': current_chunk.strip(),
                        'type': 'text',
                        'page': page_num,
                        'metadata': {'chunk_index': chunk_id}
                    })
                    chunk_id += 1

                    # Keep overlap from previous chunk
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk += " " + para

            # Save any remaining content
            if current_chunk.strip():
                chunks.append({
                    'id': f"chunk_{chunk_id}",
                    'content': current_chunk.strip(),
                    'type': 'text',
                    'page': page_num,
                    'metadata': {'chunk_index': chunk_id}
                })
                chunk_id += 1

        # Add tables as separate chunks
        for table_data in content['tables']:
            table_text = self._table_to_text(table_data['data'])
            chunks.append({
                'id': f"chunk_{chunk_id}",
                'content': table_text,
                'type': 'table',
                'page': table_data['page'],
                'metadata': {
                    'chunk_index': chunk_id,
                    'table_index': table_data['table_index']
                }
            })
            chunk_id += 1

        self.chunks = chunks
        self.rag_steps[-1]['status'] = 'completed'
        self.rag_steps[-1]['chunks_created'] = len(chunks)

        print(f"[RAG] Created {len(chunks)} chunks")
        return chunks

    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table to readable text format"""
        if not table:
            return ""

        lines = []
        for row in table:
            if row:
                # Filter out None values and join
                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                lines.append(row_text)

        return "TABLE:\n" + "\n".join(lines)

    def create_embeddings(self):
        """Create vector embeddings for all chunks"""
        if not self.chunks:
            print("[RAG] No chunks to embed")
            return

        self._load_embedding_model()

        print(f"[RAG] Creating embeddings for {len(self.chunks)} chunks")
        self.rag_steps.append({
            'step': 'embedding',
            'description': f'Creating vector embeddings for {len(self.chunks)} chunks',
            'status': 'in_progress'
        })

        try:
            # Extract text content for embedding
            texts = [chunk['content'] for chunk in self.chunks]
            ids = [chunk['id'] for chunk in self.chunks]
            metadatas = [
                {
                    'page': chunk['page'],
                    'type': chunk['type'],
                    **chunk['metadata']
                }
                for chunk in self.chunks
            ]

            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )

            self.rag_steps[-1]['status'] = 'completed'
            self.rag_steps[-1]['embeddings_created'] = len(embeddings)

            print(f"[RAG] Stored {len(embeddings)} embeddings in vector database")

        except Exception as e:
            print(f"[RAG] Error creating embeddings: {e}")
            self.rag_steps[-1]['status'] = 'error'
            self.rag_steps[-1]['error'] = str(e)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query

        Args:
            query: The question/query
            top_k: Number of top chunks to retrieve

        Returns:
            List of relevant chunks with similarity scores
        """
        self._load_embedding_model()

        print(f"[RAG] Retrieving top {top_k} chunks for query: {query[:50]}...")
        self.rag_steps.append({
            'step': 'retrieval',
            'description': f'Finding top {top_k} relevant chunks',
            'query': query[:100],
            'status': 'in_progress'
        })

        try:
            # Encode the query
            query_embedding = self.embedding_model.encode([query])[0]

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Format results
            relevant_chunks = []
            if results and results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    relevant_chunks.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })

            self.rag_steps[-1]['status'] = 'completed'
            self.rag_steps[-1]['chunks_retrieved'] = len(relevant_chunks)
            self.rag_steps[-1]['chunks'] = [
                {
                    'rank': c['rank'],
                    'page': c['metadata'].get('page', 'N/A'),
                    'type': c['metadata'].get('type', 'text'),
                    'similarity': round(c['similarity_score'], 3),
                    'preview': c['content'][:100] + '...'
                }
                for c in relevant_chunks
            ]

            print(f"[RAG] Retrieved {len(relevant_chunks)} relevant chunks")
            return relevant_chunks

        except Exception as e:
            print(f"[RAG] Error retrieving chunks: {e}")
            self.rag_steps[-1]['status'] = 'error'
            self.rag_steps[-1]['error'] = str(e)
            return []

    def get_rag_steps(self) -> List[Dict[str, Any]]:
        """Get all RAG operation steps for visualization"""
        return self.rag_steps

    def clear_rag_steps(self):
        """Clear the RAG steps log"""
        self.rag_steps = []

    def reset_collection(self):
        """Delete the vector database for this datasheet"""
        try:
            self.chroma_client.delete_collection(name=f"datasheet_{self.datasheet_id}")
            print(f"[RAG] Deleted collection for {self.datasheet_id}")
        except Exception as e:
            print(f"[RAG] Error deleting collection: {e}")
