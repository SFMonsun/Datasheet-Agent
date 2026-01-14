# Agentic AI Datasheet Agent

A sophisticated multi-agent system for analyzing electronic component datasheets using local LLMs (Ollama) or Claude API. The system employs a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, cited answers from datasheet PDFs.


# Problem Statement: Why an Agentic Datasheet Agent Is Sensible

## Background

Modern PCB projects routinely involve dozens to hundreds of components, each accompanied by extensive datasheets. These datasheets often span tens to hundreds of pages and include a mixture of critical specifications, edge-case constraints, reference designs, legal boilerplate, and marketing material.

When using large language models or other AI-assisted tools to reason about PCB designs, component selection, or design reviews, datasheets are typically ingested wholesale into the context window. This approach quickly becomes impractical.

## Core Problem

**Context windows are finite, but datasheets are not.**

Naively loading full datasheets into the context window leads to:

- **Context flooding**: Large volumes of irrelevant or low-value information (e.g., ordering tables, repeated electrical characteristics, packaging options) overwhelm the model.
- **Loss of global project awareness**: As datasheets consume the available context, higher-level project knowledge (overall PCB goals, constraints, prior decisions, trade-offs) is pushed out.
- **Reduced reasoning quality**: With a cluttered context window, the model’s ability to reason about interactions between components, system-level constraints, and design intent degrades.
- **Poor scalability**: As the PCB project grows, adding new components becomes increasingly expensive or impossible without sacrificing prior context.

The result is an AI assistant that technically “has the data,” but is no longer useful for understanding or managing the PCB as a coherent system.

## Why This Is a Structural Issue

This problem cannot be solved by:
- Larger context windows alone (they are still finite and costly)
- Manual datasheet pruning (time-consuming, error-prone, and non-repeatable)
- Static summaries (they fail to adapt to changing design questions)

What is needed is **selective, on-demand access to datasheet knowledge**, not continuous full exposure.

## The Role of an Agentic Datasheet Agent

An agentic datasheet agent addresses this problem by acting as an intelligent intermediary between raw datasheets and the project-level reasoning context.

Instead of flooding the context window, the agent:

- Parses and indexes datasheets once
- Extracts and retains *structured, queryable knowledge*
- Surfaces only **relevant fragments** (parameters, constraints, application notes) when needed
- Maintains links back to the source for traceability and verification

This allows the main reasoning agent to:

- Keep track of the **entire PCB project state**
- Retain **design intent, constraints, and decisions** in-context
- Pull in datasheet details **only when they are relevant to the current task**

## Key Insight

> The goal is not to fit all datasheets into the context window,  
> but to ensure the *right* datasheet information appears at the *right* time.

By decoupling datasheet storage from active reasoning context, an agentic datasheet agent enables scalable, coherent, and high-quality AI-assisted PCB design.

## Summary

An agentic datasheet agent is sensible because it:

- Prevents context window saturation
- Preserves system-level understanding of complex PCB projects
- Improves reasoning quality and relevance
- Scales naturally as project complexity grows

Without such an agent, AI-assisted PCB design inevitably collapses under its own informational weight.



![Dark Purple Theme](https://img.shields.io/badge/Theme-Dark%20Purple-8b7cc8)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

### Multi-Agent Architecture
- **Orchestrator Agent**: Coordinates queries and manages sub-agents
- **Datasheet Agents**: Specialized agents for each uploaded datasheet
- **Intelligent Routing**: Automatically identifies and queries only relevant datasheets
- **Real-time Visualization**: See agent-to-agent communication in action

### RAG Pipeline (Retrieval-Augmented Generation)
- **PDF Extraction**: Extracts text and tables from datasheets using pdfplumber
- **Smart Chunking**: Breaks content into semantic segments with overlap
- **Local Embeddings**: Uses sentence-transformers (no API needed)
- **Vector Database**: ChromaDB for fast similarity search
- **Source Citations**: Every answer includes page references

### User Interface
- **Dark Purple Theme**: Professional, eye-friendly color scheme
- **Session Management**: Save and load analysis sessions
- **Multi-file Upload**: Upload multiple datasheets at once
- **Live RAG Visualization**: Track the RAG pipeline in real-time
- **Agent Communication Window**: See how agents collaborate

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [RAG Pipeline Deep Dive](#rag-pipeline-deep-dive)
- [Agent Communication](#agent-communication)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running (for local LLM option)
- OR Claude API key (for cloud option)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "Datasheet Agent"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies Breakdown

#### Core Framework
- **nicegui>=2.10.0** - Modern web UI framework with reactive components
- **python-multipart>=0.0.9** - File upload handling

#### AI Providers
- **anthropic>=0.39.0** - Claude API client (optional, for cloud mode)
- **aiohttp>=3.9.0** - Async HTTP client for Ollama

#### Document Processing
- **PyPDF2>=3.0.0** - Legacy PDF support
- **pdfplumber>=0.11.0** - Advanced PDF extraction (tables, text, layout)

#### RAG Pipeline
- **sentence-transformers>=2.2.2** - Local embedding models (all-MiniLM-L6-v2)
- **chromadb>=0.4.22** - Vector database for similarity search
- **numpy>=1.24.0** - Numerical operations for embeddings

#### Utilities
- **requests>=2.31.0** - HTTP requests for legacy code

### Step 3: Install Ollama (for local LLM)

#### Windows
```bash
# Download from https://ollama.ai
# Run installer
```

#### Linux/Mac
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 4: Pull Your Preferred Model

```bash
ollama pull qwen3:8b
# or
ollama pull llama3.1:8b
# or
ollama pull mistral:7b
```

### Step 5: Start Ollama Server

```bash
ollama serve
```

## 🎯 Quick Start

### 1. Start the Application

```bash
python app.py
```

The web interface will open automatically at `http://localhost:8080`

### 2. Upload Datasheets

1. Click the **+** button in the sidebar
2. Select one or more PDF datasheets
3. Click **Upload**
4. Wait for agents to initialize (RAG processing happens here)

### 3. Ask Questions

Type questions like:
- "What is the operating voltage range for the LM75B?"
- "What are the I2C addresses for the INA219?"
- "Compare the current consumption of all loaded components"

### 4. View Agent Interactions

Watch the **Agent Communication** panel to see:
- 📋 Agent registrations (purple highlight)
- ➡️ Queries from orchestrator to sub-agents
- ⬅️ Responses back to orchestrator

### 5. Monitor RAG Pipeline

The **RAG Pipeline** panel shows:
- 📄 PDF extraction progress
- ✂️ Text chunking statistics
- 🧠 Embedding generation
- 🔍 Relevant chunk retrieval with similarity scores

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface (NiceGUI)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Sidebar    │  │  Chat Area   │  │  Visualization Panels │  │
│  │  (Upload)    │  │  (Q&A)       │  │  (Agent + RAG)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator Agent                           │
│  • Receives user questions                                       │
│  • Determines relevant datasheets (component name matching)      │
│  • Queries appropriate sub-agents                                │
│  • Synthesizes final answer                                      │
└─────────────────────────────────────────────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │         Sub-Agent Selection Logic            │
        │  • Regex word boundary matching              │
        │  • Component name matching                   │
        │  • Filename matching                         │
        └─────────────────────────────────────────────┘
                              ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Datasheet Agent  │  │ Datasheet Agent  │  │ Datasheet Agent  │
│   (LM75B.pdf)    │  │  (INA219.pdf)    │  │  (Other.pdf)     │
│                  │  │                  │  │                  │
│  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │
│  │ RAG Service│  │  │  │ RAG Service│  │  │  │ RAG Service│  │
│  │            │  │  │  │            │  │  │  │            │  │
│  │ • Extract  │  │  │  │ • Extract  │  │  │  │ • Extract  │  │
│  │ • Chunk    │  │  │  │ • Chunk    │  │  │  │ • Chunk    │  │
│  │ • Embed    │  │  │  │ • Embed    │  │  │  │ • Embed    │  │
│  │ • Retrieve │  │  │  │ • Retrieve │  │  │  │ • Retrieve │  │
│  └────────────┘  │  │  └────────────┘  │  │  └────────────┘  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   LLM Backend    │
                    │                  │
                    │  • Ollama (local)│
                    │  • Claude (API)  │
                    └──────────────────┘
```

### Component Responsibilities

#### 1. **Orchestrator Agent** (`services/ollama_orchestrator_agent.py`)
- Maintains a registry of all datasheet agents
- Parses user queries to identify relevant components
- Routes queries to appropriate sub-agents
- Aggregates responses and provides final answer
- Tracks agent interactions for visualization

#### 2. **Datasheet Agent** (`services/ollama_datasheet_agent_rag.py`)
- One agent per uploaded datasheet
- Initializes RAG pipeline on creation
- Identifies component type during initialization
- Answers specific questions about its datasheet
- Provides page-level source citations

#### 3. **RAG Service** (`services/rag_service.py`)
- Extracts text and tables from PDFs
- Chunks content into semantic segments
- Generates embeddings using sentence-transformers
- Stores vectors in ChromaDB
- Retrieves most relevant chunks for queries

## 🔬 RAG Pipeline Deep Dive

### What is RAG?

**Retrieval-Augmented Generation** combines information retrieval with language model generation. Instead of relying solely on the LLM's training data, RAG:

1. **Retrieves** relevant information from your documents
2. **Augments** the LLM prompt with this context
3. **Generates** accurate answers grounded in your data

### Pipeline Stages

#### Stage 1: PDF Extraction 📄

**What Happens:**
```python
with pdfplumber.open(filepath) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        tables = page.extract_tables()
```

**Visualized As:**
```
📄 Extracting content from PDF
   (12 pages)
```

**Details:**
- Uses `pdfplumber` for high-quality extraction
- Extracts both text and tabular data
- Preserves page numbers for citation
- Handles complex layouts better than PyPDF2

**Why This Matters:**
Datasheets often contain critical information in tables (pin configurations, electrical characteristics). pdfplumber preserves this structure.

---

#### Stage 2: Text Chunking ✂️

**What Happens:**
```python
chunk_content(content, chunk_size=1000, overlap=200)
```

**Algorithm:**
1. Split text by double newlines (paragraphs)
2. Build chunks up to 1000 characters
3. Add 200 character overlap between chunks
4. Convert tables to readable text format

**Visualized As:**
```
✂️ Chunking text into 1000 char segments
   (47 chunks)
```

**Example Chunk:**
```
[Chunk 23, Page 5]
Operating Conditions
The LM75B operates from 2.8V to 5.5V supply voltage.
Temperature measurement range: -55°C to +125°C
Accuracy: ±2°C from -25°C to +100°C
Accuracy: ±3°C from -55°C to +125°C
[200 char overlap with next chunk...]
```

**Why Overlap?**
Context isn't lost at chunk boundaries. If a sentence is split, it appears in both chunks.

---

#### Stage 3: Embedding Generation 🧠

**What Happens:**
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)  # → 384-dimensional vectors
```

**Visualized As:**
```
🧠 Creating vector embeddings for 47 chunks
   (47 embeddings)
```

**Technical Details:**
- **Model**: all-MiniLM-L6-v2 (local, 80MB download)
- **Output**: 384-dimensional vector per chunk
- **Speed**: ~100 chunks/second on CPU
- **Quality**: Optimized for semantic similarity

**Vector Representation:**
```
"Operating voltage range 2.8V to 5.5V"
    ↓
[0.042, -0.183, 0.291, ..., 0.167]  # 384 numbers
```

Similar sentences → Similar vectors in 384D space

---

#### Stage 4: Vector Storage 💾

**What Happens:**
```python
collection.add(
    ids=chunk_ids,
    embeddings=embeddings,
    documents=texts,
    metadatas=page_numbers
)
```

**Storage Format (ChromaDB):**
```
datasheet_lm75b/
  ├── chunk_0: [vector], "Page 1", "LM75B Digital..."
  ├── chunk_1: [vector], "Page 1", "Features: ±2°C..."
  ├── chunk_2: [vector], "Page 2", "Pin Configuration..."
  └── ...
```

**Why ChromaDB?**
- Fast similarity search (HNSW algorithm)
- Persistent storage (survives restarts)
- Metadata filtering (search by page)
- Local (no external database needed)

---

#### Stage 5: Query Processing 🔍

**What Happens When You Ask a Question:**

```python
# 1. Embed the question
query_embedding = model.encode("What is the I2C address?")

# 2. Find similar chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5  # Top 5 most relevant
)

# 3. Rank by similarity
# Cosine similarity: 1.0 = identical, 0.0 = unrelated
```

**Visualized As:**
```
🔍 Finding top 5 relevant chunks
   (5 found)

📄 Page 8 (Relevance: 94%, text)
   Preview: "I2C Interface The LM75B uses a 2-wire I2C-bus..."

📄 Page 9 (Relevance: 87%, table)
   Preview: "Address Pin Configuration | A0 | A1 | A2 | Addr..."

📄 Page 3 (Relevance: 82%, text)
   Preview: "Communication Protocol The device supports..."
```

**Similarity Calculation:**
```
Query: "What is the I2C address?"
  Vector: [0.123, -0.456, 0.789, ...]

Chunk 42 (Page 8): "I2C Interface..."
  Vector: [0.119, -0.442, 0.801, ...]

Cosine Similarity = 0.94 → 94% relevance
```

---

#### Stage 6: Context Building 📝

**What Happens:**
```python
prompt = f"""
Datasheet excerpts:

[Source 1: Page 8]
The LM75B uses a 2-wire I2C-bus interface. The 7-bit
base address is 1001xxx where xxx are determined by
pins A0, A1, A2.

[Source 2: Page 9]
TABLE: Address Configuration
A2 | A1 | A0 | Address
0  | 0  | 0  | 0x48
0  | 0  | 1  | 0x49
...

Question: What is the I2C address?

IMPORTANT: Cite pages using (Page X) format.

Your answer:
"""
```

**LLM Response:**
```
The LM75B uses I2C address 0x48 as the default when all
address pins (A0, A1, A2) are grounded (Page 9). The 7-bit
base address is 1001xxx, where the last three bits are
configurable via the address pins (Page 8), allowing 8
possible addresses from 0x48 to 0x4F (Page 9).

---
**Sources Referenced:** Pages 8, 9 in LM75B.pdf
```

---

### RAG vs. Non-RAG Comparison

#### Without RAG:
```
User: "What is the I2C address of LM75B?"
LLM: "The LM75B typically uses I2C address 0x48, but this
      may vary. Check your specific datasheet."

❌ Vague, no citations
❌ May be wrong for your specific chip
❌ No page reference for verification
```

#### With RAG:
```
User: "What is the I2C address of LM75B?"

RAG Pipeline:
1. Extract question embedding
2. Find 5 most relevant chunks from LM75B.pdf
3. Build context with page numbers
4. Generate answer with citations

LLM: "The default I2C address is 0x48 (Page 9). The address
      is configurable using pins A0, A1, A2, allowing 8
      addresses from 0x48 to 0x4F (Page 9, Table 8-1)."

✅ Specific to your datasheet
✅ Page-level citations
✅ Verifiable information
```

---

### RAG Visualization Panel Explained

When you ask a question, the **RAG Pipeline** panel shows each stage in real-time:

```
┌─────────────────────────────────────────────────────────────┐
│ RAG Pipeline                                                 │
├─────────────────────────────────────────────────────────────┤
│ 📄 Extracting content from PDF                 (12 pages)   │
│ ✂️ Chunking text into 1000 char segments       (47 chunks)  │
│ 🧠 Creating vector embeddings for 47 chunks    (47 emb.)    │
│ 🔍 Finding top 5 relevant chunks                (5 found)   │
│   📄 Page 8 (Relevance: 94%, text)                          │
│      I2C Interface The LM75B uses a 2-wire I2C-bus...       │
│   📄 Page 9 (Relevance: 87%, table)                         │
│      Address Pin Configuration | A0 | A1 | A2 | Addr...    │
│   📄 Page 3 (Relevance: 82%, text)                          │
│      Communication Protocol The device supports...          │
└─────────────────────────────────────────────────────────────┘
```

**Color Coding:**
- 🟡 **Orange** (warning) - In progress
- 🟢 **Green** (success) - Completed successfully
- 🔴 **Red** (error) - Failed (with error message)

---

### Performance Characteristics

| Stage | Time (3MB PDF) | Scalability |
|-------|----------------|-------------|
| PDF Extraction | 2-5 seconds | O(n pages) |
| Chunking | <1 second | O(n chars) |
| Embedding | 5-15 seconds | O(n chunks) |
| Storage | <1 second | O(n chunks) |
| Query | 0.1-0.5 seconds | O(log n) |

**First-time Upload:** 10-25 seconds (one-time cost)
**Subsequent Queries:** <1 second (vectors cached in ChromaDB)

---

## 🤝 Agent Communication

### How Agent Communication Works

#### 1. Registration Phase

When you upload a datasheet, each sub-agent:

1. Processes the PDF with RAG pipeline
2. Analyzes first 3 chunks to identify component
3. Registers with orchestrator

**Visualization:**
```
┌─────────────────────────────────────────────────────────────┐
│ Agent Communication                                          │
├─────────────────────────────────────────────────────────────┤
│ 📋 LM75B.pdf registered                                      │
│    LM75B (Temperature Sensor)                                │
│                                                              │
│ 📋 INA219.pdf registered                                     │
│    INA219 (Current/Power Monitor)                            │
└─────────────────────────────────────────────────────────────┘
```

**Purple Highlight:** Registration events have a distinct purple border and icon

---

#### 2. Query Routing

When you ask a question, the orchestrator:

```python
# 1. Parse question for component names
user_message = "What is the operating voltage of LM75B?"

# 2. Match against registered agents
for agent in agents:
    if "lm75b" in user_message.lower():
        relevant_agents.append(agent)  # ✅ Matches LM75B agent

# 3. Query only relevant agents (not INA219!)
```

**Smart Matching:**
- Word boundary regex: `\bLM75B\b` matches "LM75B" but not "LM75BC"
- Case insensitive
- Matches component names AND filenames

---

#### 3. Query Execution

**Visualization:**
```
┌─────────────────────────────────────────────────────────────┐
│ Agent Communication                                          │
├─────────────────────────────────────────────────────────────┤
│ ➡️ Orchestrator → LM75B.pdf                                  │
│    "What is the operating voltage of LM75B?"                 │
│                                                              │
│ ⬅️ LM75B.pdf → Orchestrator                                  │
│    Response received                                         │
└─────────────────────────────────────────────────────────────┘
```

**Flow:**
1. **Blue Arrow (→)**: Orchestrator sends query to sub-agent
2. Sub-agent runs RAG pipeline (shown in RAG panel)
3. **Green Arrow (←)**: Sub-agent returns answer with citations

---

#### 4. Multi-Agent Queries

If you ask: "Compare the LM75B and INA219 current consumption"

**Visualization:**
```
┌─────────────────────────────────────────────────────────────┐
│ Agent Communication                                          │
├─────────────────────────────────────────────────────────────┤
│ ➡️ Orchestrator → LM75B.pdf                                  │
│    "Compare the LM75B and INA219 current con..."             │
│                                                              │
│ ⬅️ LM75B.pdf → Orchestrator                                  │
│    Response received                                         │
│                                                              │
│ ➡️ Orchestrator → INA219.pdf                                 │
│    "Compare the LM75B and INA219 current con..."             │
│                                                              │
│ ⬅️ INA219.pdf → Orchestrator                                 │
│    Response received                                         │
└─────────────────────────────────────────────────────────────┘
```

The orchestrator then synthesizes both responses into a comparison.

---

### Agent Communication Icons

| Icon | Meaning | Color |
|------|---------|-------|
| 📋 `app_registration` | Agent registered | Purple |
| ➡️ `arrow_forward` | Query sent | Blue |
| ⬅️ `arrow_back` | Response received | Green |

---

## 🛡️ Guardrail System

The orchestrator includes an intelligent guardrail system that prevents hallucination by refusing to answer questions about components not in the database.

### How It Works

#### 1. Component Detection

When you ask a question, the system scans for electronic component part numbers using regex patterns:

```python
# Patterns detected:
r'\b[A-Z]{2,4}\d{2,5}[A-Z]?\b'   # LM75B, INA219, STM32F4
r'\b[A-Z]{2,3}-?\d{3,5}\b'        # AT-24C02, NE-555
r'\b\d{2,3}[A-Z]{2,4}\d{2,4}\b'   # 74HC595
```

**False Positive Filtering:**

The system ignores common protocol/interface names that match component patterns:
- I2C, SPI, USB, GPIO, ADC, DAC, PWM, UART, LED, LCD, PCB

#### 2. Database Matching

After detecting component names, the system checks if any match the loaded datasheets:

```
User asks about: "STM32F4"
                    ↓
Loaded datasheets: [LM75B, INA219]
                    ↓
Match found? NO → Guardrail triggered
```

#### 3. Guardrail Response

**Scenario A: No datasheets loaded**
```
I don't have any datasheets loaded in my database.

You asked about: **STM32F4**

Please upload the relevant datasheet(s) using the **+** button
in the sidebar, and I'll be happy to help you with detailed
information.
```

**Scenario B: Some datasheets loaded, but not the requested one**
```
I don't have **STM32F4** in my datasheet database.

**Available components:**
- LM75B
- INA219

Please upload the datasheet for **STM32F4** using the **+**
button in the sidebar, or ask me about one of the components
I have loaded.
```

### Why Guardrails Matter

| Without Guardrails | With Guardrails |
|-------------------|-----------------|
| LLM might hallucinate specifications | Refuses to answer without data |
| No indication data is unreliable | Clear message about missing datasheet |
| User might trust incorrect info | User knows to upload the datasheet |
| Could lead to design errors | Prevents misinformation |

### Guardrail Behavior Examples

| Question | Datasheets Loaded | Guardrail? | Behavior |
|----------|------------------|------------|----------|
| "What is the I2C address of LM75B?" | LM75B.pdf | No | Queries LM75B agent |
| "What is the I2C address of STM32F4?" | LM75B.pdf | **Yes** | Refuses, lists available |
| "What is I2C?" | LM75B.pdf | No | General knowledge answer |
| "Compare LM75B and INA219" | Both loaded | No | Queries both agents |
| "Tell me about the ATmega328" | LM75B.pdf | **Yes** | Refuses, suggests upload |

### Console Output

When a guardrail is triggered, you'll see in the console:

```
[Orchestrator] Found 0 relevant datasheets
[Orchestrator] GUARDRAIL: Blocked query about unknown component(s): ['STM32F4']
```

### Customizing Detection Patterns

To add custom component patterns, edit `_detect_component_query()` in `ollama_orchestrator_agent.py`:

```python
component_patterns = [
    r'\b[A-Z]{2,4}\d{2,5}[A-Z]?\b',  # Standard ICs
    r'\b[A-Z]{2,3}-?\d{3,5}\b',       # Hyphenated parts
    r'\b\d{2,3}[A-Z]{2,4}\d{2,4}\b',  # 74-series logic
    # Add your custom patterns here:
    r'\bYOUR_PATTERN\b',
]
```

To add false positives to ignore:

```python
false_positives = {
    'I2C', 'SPI', 'USB', 'GPIO', 'ADC', 'DAC',
    'PWM', 'UART', 'LED', 'LCD', 'PCB',
    # Add more here:
    'YOUR_TERM',
}
```

---

## ⚙️ Configuration

### `config.py` Options

#### UI Colors (Dark Purple Theme)
```python
COLORS = {
    'primary': '#8b7cc8',        # Medium purple (buttons, highlights)
    'secondary': '#6b5b95',      # Dark purple (registration events)
    'success': '#7c9885',        # Muted green-grey (success states)
    'warning': '#c8a87c',        # Muted orange-grey (in-progress)
    'error': '#c87c85',          # Muted red-grey (errors)
    'background': '#1a1625',     # Very dark purple-black
    'surface': '#2a2438',        # Dark purple-grey (cards)
    'surface_light': '#3d3550',  # Medium purple-grey (nested cards)
    'text': '#f5f3f7',           # Off-white (primary text)
    'text_secondary': '#a89fb8', # Light purple-grey (labels)
    'border': '#4a4158'          # Medium-dark purple-grey
}
```

#### AI Provider Settings
```python
AGENT_CONFIG = {
    'provider': 'ollama',           # 'ollama' or 'claude'
    'ollama_model': 'qwen3:8b',     # Your Ollama model
    'ollama_url': 'http://localhost:11434',
    'use_rag': True,                # Enable RAG pipeline
    'model': 'claude-3-5-sonnet-20241022',  # For Claude mode
}
```

#### RAG Settings (in `services/rag_service.py`)
```python
# Chunking
chunk_size = 1000      # Characters per chunk
overlap = 200          # Overlap between chunks

# Retrieval
top_k = 5              # Number of chunks to retrieve

# Embedding model
embedding_model = "all-MiniLM-L6-v2"  # Sentence transformer
```

---

## 💡 Usage Examples

### Example 1: Single Component Query

**Question:**
```
What is the I2C address of the LM75B?
```

**Agent Communication:**
```
➡️ Orchestrator → LM75B.pdf
   "What is the I2C address of the LM75B?"
⬅️ LM75B.pdf → Orchestrator
   Response received
```

**RAG Pipeline:**
```
🔍 Finding top 5 relevant chunks (5 found)
   📄 Page 8 (Relevance: 94%, text)
   📄 Page 9 (Relevance: 87%, table)
```

**Answer:**
```
The LM75B default I2C address is 0x48 when all address pins
(A0, A1, A2) are tied to ground (Page 9). The 7-bit base
address is 1001xxx where the last 3 bits are set by the
address pins, allowing 8 possible addresses from 0x48 to
0x4F (Page 8, Page 9).

---
Sources Referenced: Pages 8, 9 in LM75B.pdf
```

---

### Example 2: Multi-Component Comparison

**Question:**
```
Compare the supply current of LM75B and INA219
```

**Agent Communication:**
```
➡️ Orchestrator → LM75B.pdf
   "Compare the supply current of LM75B and INA219"
⬅️ LM75B.pdf → Orchestrator
   Response received

➡️ Orchestrator → INA219.pdf
   "Compare the supply current of LM75B and INA219"
⬅️ INA219.pdf → Orchestrator
   Response received
```

**Answer:**
```
From LM75B.pdf:
The LM75B supply current is typically 200μA in active mode
(Page 4) and 1μA in shutdown mode (Page 5).

From INA219.pdf:
The INA219 supply current is typically 1mA in continuous
mode (Page 7).

Comparison:
The LM75B consumes significantly less power (200μA vs 1000μA),
making it more suitable for battery-powered applications.

---
Sources Referenced: Pages 4, 5 in LM75B.pdf; Page 7 in INA219.pdf
```

---

### Example 3: General Question (No Agent Querying)

**Question:**
```
What is I2C protocol?
```

**Agent Communication:**
```
(No agent interactions - general knowledge question)
```

**Behavior:**
No datasheet agents are queried. The orchestrator answers directly using the LLM's general knowledge.

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to Ollama"

**Cause:** Ollama server not running

**Solution:**
```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

---

### Issue: RAG Pipeline Shows "0 chunks"

**Cause:** PDF extraction failed

**Debug:**
```python
# Check if pdfplumber can read your PDF
import pdfplumber
with pdfplumber.open("your_datasheet.pdf") as pdf:
    print(f"Pages: {len(pdf.pages)}")
    print(f"First page text: {pdf.pages[0].extract_text()[:100]}")
```

**Solutions:**
- Ensure PDF is not encrypted
- Check if PDF contains actual text (not scanned images)
- Try re-saving PDF from another viewer

---

### Issue: Wrong Datasheet Queried

**Cause:** Component name ambiguity

**Example:**
Asking about "current" matches "INA219 (Current Sensor)" incorrectly

**Solution:**
Be specific in your questions:
- ✅ "What is the LM75B current consumption?"
- ❌ "What is the current consumption?" (too vague)

---

### Issue: Slow Response Times

**Causes:**
1. First query after upload (RAG processing)
2. Large PDF (>50 pages)
3. CPU-bound embedding generation

**Solutions:**
- **First query:** Wait for RAG pipeline (one-time cost)
- **Large PDFs:** Consider reducing `chunk_size` or using GPU
- **Speed up embeddings:** Use smaller model or GPU acceleration

**GPU Acceleration (optional):**
```python
# In rag_service.py
device = 'cuda' if torch.cuda.is_available() else 'cpu'
self.embedding_model = SentenceTransformer(model_name, device=device)
```

---

### Issue: "List.remove(x): x not in list" Error

**Cause:** UI element deletion race condition (fixed in latest version)

**Solution:** Update to latest version with error handling:
```python
try:
    typing_msg.delete()
except (ValueError, RuntimeError):
    pass
```

---

### Issue: Citations Missing or Incorrect

**Cause:** LLM not following citation instructions

**Solutions:**
1. **Increase temperature:** More creative citations
   ```python
   "temperature": 0.3  # In ollama_datasheet_agent_rag.py
   ```

2. **Use better model:** Larger models follow instructions better
   ```bash
   ollama pull qwen3:14b  # Instead of 8b
   ```

3. **Verify chunks:** Check if relevant chunks contain page numbers
   ```python
   # In RAG visualization, verify "Page X" is shown
   ```

---

## 📊 Performance Tips

### Optimizing Upload Speed

1. **Batch uploads:** Upload multiple datasheets at once
2. **Smaller chunks:** Reduce `chunk_size` for faster processing
3. **Skip tables:** If tables aren't needed, disable table extraction

### Optimizing Query Speed

1. **Reduce `top_k`:** Retrieve fewer chunks (3 instead of 5)
2. **Cache embeddings:** ChromaDB persists between sessions
3. **Use smaller model:** `qwen2.5:3b` is faster than `qwen3:8b`

### Memory Usage

| Component | Memory Usage |
|-----------|--------------|
| Embedding Model | ~80 MB (all-MiniLM-L6-v2) |
| ChromaDB | ~50 MB per 1000 chunks |
| Ollama Model | 5-8 GB (qwen3:8b) |
| NiceGUI | ~100 MB |

**Total:** ~6-9 GB for typical setup

---

## 🔒 Security Notes

- API keys stored in `data/.api_key` (gitignored)
- Uploaded files stored in `data/uploads/` (local only)
- No data sent to external services (Ollama mode)
- Session data in `data/sessions/` (JSON files)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Support for scanned PDFs (OCR)
- [ ] Export chat history to PDF
- [ ] Multi-language support
- [ ] Custom embedding models
- [ ] Graph visualization of agent interactions
- [ ] Conversation memory across sessions

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **NiceGUI** - Modern Python web framework
- **Ollama** - Local LLM runtime
- **Anthropic** - Claude API
- **ChromaDB** - Vector database
- **Sentence Transformers** - Embedding models
- **pdfplumber** - PDF extraction

---

## 📧 Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Documentation: This README
- Ollama Docs: https://ollama.ai/docs
- ChromaDB Docs: https://docs.trychroma.com/

---

**Built with ❤️ for PCB designers and electronics engineers**
