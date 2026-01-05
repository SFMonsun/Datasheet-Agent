# Multi-Agent Datasheet Analyzer - Setup Guide

## Overview

This application uses a multi-agent architecture:
- **Main Orchestrator Agent**: Helps you design PCBs and coordinates queries to datasheet agents
- **Datasheet Agents**: One agent per uploaded PDF, specialized in understanding that specific component

## How It Works

1. When you upload a PDF datasheet, a dedicated agent is created for that component
2. The main chat agent can query these datasheet agents when it needs specific information
3. You only pay for tokens when agents are actively queried - the orchestrator decides when to fetch information

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your Claude API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign in with your Claude Pro account
3. Navigate to API Keys
4. Create a new API key
5. Copy the key (starts with `sk-ant-...`)

### 3. Configure the Application

When you first run the application:
1. Click the **"Set API Key"** button (yellow key icon) in the chat header
2. Paste your API key
3. Click Save

Alternatively, set the environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Run the Application

```bash
python main.py
```

The app will open at http://localhost:8080

## Usage Guide

### Uploading Datasheets

1. Click the **+** button next to "Datasheets" in the sidebar
2. Select one or more PDF datasheets
3. Click **Upload** to confirm
4. The datasheets will appear in the sidebar

### Chatting with the Agent

Ask questions like:
- "What are the power requirements for all my components?"
- "Can I use a 3.3V supply for the ESP32?"
- "What pins do I need to connect for SPI communication?"
- "Help me design a power supply circuit for these components"

The orchestrator will automatically query the relevant datasheet agents when needed.

## Cost Estimation (Claude Pro)

With Claude Pro, you get generous API usage. For occasional testing:

- **Datasheet Agent**: ~5K-20K input tokens per query (depends on PDF size)
- **Main Chat**: ~1K-5K tokens per message
- **Response**: ~500-2K output tokens

Example session (uploading 3 datasheets + 10 questions):
- Datasheet loading: ~30K tokens (one-time)
- 10 conversations: ~50K-100K tokens
- **Total**: Well within Claude Pro limits for occasional use

## Tips

1. **PDF Quality**: Text-based PDFs work best. Scanned images may have poor text extraction
2. **File Size**: First 50 pages of each PDF are processed to stay within context limits
3. **Specific Questions**: More specific questions get better results from datasheet agents
4. **Cost Management**: The orchestrator only queries datasheet agents when necessary

## Troubleshooting

### "Please configure your Claude API key"
- Click the "Set API Key" button and enter your key from console.anthropic.com

### Datasheets not showing up
- Check the uploads/ directory was created
- Check console for error messages

### Agent not responding
- Verify your API key is valid
- Check your internet connection
- Look at the terminal output for error messages

### PDF text extraction failing
- Ensure the PDF is text-based, not a scanned image
- Try a different PDF viewer to verify the PDF has selectable text

## Architecture

```
User Question
     ↓
Main Orchestrator Agent
     ↓
[Decides which datasheet(s) to query]
     ↓
Datasheet Agent(s) ← PDF Content
     ↓
Synthesized Response
     ↓
User
```

The system uses Claude's tool use feature to intelligently decide when to fetch information from specific datasheets.
