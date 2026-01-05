# Ollama Local Model Setup

## Overview

You can now use the Datasheet Agent completely FREE with Ollama! This eliminates all API costs by running AI models locally on your machine.

## Features

✅ **Zero API Costs** - Everything runs on your local machine
✅ **Privacy** - Your datasheets never leave your computer
✅ **Agent Visualization** - See the orchestrator communicating with datasheet agents in real-time
✅ **Model Selection** - Easy dropdown to switch between Claude API and Ollama

## Setup Instructions

### 1. Ensure Ollama is Installed

You mentioned you already have Ollama installed with qwen2.5:3b. Verify it's working:

```bash
ollama list
```

You should see `qwen2.5:3b` in the list.

### 2. Start Ollama Server

Make sure Ollama is running:

```bash
ollama serve
```

Keep this terminal open while using the app.

### 3. Configure the Model

The app is pre-configured to use `qwen2.5:3b`. If you want to use a different model:

1. Edit `config.py`
2. Change `'ollama_model': 'qwen2.5:3b'` to your preferred model
3. Restart the app

### 4. Using the App

1. Start the application:
   ```bash
   python main.py
   ```

2. In the chat header, you'll see a **Model** dropdown
3. Select **"Ollama (Local)"**
4. The page will reload with Ollama configured

5. Upload your datasheets and start chatting!

## Agent Communication Visualization

When using Ollama, you'll see an **"Agent Communication"** panel above the chat. This shows:

- **Blue arrows (→)**: Orchestrator querying a datasheet agent
- **Green arrows (←)**: Datasheet agent responding
- **Question previews**: What the orchestrator is asking each agent

This helps you understand how the multi-agent system works behind the scenes!

## Model Recommendations

### For Best Results:
- **qwen2.5:7b** - Good balance of speed and quality
- **llama3.1:8b** - Excellent reasoning capabilities
- **mistral:7b** - Fast and capable

### For Faster Performance:
- **qwen2.5:3b** - Very fast, decent quality (your current model)
- **phi3:mini** - Extremely fast, good for simple queries

### For Maximum Quality (if you have GPU):
- **qwen2.5:14b** - Best quality, slower
- **llama3.1:70b** - Excellent but requires significant RAM/VRAM

## Install Additional Models

```bash
# Install a specific model
ollama pull qwen2.5:7b

# List available models
ollama list

# Remove a model
ollama rm model_name
```

## Performance Tips

1. **First Query is Slow**: The first query loads the model into memory. Subsequent queries are much faster.

2. **Keep Ollama Running**: Don't stop `ollama serve` between sessions to avoid reload delays.

3. **PDF Size**: Local models have smaller context windows. The app automatically limits PDFs to the first 20 pages for Ollama.

4. **RAM Usage**:
   - 3B models: ~4GB RAM
   - 7B models: ~8GB RAM
   - 14B+ models: 16GB+ RAM recommended

## Troubleshooting

### "Cannot connect to Ollama"
- Ensure `ollama serve` is running
- Check Ollama is on `http://localhost:11434`
- Try: `curl http://localhost:11434/api/tags`

### Model Not Found
```bash
ollama pull qwen2.5:3b
```

### Slow Responses
- Use a smaller model (3B or 7B)
- Close other applications
- Check if you have GPU acceleration enabled

### Agent Not Responding
- Check terminal running `ollama serve` for errors
- Restart Ollama: Stop and run `ollama serve` again
- Verify model is downloaded: `ollama list`

## Switching Back to Claude

Simply use the dropdown in the chat header:
1. Select **"Claude (API)"**
2. Page reloads
3. Configure your API key if needed

## Architecture with Ollama

```
User Question
     ↓
Ollama Orchestrator Agent (qwen2.5:3b)
     ↓
[Analyzes question, decides if datasheet info needed]
     ↓
Ollama Datasheet Agent(s) (qwen2.5:3b)
     ↓
[Extract info from specific PDFs]
     ↓
Synthesized Response
     ↓
User
```

**All processing happens on your local machine!** 🎉

## Cost Comparison

| Provider | Cost | Privacy | Speed |
|----------|------|---------|-------|
| Claude API | ~$0.003/request | Cloud | Very Fast |
| Ollama Local | **FREE** | 100% Local | Fast (with GPU) |

## Next Steps

1. Start Ollama: `ollama serve`
2. Run the app: `python main.py`
3. Switch to Ollama in the dropdown
4. Upload datasheets
5. Watch the agent communication panel!
6. Ask questions about your PCB design

Enjoy zero-cost AI assistance! 🚀
