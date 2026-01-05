"""
Application configuration settings
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
SESSIONS_DIR = DATA_DIR / 'sessions'
UPLOADS_DIR = DATA_DIR / 'uploads'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Application settings
APP_CONFIG = {
    'title': 'Agentic AI Datasheet Agent',
    'host': '0.0.0.0',
    'port': 8080,
    'reload': True,
    'show': True
}

# UI Colors and Styling - Dark Purple/Grey Theme
COLORS = {
    'primary': '#8b7cc8',        # Medium purple
    'secondary': '#6b5b95',      # Dark purple
    'success': '#7c9885',        # Muted green-grey
    'warning': '#c8a87c',        # Muted orange-grey
    'error': '#c87c85',          # Muted red-grey
    'background': '#1a1625',     # Very dark purple-black
    'surface': '#2a2438',        # Dark purple-grey
    'surface_light': '#3d3550',  # Medium purple-grey
    'text': '#f5f3f7',           # Off-white with slight purple tint
    'text_secondary': '#a89fb8', # Light purple-grey
    'border': '#4a4158'          # Medium-dark purple-grey
}

# File upload settings
UPLOAD_CONFIG = {
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'allowed_extensions': ['.pdf', '.xlsx', '.xls', '.csv', '.txt', '.json']
}

# Chat settings
CHAT_CONFIG = {
    'max_history': 100,
    'placeholder': 'Ask me about your datasheets...',
    'not_implemented_msg': "I'm not implemented yet! 😊 But I'll be able to help you analyze datasheets soon."
}

# AI Agent settings
AGENT_CONFIG = {
    'api_key_file': DATA_DIR / '.api_key',  # Store API key securely
    'model': 'claude-3-5-sonnet-20241022',
    'max_tokens': 2048,
    'provider': 'ollama',  # 'claude' or 'ollama' - set to ollama by default
    'ollama_model': 'qwen3:8b',  # Changed to qwen3:8b (the model you have)
    'ollama_url': 'http://localhost:11434',
    'use_rag': True,  # Enable RAG pipeline for better datasheet processing
}

# Load API key if exists
def get_api_key() -> str:
    """Get API key from file or environment"""
    # Try to load from file first
    if AGENT_CONFIG['api_key_file'].exists():
        with open(AGENT_CONFIG['api_key_file'], 'r') as f:
            return f.read().strip()

    # Try environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    return api_key

def save_api_key(api_key: str) -> bool:
    """Save API key to file"""
    try:
        with open(AGENT_CONFIG['api_key_file'], 'w') as f:
            f.write(api_key)
        return True
    except Exception as e:
        print(f"Error saving API key: {e}")
        return False

def get_provider() -> str:
    """Get current AI provider"""
    provider_file = DATA_DIR / '.provider'
    if provider_file.exists():
        with open(provider_file, 'r') as f:
            return f.read().strip()
    return AGENT_CONFIG['provider']

def save_provider(provider: str) -> bool:
    """Save AI provider preference"""
    try:
        provider_file = DATA_DIR / '.provider'
        with open(provider_file, 'w') as f:
            f.write(provider)
        return True
    except Exception as e:
        print(f"Error saving provider: {e}")
        return False