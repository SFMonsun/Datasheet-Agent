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

# UI Colors and Styling
COLORS = {
    'primary': '#3b82f6',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'background': '#0f172a',
    'surface': '#1e293b',
    'surface_light': '#334155',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'border': '#475569'
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