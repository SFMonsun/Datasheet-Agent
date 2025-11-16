"""
Data models for the application
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path

@dataclass
class Datasheet:
    """Represents an uploaded datasheet"""
    id: str
    filename: str
    filepath: Path
    upload_date: datetime
    file_size: int
    file_type: str
    status: str = "active"  # active, processing, error

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'filename': self.filename,
            'filepath': str(self.filepath),
            'upload_date': self.upload_date.isoformat(),
            'file_size': self.file_size,
            'file_type': self.file_type,
            'status': self.status
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Datasheet':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            filename=data['filename'],
            filepath=Path(data['filepath']),
            upload_date=datetime.fromisoformat(data['upload_date']),
            file_size=data['file_size'],
            file_type=data['file_type'],
            status=data.get('status', 'active')
        )

@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class Session:
    """Represents a user session"""
    id: str
    name: str
    created_at: datetime
    datasheets: List[Datasheet] = field(default_factory=list)
    chat_history: List[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'datasheets': [ds.to_dict() for ds in self.datasheets],
            'chat_history': [msg.to_dict() for msg in self.chat_history]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Session':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            created_at=datetime.fromisoformat(data['created_at']),
            datasheets=[Datasheet.from_dict(ds) for ds in data.get('datasheets', [])],
            chat_history=[
                ChatMessage(
                    role=msg['role'],
                    content=msg['content'],
                    timestamp=datetime.fromisoformat(msg['timestamp'])
                ) for msg in data.get('chat_history', [])
            ]
        )