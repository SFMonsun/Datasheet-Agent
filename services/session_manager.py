"""
Session management service
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from models.datasheet import Session, Datasheet
from config import SESSIONS_DIR


class SessionManager:
    """Manages user sessions and persistence"""

    def __init__(self):
        self.current_session: Optional[Session] = None
        self.sessions_dir = SESSIONS_DIR

    def create_new_session(self, name: Optional[str] = None) -> Session:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        if name is None:
            name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        self.current_session = Session(
            id=session_id,
            name=name,
            created_at=datetime.now()
        )
        return self.current_session

    def save_session(self, session: Session) -> bool:
        """Save session to disk"""
        try:
            session_file = self.sessions_dir / f"{session.id}.json"
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load session from disk"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                return None

            with open(session_file, 'r') as f:
                data = json.load(f)

            session = Session.from_dict(data)
            self.current_session = session
            return session
        except Exception as e:
            print(f"Error loading session: {e}")
            return None

    def list_sessions(self) -> List[dict]:
        """List all saved sessions"""
        sessions = []
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                sessions.append({
                    'id': data['id'],
                    'name': data['name'],
                    'created_at': data['created_at'],
                    'datasheet_count': len(data.get('datasheets', []))
                })
            except Exception as e:
                print(f"Error reading session {session_file}: {e}")

        # Sort by creation date, newest first
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False

    def add_datasheet(self, datasheet: Datasheet):
        """Add a datasheet to the current session"""
        if self.current_session:
            self.current_session.datasheets.append(datasheet)

    def remove_datasheet(self, datasheet_id: str):
        """Remove a datasheet from the current session"""
        if self.current_session:
            self.current_session.datasheets = [
                ds for ds in self.current_session.datasheets
                if ds.id != datasheet_id
            ]

    def add_chat_message(self, role: str, content: str):
        """Add a chat message to the current session"""
        if self.current_session:
            from models.datasheet import ChatMessage
            message = ChatMessage(role=role, content=content)
            self.current_session.chat_history.append(message)