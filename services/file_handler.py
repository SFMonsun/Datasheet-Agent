"""
File upload and management service
"""
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from models.datasheet import Datasheet
from config import UPLOADS_DIR, UPLOAD_CONFIG


class FileHandler:
    """Handles file uploads and management"""

    def __init__(self):
        self.uploads_dir = UPLOADS_DIR

    def save_uploaded_file(self, file_path: str, original_filename: str) -> Optional[Datasheet]:
        """Save an uploaded file and create a Datasheet object"""
        try:
            source = Path(file_path)

            # Check file size
            file_size = source.stat().st_size
            if file_size > UPLOAD_CONFIG['max_file_size']:
                print(f"File too large: {file_size} bytes")
                return None

            # Check file extension
            file_ext = source.suffix.lower()
            if file_ext not in UPLOAD_CONFIG['allowed_extensions']:
                print(f"File type not allowed: {file_ext}")
                return None

            # Create unique filename
            datasheet_id = str(uuid.uuid4())
            safe_filename = f"{datasheet_id}{file_ext}"
            destination = self.uploads_dir / safe_filename

            # Copy file
            shutil.copy2(source, destination)

            # Create Datasheet object
            datasheet = Datasheet(
                id=datasheet_id,
                filename=original_filename,
                filepath=destination,
                upload_date=datetime.now(),
                file_size=file_size,
                file_type=file_ext[1:]  # Remove the dot
            )

            return datasheet

        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def delete_file(self, datasheet: Datasheet) -> bool:
        """Delete a datasheet file"""
        try:
            if datasheet.filepath.exists():
                datasheet.filepath.unlink()
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size for display"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"