"""
Main entry point for the Agentic AI Datasheet Agent
"""
from nicegui import ui, app
from ui.layout import MainLayout
from services.session_manager import SessionManager
from config import APP_CONFIG


def main():
    """Initialize and run the application"""

    # Initialize session manager
    session_manager = SessionManager()

    @ui.page('/')
    async def index():
        """Main application page"""
        layout = MainLayout(session_manager)
        await layout.build()

    # Run the application
    ui.run(
        title=APP_CONFIG['title'],
        host=APP_CONFIG['host'],
        port=APP_CONFIG['port'],
        reload=APP_CONFIG['reload'],
        show=APP_CONFIG['show']
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()