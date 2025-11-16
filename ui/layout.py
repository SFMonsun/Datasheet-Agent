"""
Main application layout
"""
from nicegui import ui
from config import COLORS
from ui.components.chat import ChatComponent
from ui.components.sidebar import DatasheetSidebar

class MainLayout:
    """Main application layout"""

    def __init__(self, session_manager):
        self.session_manager = session_manager
        self.chat_component = None
        self.sidebar_component = None

    async def build(self):
        """Build the main layout"""
        # Apply dark theme styling
        ui.dark_mode().enable()

        # Add custom CSS
        ui.add_head_html(f'''
            <style>
                body {{
                    background: {COLORS['background']} !important;
                }}
                .nicegui-content {{
                    padding: 0 !important;
                }}
                * {{
                    scrollbar-width: thin;
                    scrollbar-color: {COLORS['border']} {COLORS['surface']};
                }}
                *::-webkit-scrollbar {{
                    width: 8px;
                    height: 8px;
                }}
                *::-webkit-scrollbar-track {{
                    background: {COLORS['surface']};
                }}
                *::-webkit-scrollbar-thumb {{
                    background: {COLORS['border']};
                    border-radius: 4px;
                }}
                *::-webkit-scrollbar-thumb:hover {{
                    background: {COLORS['surface_light']};
                }}
            </style>
        ''')

        # Initialize session if not exists
        if not self.session_manager.current_session:
            self.session_manager.create_new_session()

        # Header (top level element)
        with ui.header().classes('items-center').style(
            f'background: {COLORS["surface"]}; border-bottom: 2px solid {COLORS["primary"]}; height: 64px; padding: 0 24px;'
        ):
            with ui.row().classes('w-full items-center justify-between'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('analytics', size='32px').style(f'color: {COLORS["primary"]}')
                    ui.label('Agentic AI Datasheet Agent').classes('text-2xl font-bold').style(
                        f'color: {COLORS["text"]}'
                    )

                # Session indicator
                if self.session_manager.current_session:
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('folder', size='20px').style(f'color: {COLORS["text_secondary"]}')
                        ui.label(self.session_manager.current_session.name).classes('text-sm').style(
                            f'color: {COLORS["text_secondary"]}'
                        )

        # Main content area
        with ui.row().classes('w-full').style(f'height: calc(100vh - 64px); background: {COLORS["background"]}; overflow: hidden;'):
            # Chat area (main content)
            with ui.column().classes('flex-grow h-full').style('padding: 24px; overflow: hidden;'):
                self.chat_component = ChatComponent(self.session_manager)
                await self.chat_component.build()

            # Sidebar
            with ui.column().classes('h-full').style(
                f'width: 350px; background: {COLORS["surface"]}; border-left: 1px solid {COLORS["border"]}; padding: 24px; overflow: hidden;'
            ):
                self.sidebar_component = DatasheetSidebar(self.session_manager)
                await self.sidebar_component.build()