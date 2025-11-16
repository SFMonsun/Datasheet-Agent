"""
Chat interface component
"""
from nicegui import ui
from config import COLORS, CHAT_CONFIG
from models.datasheet import ChatMessage


class ChatComponent:
    """Chat interface component"""

    def __init__(self, session_manager):
        self.session_manager = session_manager
        self.chat_container = None
        self.input_field = None

    async def build(self):
        """Build the chat interface"""
        with ui.column().classes('w-full h-full flex flex-col'):
            # Header
            with ui.card().classes('w-full mb-4').style(
                    f'background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Main Agent Chat').classes('text-xl font-bold').style(f'color: {COLORS["text"]}')
                    with ui.row().classes('gap-2'):
                        ui.button('Clear Chat', on_click=self.clear_chat).props('flat dense').classes('text-xs').style(
                            f'color: {COLORS["text_secondary"]}'
                        )

            # Chat messages container
            with ui.scroll_area().classes('flex-grow w-full mb-4').style(
                    f'background: {COLORS["surface"]}; border-radius: 8px; padding: 16px; border: 1px solid {COLORS["border"]}'
            ) as scroll:
                self.chat_container = ui.column().classes('w-full gap-3')
                self.scroll_area = scroll

            # Input area
            with ui.card().classes('w-full').style(
                    f'background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}'):
                with ui.row().classes('w-full items-center gap-2'):
                    self.input_field = ui.input(
                        placeholder=CHAT_CONFIG['placeholder']
                    ).classes('flex-grow').style(f'color: {COLORS["text"]}').props('outlined dense')

                    self.input_field.on('keydown.enter', self.send_message)

                    ui.button(
                        icon='send',
                        on_click=self.send_message
                    ).props('flat round').style(f'color: {COLORS["primary"]}')

        # Load chat history if exists
        await self.load_chat_history()

    async def send_message(self):
        """Send a chat message"""
        message = self.input_field.value.strip()
        if not message:
            return

        # Clear input
        self.input_field.value = ''

        # Add user message
        self.add_message('user', message)

        # Save to session
        self.session_manager.add_chat_message('user', message)

        # Simulate agent response
        await ui.run_javascript('new Promise(r => setTimeout(r, 500))')
        response = CHAT_CONFIG['not_implemented_msg']
        self.add_message('assistant', response)

        # Save response to session
        self.session_manager.add_chat_message('assistant', response)

        # Scroll to bottom
        self.scroll_area.scroll_to(percent=1.0)

    def add_message(self, role: str, content: str):
        """Add a message to the chat display"""
        with self.chat_container:
            if role == 'user':
                with ui.row().classes('w-full justify-end'):
                    with ui.card().classes('max-w-[70%]').style(
                            f'background: {COLORS["primary"]}; color: white; padding: 12px 16px;'
                    ):
                        ui.label(content).classes('text-sm whitespace-pre-wrap')
            else:
                with ui.row().classes('w-full justify-start'):
                    with ui.card().classes('max-w-[70%]').style(
                            f'background: {COLORS["surface_light"]}; color: {COLORS["text"]}; padding: 12px 16px;'
                    ):
                        ui.label(content).classes('text-sm whitespace-pre-wrap')

    async def clear_chat(self):
        """Clear the chat history"""
        self.chat_container.clear()
        if self.session_manager.current_session:
            self.session_manager.current_session.chat_history.clear()
        ui.notify('Chat cleared', type='info')

    async def load_chat_history(self):
        """Load chat history from current session"""
        if self.session_manager.current_session:
            for message in self.session_manager.current_session.chat_history:
                self.add_message(message.role, message.content)
            if self.session_manager.current_session.chat_history:
                self.scroll_area.scroll_to(percent=1.0)