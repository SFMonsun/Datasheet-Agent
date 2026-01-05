"""
Chat interface component
"""
from nicegui import ui
from config import COLORS, CHAT_CONFIG, get_api_key, get_provider, save_provider, AGENT_CONFIG
from models.datasheet import ChatMessage
from services.orchestrator_agent import OrchestratorAgent
from services.datasheet_agent import DatasheetAgent
from services.ollama_orchestrator_agent import OllamaOrchestratorAgent
from services.ollama_datasheet_agent import OllamaDatasheetAgent
from services.ollama_datasheet_agent_rag import OllamaDatasheetAgentRAG


class ChatComponent:
    """Chat interface component"""

    def __init__(self, session_manager):
        self.session_manager = session_manager
        self.chat_container = None
        self.input_field = None
        self.orchestrator = None
        self.api_key = None
        self.is_processing = False
        self.provider = get_provider()
        self.provider_selector = None
        self.agent_viz_container = None
        self.rag_viz_container = None  # For RAG pipeline visualization

    async def build(self):
        """Build the chat interface"""
        # Initialize orchestrator based on provider
        await self._initialize_orchestrator()

        with ui.column().classes('w-full h-full flex flex-col'):
            # Header
            with ui.card().classes('w-full mb-4').style(
                    f'background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('PCB Design Assistant').classes('text-xl font-bold').style(f'color: {COLORS["text"]}')
                    with ui.row().classes('gap-2 items-center'):
                        # Provider selector
                        ui.label('Model:').classes('text-sm').style(f'color: {COLORS["text_secondary"]}')
                        self.provider_selector = ui.select(
                            options={'claude': 'Claude (API)', 'ollama': 'Ollama (Local)'},
                            value=self.provider,
                            on_change=self.on_provider_change
                        ).classes('w-40').style(f'color: {COLORS["text"]}')

                        if self.provider == 'claude' and not get_api_key():
                            ui.button('Set API Key', on_click=self.show_api_key_dialog, icon='key').props('flat dense').style(
                                f'color: {COLORS["warning"]}'
                            ).tooltip('Configure Claude API Key')
                        ui.button('Clear Chat', on_click=self.clear_chat, icon='delete').props('flat dense').classes('text-xs').style(
                            f'color: {COLORS["text_secondary"]}'
                        )

            # Agent Communication Visualization (if Ollama)
            if self.provider == 'ollama':
                with ui.card().classes('w-full mb-4').style(
                        f'background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; max-height: 150px;'):
                    ui.label('Agent Communication').classes('text-sm font-semibold mb-2').style(f'color: {COLORS["text_secondary"]}')
                    with ui.scroll_area().classes('w-full').style('height: 100px;'):
                        self.agent_viz_container = ui.column().classes('w-full gap-1')

                # RAG Pipeline Visualization (if using RAG)
                if AGENT_CONFIG.get('use_rag', False):
                    with ui.card().classes('w-full mb-4').style(
                            f'background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; max-height: 200px;'):
                        ui.label('RAG Pipeline').classes('text-sm font-semibold mb-2').style(f'color: {COLORS["warning"]}')
                        with ui.scroll_area().classes('w-full').style('height: 150px;'):
                            self.rag_viz_container = ui.column().classes('w-full gap-1')

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

        if self.is_processing:
            ui.notify('Please wait for the current response to complete', type='warning')
            return

        # Clear input
        self.input_field.value = ''

        # Add user message
        self.add_message('user', message)

        # Save to session
        self.session_manager.add_chat_message('user', message)

        # Check if orchestrator is configured
        if not self.orchestrator:
            if self.provider == 'claude':
                response = "⚠️ Please configure your Claude API key using the 'Set API Key' button above."
            else:
                response = "⚠️ Please ensure Ollama is running (ollama serve) and the model is available."
            self.add_message('assistant', response)
            self.session_manager.add_chat_message('assistant', response)
            self.scroll_area.scroll_to(percent=1.0)
            return

        # Show typing indicator
        self.is_processing = True
        typing_msg = self.add_typing_indicator()

        try:
            # Sync datasheet agents
            await self._sync_datasheet_agents()

            # Show agent interactions visualization area if using Ollama
            if self.agent_viz_container:
                self.agent_viz_container.clear()
                with self.agent_viz_container:
                    ui.label('Processing query...').classes('text-xs').style(f'color: {COLORS["text_secondary"]}')

            # Get response from orchestrator
            result = await self.orchestrator.chat(message)
            response = result['response']

            # Remove typing indicator
            try:
                typing_msg.delete()
            except (ValueError, RuntimeError):
                pass  # Indicator already removed

            # Show agent interactions if available
            print(f"[ChatUI] Agent interactions in result: {len(result.get('agent_interactions', []))}")
            if 'agent_interactions' in result and result['agent_interactions']:
                if self.agent_viz_container:
                    print(f"[ChatUI] Visualizing {len(result['agent_interactions'])} agent interactions")
                    self.visualize_agent_interactions(result['agent_interactions'])
                else:
                    print("[ChatUI] WARNING: agent_viz_container is None!")
            else:
                print("[ChatUI] No agent interactions to display")

            # Show RAG pipeline steps if available
            if 'rag_steps' in result and result['rag_steps']:
                if self.rag_viz_container:
                    print(f"[ChatUI] Visualizing {len(result['rag_steps'])} RAG steps")
                    self.visualize_rag_steps(result['rag_steps'])
                else:
                    print("[ChatUI] WARNING: rag_viz_container is None!")

            # Add assistant response
            self.add_message('assistant', response)

            # Save response to session
            self.session_manager.add_chat_message('assistant', response)

        except Exception as e:
            # Remove typing indicator
            try:
                typing_msg.delete()
            except (ValueError, RuntimeError):
                pass  # Indicator already removed

            error_response = f"Error: {str(e)}\n\nPlease check your API key and try again."
            self.add_message('assistant', error_response)
            self.session_manager.add_chat_message('assistant', error_response)

        finally:
            self.is_processing = False

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
                        ui.markdown(content).classes('text-sm').style('color: white;')
            else:
                with ui.row().classes('w-full justify-start'):
                    with ui.card().classes('max-w-[70%]').style(
                            f'background: {COLORS["surface_light"]}; color: {COLORS["text"]}; padding: 12px 16px;'
                    ):
                        ui.markdown(content).classes('text-sm').style(f'color: {COLORS["text"]};')

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

    def add_typing_indicator(self):
        """Add a typing indicator"""
        with self.chat_container:
            with ui.row().classes('w-full justify-start') as row:
                with ui.card().classes('max-w-[70%]').style(
                        f'background: {COLORS["surface_light"]}; color: {COLORS["text"]}; padding: 12px 16px;'
                ):
                    ui.spinner(size='sm').classes('mr-2')
                    ui.label('Thinking...').classes('text-sm')
        return row

    async def _sync_datasheet_agents(self):
        """Sync datasheet agents with current session datasheets"""
        if not self.orchestrator or not self.session_manager.current_session:
            return

        current_datasheets = self.session_manager.current_session.datasheets
        current_ids = {ds.id for ds in current_datasheets}

        # Remove agents for deleted datasheets
        for agent_id in list(self.orchestrator.datasheet_agents.keys()):
            if agent_id not in current_ids:
                self.orchestrator.unregister_datasheet_agent(agent_id)

        # Add agents for new datasheets
        new_agents_registered = False
        for datasheet in current_datasheets:
            if datasheet.id not in self.orchestrator.datasheet_agents:
                if self.provider == 'claude':
                    agent = DatasheetAgent(
                        datasheet_id=datasheet.id,
                        filename=datasheet.filename,
                        filepath=datasheet.filepath,
                        api_key=self.api_key
                    )
                else:  # ollama
                    # Use RAG-enhanced agent if enabled
                    if AGENT_CONFIG.get('use_rag', False):
                        agent = OllamaDatasheetAgentRAG(
                            datasheet_id=datasheet.id,
                            filename=datasheet.filename,
                            filepath=datasheet.filepath,
                            model=AGENT_CONFIG['ollama_model'],
                            ollama_url=AGENT_CONFIG['ollama_url']
                        )
                    else:
                        agent = OllamaDatasheetAgent(
                            datasheet_id=datasheet.id,
                            filename=datasheet.filename,
                            filepath=datasheet.filepath,
                            model=AGENT_CONFIG['ollama_model'],
                            ollama_url=AGENT_CONFIG['ollama_url']
                        )
                await self.orchestrator.register_datasheet_agent(agent)
                new_agents_registered = True

        # Show registration interactions if new agents were added
        if new_agents_registered and hasattr(self.orchestrator, 'registration_interactions'):
            if self.agent_viz_container and self.orchestrator.registration_interactions:
                self.visualize_agent_interactions(self.orchestrator.registration_interactions)

    def show_api_key_dialog(self):
        """Show dialog to configure API key"""
        from config import save_api_key

        with ui.dialog() as dialog, ui.card().style(f'background: {COLORS["surface"]}; min-width: 500px'):
            ui.label('Configure Claude API Key').classes('text-xl font-bold mb-4').style(f'color: {COLORS["text"]}')

            ui.label('Enter your Anthropic API key:').classes('text-sm mb-2').style(f'color: {COLORS["text_secondary"]}')

            api_key_input = ui.input(
                placeholder='sk-ant-...',
                password=True,
                password_toggle_button=True
            ).classes('w-full').style(f'color: {COLORS["text"]}')

            ui.label('You can get your API key from console.anthropic.com').classes('text-xs mt-2').style(
                f'color: {COLORS["text_secondary"]}'
            )

            async def save_key():
                key = api_key_input.value.strip()
                if not key:
                    ui.notify('Please enter an API key', type='warning')
                    return

                if save_api_key(key):
                    self.api_key = key
                    self.orchestrator = OrchestratorAgent(key)
                    await self._sync_datasheet_agents()
                    ui.notify('API key saved successfully', type='positive')
                    dialog.close()
                    # Rebuild to show updated UI
                    ui.run_javascript('location.reload()')
                else:
                    ui.notify('Failed to save API key', type='negative')

            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat').style(f'color: {COLORS["text_secondary"]}')
                ui.button('Save', on_click=save_key).style(f'color: {COLORS["primary"]}')

        dialog.open()

    async def _initialize_orchestrator(self):
        """Initialize the orchestrator based on selected provider"""
        if self.provider == 'claude':
            self.api_key = get_api_key()
            if self.api_key:
                self.orchestrator = OrchestratorAgent(self.api_key)
                await self._sync_datasheet_agents()
        else:  # ollama
            self.orchestrator = OllamaOrchestratorAgent(
                model=AGENT_CONFIG['ollama_model'],
                ollama_url=AGENT_CONFIG['ollama_url']
            )
            await self._sync_datasheet_agents()

    async def on_provider_change(self):
        """Handle provider selection change"""
        self.provider = self.provider_selector.value
        save_provider(self.provider)
        ui.notify(f'Switching to {self.provider}...', type='info')
        # Reload page to reinitialize with new provider
        ui.run_javascript('location.reload()')

    def visualize_agent_interactions(self, interactions):
        """Visualize agent communication"""
        if not self.agent_viz_container:
            return

        self.agent_viz_container.clear()
        with self.agent_viz_container:
            for interaction in interactions:
                if interaction['type'] == 'registration':
                    # Special styling for registration interactions
                    with ui.card().classes('w-full').style(
                        f'background: {COLORS["surface_light"]}; border-left: 3px solid {COLORS["secondary"]}; padding: 8px; margin-bottom: 4px;'
                    ):
                        with ui.row().classes('w-full items-center gap-2'):
                            ui.icon('app_registration').style(f'color: {COLORS["secondary"]}; font-size: 16px;')
                            comp_info = interaction.get('component_info', {})
                            ui.label(f"{interaction['from']} registered").classes('text-xs font-bold').style(
                                f'color: {COLORS["secondary"]}'
                            )
                        with ui.row().classes('w-full ml-6'):
                            ui.label(f"{comp_info.get('name', 'Unknown')} ({comp_info.get('type', 'Unknown')})").classes('text-xs').style(
                                f'color: {COLORS["text"]}'
                            )

                elif interaction['type'] == 'query':
                    with ui.row().classes('w-full items-center gap-2'):
                        ui.icon('arrow_forward').style(f'color: {COLORS["primary"]}; font-size: 16px;')
                        ui.label(f"{interaction['from']} → {interaction['to']}").classes('text-xs').style(
                            f'color: {COLORS["text_secondary"]}'
                        )
                        ui.label(f'"{interaction["question"][:40]}..."').classes('text-xs italic').style(
                            f'color: {COLORS["text"]}'
                        )
                elif interaction['type'] == 'response':
                    with ui.row().classes('w-full items-center gap-2 ml-4'):
                        ui.icon('arrow_back').style(f'color: {COLORS["success"]}; font-size: 16px;')
                        ui.label(f"{interaction['from']} → {interaction['to']}").classes('text-xs').style(
                            f'color: {COLORS["text_secondary"]}'
                        )
                        ui.label('Response received').classes('text-xs').style(f'color: {COLORS["success"]}')

    def visualize_rag_steps(self, rag_steps):
        """Visualize RAG pipeline steps"""
        if not self.rag_viz_container or not rag_steps:
            return

        self.rag_viz_container.clear()
        with self.rag_viz_container:
            for step in rag_steps:
                step_type = step.get('step', '')
                description = step.get('description', '')
                status = step.get('status', 'unknown')

                # Choose icon and color based on step type
                icon_map = {
                    'pdf_extraction': 'picture_as_pdf',
                    'chunking': 'content_cut',
                    'model_loading': 'model_training',
                    'embedding': 'scatter_plot',
                    'retrieval': 'search'
                }
                icon = icon_map.get(step_type, 'settings')

                # Status colors
                status_color = {
                    'in_progress': COLORS['warning'],
                    'completed': COLORS['success'],
                    'error': COLORS['error']
                }.get(status, COLORS['text_secondary'])

                with ui.row().classes('w-full items-center gap-2'):
                    ui.icon(icon).style(f'color: {status_color}; font-size: 16px;')
                    ui.label(description).classes('text-xs').style(f'color: {COLORS["text"]}')

                    # Show additional info for certain steps
                    if step_type == 'pdf_extraction' and 'pages_extracted' in step:
                        ui.label(f"({step['pages_extracted']} pages)").classes('text-xs').style(f'color: {COLORS["text_secondary"]}')
                    elif step_type == 'chunking' and 'chunks_created' in step:
                        ui.label(f"({step['chunks_created']} chunks)").classes('text-xs').style(f'color: {COLORS["text_secondary"]}')
                    elif step_type == 'retrieval' and 'chunks_retrieved' in step:
                        ui.label(f"({step['chunks_retrieved']} found)").classes('text-xs').style(f'color: {COLORS["text_secondary"]}')

                # Show retrieved chunks details
                if step_type == 'retrieval' and 'chunks' in step:
                    for chunk in step['chunks'][:3]:  # Show first 3 chunks
                        with ui.row().classes('w-full items-center gap-2 ml-6'):
                            ui.icon('description').style(f'color: {COLORS["success"]}; font-size: 14px;')
                            # Highlight page number using markdown instead of html
                            ui.label(f"**Page {chunk['page']}** (Relevance: {chunk['similarity']:.0%}, {chunk['type']})").classes('text-xs').style(f'color: {COLORS["text"]}')
                            # Show preview on hover
                            if 'preview' in chunk:
                                ui.tooltip(chunk['preview'])