"""
Datasheet sidebar component
"""
from nicegui import ui, events
from config import COLORS
from services.file_handler import FileHandler
from datetime import datetime


class DatasheetSidebar:
    """Sidebar showing uploaded datasheets"""

    def __init__(self, session_manager):
        self.session_manager = session_manager
        self.file_handler = FileHandler()
        self.datasheets_container = None
        self.upload_dialog = None

    async def build(self):
        """Build the sidebar"""
        with ui.column().classes('w-full h-full flex flex-col gap-4'):
            # Header with add button
            with ui.card().classes('w-full').style(
                    f'background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Datasheets').classes('text-lg font-bold').style(f'color: {COLORS["text"]}')
                    ui.button(
                        icon='add',
                        on_click=self.show_upload_dialog
                    ).props('flat round').style(f'color: {COLORS["success"]}').tooltip('Add Datasheet')

            # Datasheets list
            with ui.scroll_area().classes('flex-grow w-full').style('border-radius: 8px;'):
                self.datasheets_container = ui.column().classes('w-full gap-2')

            # Session controls
            with ui.card().classes('w-full').style(
                    f'background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}'):
                ui.label('Session').classes('text-sm font-semibold mb-2').style(f'color: {COLORS["text_secondary"]}')
                with ui.column().classes('w-full gap-2'):
                    ui.button(
                        'Save Session',
                        on_click=self.save_session,
                        icon='save'
                    ).props('outline').classes('w-full').style(f'color: {COLORS["success"]}')

                    ui.button(
                        'Load Session',
                        on_click=self.show_load_session_dialog,
                        icon='folder_open'
                    ).props('outline').classes('w-full').style(f'color: {COLORS["primary"]}')

                    ui.button(
                        'New Session',
                        on_click=self.new_session,
                        icon='add_circle'
                    ).props('outline').classes('w-full').style(f'color: {COLORS["secondary"]}')

        # Load datasheets if session exists
        await self.refresh_datasheets()

    def show_upload_dialog(self):
        """Show file upload dialog"""
        with ui.dialog() as dialog, ui.card().style(f'background: {COLORS["surface"]}; min-width: 400px'):
            ui.label('Upload Datasheet').classes('text-xl font-bold mb-4').style(f'color: {COLORS["text"]}')

            uploaded_files = []

            async def handle_upload(e: events.UploadEventArguments):
                """Handle file upload and store for batch processing"""
                import tempfile
                import os

                # Access file object (NiceGUI stores uploaded file in e.file)
                if not hasattr(e, 'file') or e.file is None:
                    ui.notify('Could not access uploaded file', type='negative')
                    return

                file_obj = e.file
                filename = file_obj.name

                # Create a temporary file with the same extension
                file_ext = os.path.splitext(filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    # Read content from file object (await the async read)
                    content = await file_obj.read()
                    temp_file.write(content)
                    temp_path = temp_file.name

                # Store for batch upload
                uploaded_files.append({'temp_path': temp_path, 'filename': filename})

            async def confirm_uploads():
                """Process all uploaded files"""
                if not uploaded_files:
                    ui.notify('No files selected', type='warning')
                    return

                success_count = 0
                for file_info in uploaded_files:
                    try:
                        datasheet = self.file_handler.save_uploaded_file(
                            file_info['temp_path'],
                            file_info['filename']
                        )
                        if datasheet:
                            self.session_manager.add_datasheet(datasheet)
                            success_count += 1
                    finally:
                        # Clean up temporary file
                        import os
                        if os.path.exists(file_info['temp_path']):
                            os.unlink(file_info['temp_path'])

                if success_count > 0:
                    await self.refresh_datasheets()
                    ui.notify(f'Uploaded {success_count} datasheet(s)', type='positive')
                    dialog.close()
                else:
                    ui.notify('Failed to upload files', type='negative')

            ui.upload(
                on_upload=handle_upload,
                multiple=True,
                label='Choose file(s)'
            ).props('accept=".pdf,.xlsx,.xls,.csv,.txt,.json"').classes('w-full').style(
                f'color: {COLORS["text"]}'
            )

            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat').style(f'color: {COLORS["text_secondary"]}')
                ui.button('Upload', on_click=confirm_uploads).props('flat').style(
                    f'color: {COLORS["primary"]}'
                )

        dialog.open()

    async def refresh_datasheets(self):
        """Refresh the datasheets list"""
        self.datasheets_container.clear()

        if not self.session_manager.current_session:
            with self.datasheets_container:
                ui.label('No session loaded').classes('text-sm text-center p-4').style(
                    f'color: {COLORS["text_secondary"]}'
                )
            return

        datasheets = self.session_manager.current_session.datasheets

        if not datasheets:
            with self.datasheets_container:
                ui.label('No datasheets uploaded').classes('text-sm text-center p-4').style(
                    f'color: {COLORS["text_secondary"]}'
                )
            return

        with self.datasheets_container:
            for datasheet in datasheets:
                self.create_datasheet_card(datasheet)

    def create_datasheet_card(self, datasheet):
        """Create a card for a datasheet"""
        with ui.card().classes('w-full cursor-pointer hover:shadow-lg transition-shadow').style(
                f'background: {COLORS["surface_light"]}; border: 1px solid {COLORS["border"]}; padding: 12px;'
        ):
            with ui.row().classes('w-full items-start justify-between'):
                with ui.column().classes('flex-grow gap-1'):
                    # Filename
                    ui.label(datasheet.filename).classes('font-semibold text-sm').style(
                        f'color: {COLORS["text"]}'
                    ).classes('break-all')

                    # File info
                    file_size = self.file_handler.format_file_size(datasheet.file_size)
                    upload_date = datasheet.upload_date.strftime('%Y-%m-%d %H:%M')
                    ui.label(f'{datasheet.file_type.upper()} • {file_size}').classes('text-xs').style(
                        f'color: {COLORS["text_secondary"]}'
                    )
                    ui.label(upload_date).classes('text-xs').style(
                        f'color: {COLORS["text_secondary"]}'
                    )

                # Delete button
                ui.button(
                    icon='delete',
                    on_click=lambda: self.delete_datasheet(datasheet)
                ).props('flat round dense').style(f'color: {COLORS["error"]}').tooltip('Delete')

    async def delete_datasheet(self, datasheet):
        """Delete a datasheet"""
        self.session_manager.remove_datasheet(datasheet.id)
        self.file_handler.delete_file(datasheet)
        await self.refresh_datasheets()
        ui.notify(f'Deleted {datasheet.filename}', type='info')

    async def save_session(self):
        """Save current session"""
        if not self.session_manager.current_session:
            ui.notify('No active session to save', type='warning')
            return

        success = self.session_manager.save_session(self.session_manager.current_session)
        if success:
            ui.notify(f'Session "{self.session_manager.current_session.name}" saved', type='positive')
        else:
            ui.notify('Failed to save session', type='negative')

    def show_load_session_dialog(self):
        """Show dialog to load a session"""
        sessions = self.session_manager.list_sessions()

        with ui.dialog() as dialog, ui.card().style(f'background: {COLORS["surface"]}; min-width: 500px'):
            ui.label('Load Session').classes('text-xl font-bold mb-4').style(f'color: {COLORS["text"]}')

            if not sessions:
                ui.label('No saved sessions found').classes('text-center p-4').style(
                    f'color: {COLORS["text_secondary"]}'
                )
            else:
                with ui.column().classes('w-full gap-2 max-h-96 overflow-auto'):
                    for session in sessions:
                        with ui.card().classes('w-full cursor-pointer hover:shadow-lg').style(
                                f'background: {COLORS["surface_light"]}; border: 1px solid {COLORS["border"]}'
                        ).on('click', lambda s=session: self.load_session_handler(s['id'], dialog)):
                            with ui.row().classes('w-full items-center justify-between'):
                                with ui.column().classes('gap-1'):
                                    ui.label(session['name']).classes('font-semibold').style(
                                        f'color: {COLORS["text"]}'
                                    )
                                    created = datetime.fromisoformat(session['created_at']).strftime('%Y-%m-%d %H:%M')
                                    ui.label(f'{created} • {session["datasheet_count"]} datasheets').classes(
                                        'text-xs').style(
                                        f'color: {COLORS["text_secondary"]}'
                                    )
                                ui.icon('chevron_right').style(f'color: {COLORS["primary"]}')

            with ui.row().classes('w-full justify-end mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat').style(f'color: {COLORS["text_secondary"]}')

        dialog.open()

    async def load_session_handler(self, session_id: str, dialog):
        """Load a session"""
        session = self.session_manager.load_session(session_id)
        if session:
            await self.refresh_datasheets()
            ui.notify(f'Loaded session "{session.name}"', type='positive')
            dialog.close()
        else:
            ui.notify('Failed to load session', type='negative')

    async def new_session(self):
        """Create a new session"""
        self.session_manager.create_new_session()
        await self.refresh_datasheets()
        ui.notify('New session created', type='info')