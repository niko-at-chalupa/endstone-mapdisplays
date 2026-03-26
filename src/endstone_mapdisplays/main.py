from endstone import ColorFormat
from endstone.event import event_handler, PlayerJoinEvent, PlayerChatEvent
from endstone.plugin import Plugin

class TemplatePlugin(Plugin):
    def on_enable(self) -> None:
        self.logger.info("on_enable is called!")
        self.register_events(self)