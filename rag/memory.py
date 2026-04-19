"""Sliding-window chat memory in Anthropic message format."""
from config import CHAT_MEMORY_TURNS


class ChatMemory:
    def __init__(self, max_turns: int = CHAT_MEMORY_TURNS):
        self._history: list[dict] = []
        self._max_turns = max_turns

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        max_messages = self._max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def get_messages(self) -> list[dict]:
        return list(self._history)

    def clear(self) -> None:
        self._history = []

    def __len__(self) -> int:
        return len(self._history) // 2
