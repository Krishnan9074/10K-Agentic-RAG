from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.messages import message_to_dict, messages_from_dict
from typing import Sequence
import os
import json


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, storage_path: str = "./chat_history"):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, session_id)
        os.makedirs(self.storage_path, exist_ok=True)

    @property
    def messages(self) -> list[BaseMessage]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return messages_from_dict(data)

    def add_message(self, message: BaseMessage) -> None:
        all_messages = list(self.messages)
        all_messages.append(message)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([message_to_dict(m) for m in all_messages],
                      f, ensure_ascii=False, indent=2)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)


def get_his(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(session_id)
