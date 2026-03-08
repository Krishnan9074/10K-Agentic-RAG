from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from typing import Sequence


import os, json, re
from langchain_core.messages import message_to_dict, messages_from_dict

_SAFE_SESSION_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        if not _SAFE_SESSION_ID.match(session_id):
            raise ValueError(
                f"Invalid session_id '{session_id}': only alphanumeric characters, "
                "underscores, and hyphens are allowed."
            )
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
    def add_messages(self,messages: Sequence[BaseMessage])->None:
        all_messages=list(self.messages)
        all_messages.extend(messages)
        new_messages=[message_to_dict(message) for message in all_messages]
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)
    def add_message(self, message: BaseMessage) -> None:
        messages = list(self.messages)
        messages.append(message)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(
                [message_to_dict(m) for m in messages],
                f,
                ensure_ascii=False,
                indent=2
            )
    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data=json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []
    def clear(self) ->None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)



def get_his(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")