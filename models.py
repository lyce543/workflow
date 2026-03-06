from enum import Enum
from pydantic import BaseModel


class ChatStatus(str, Enum):
    IDLE = "idle"
    STARTED = "started"
    FINISHED = "finished"
    BLOCKED = "blocked"


class StudentMessage(BaseModel):
    ub_id: int
    content: str
    files: list = []


class AssistantResponse(BaseModel):
    title: str = "-"
    text: str
    type: str = "interview"