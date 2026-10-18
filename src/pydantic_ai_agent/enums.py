from enum import StrEnum


class ConversationStatus(StrEnum):
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_APPROVAL = "awaiting_approval"