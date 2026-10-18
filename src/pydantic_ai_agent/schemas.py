from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, model_validator

# re-export: purane `from pydantic_ai_agent.schemas import ConversationStatus` chalte rahein
from pydantic_ai_agent.enums import ConversationStatus  # noqa: F401


class Token(BaseModel):
    access_token: str
    token_type: str


# strip karne se "   " jaisa khali message bhi min_length pe fail hoga
ChatMessage = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    message: ChatMessage


class ToolApprovalItem(BaseModel):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]  # sirf dikhane ke liye, client se kabhi nahi aata


class ChatResponse(BaseModel):
    conversation_id: UUID
    reply: str | None = None
    pending_approvals: list[ToolApprovalItem] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    tool_call_id: str
    approved: bool


class ApprovalRequest(BaseModel):
    decisions: list[ApprovalDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def no_duplicate_ids(self):
        ids = [d.tool_call_id for d in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate tool_call_id")
        return self