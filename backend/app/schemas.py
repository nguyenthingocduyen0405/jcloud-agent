from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


OperationStatus = Literal["waiting_for_confirmation", "running", "completed", "failed", "cancelled"]
AllowedAction = Literal[
    "list_instances",
    "get_quota",
    "list_images",
    "list_flavors",
    "plan_create_instance",
    "start_instance",
    "stop_instance",
    "reboot_instance",
]


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=500)


class RequestIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._:-]+$")
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._:-]+$")
    project_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._:-]+$")


class ActionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operating_system: str | None = Field(default=None, max_length=80)
    operating_system_version: Literal["22.04", "24.04"] | None = None
    vcpus: int | None = Field(default=None, ge=1, le=128)
    ram_gb: int | None = Field(default=None, ge=1, le=1024)
    requires_gpu: bool | None = None
    name: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")


class LLMDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_type: Literal["action", "clarification", "answer"]
    action: AllowedAction | None = None
    pending_action: Literal[
        "plan_create_instance", "start_instance", "stop_instance", "reboot_instance"
    ] | None = None
    parameters: ActionParameters = Field(default_factory=ActionParameters)
    message: str = Field(min_length=1, max_length=500)
    requires_confirmation: bool = False

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "LLMDecision":
        if self.decision_type == "action" and self.action is None:
            raise ValueError("An action decision must include an allowed action")
        if self.decision_type != "action" and self.action is not None:
            raise ValueError("Only an action decision may include an action")
        if self.decision_type != "clarification" and self.pending_action is not None:
            raise ValueError("Only a clarification decision may include a pending action")
        if self.decision_type != "action" and self.requires_confirmation:
            raise ValueError("Only an action decision may request confirmation")
        return self


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    conversation_context: list[ConversationMessage] = Field(default_factory=list, max_length=10)


class Operation(BaseModel):
    id: str
    session_id: str
    user_id: str
    project_id: str
    action: Literal["create_instance", "start_instance", "stop_instance", "reboot_instance"]
    status: OperationStatus
    summary: str
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class ChatResponse(BaseModel):
    message: str
    operation: Operation | None = None
    data: Any | None = None


class SandboxResetResponse(BaseModel):
    status: Literal["reset"]
    instances: list[dict[str, Any]]
