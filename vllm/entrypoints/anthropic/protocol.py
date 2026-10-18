# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic models for Anthropic API protocol"""

import time
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

import vllm.envs as envs


class AnthropicError(BaseModel):
    """Error structure for Anthropic API"""

    type: str
    message: str


_ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def anthropic_error_type(status_code: int) -> str:
    """Anthropic's error type for an HTTP status.

    Clients such as the Anthropic SDKs and Claude Code read ``error.type``
    (for instance to retry on ``overloaded_error`` / ``api_error``), so the
    vLLM/OpenAI type names (``BadRequestError``, ``internal_error``...) must
    not leak into Anthropic error bodies or stream ``error`` events.
    """
    code = int(status_code)
    if code in _ANTHROPIC_ERROR_TYPES:
        return _ANTHROPIC_ERROR_TYPES[code]
    return "invalid_request_error" if 400 <= code < 500 else "api_error"


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API"""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


AnthropicBlockType = Literal[
    "text",
    "image",
    "tool_use",
    "tool_result",
    "tool_reference",
    "thinking",
    "redacted_thinking",
    "tool_addition",
    "tool_removal",
]


def block_as_text(block: dict[str, Any]) -> str:
    """Text for a content block the chat path has no other form for.

    ``search_result`` and text ``document`` blocks keep their text; a binary
    document (PDF) becomes a note that it was not shown, and any other block
    type a note naming it. Failing the request instead would fail every later
    request of the session too, since the block stays in the history.
    """
    kind = block.get("type")
    title = block.get("title")
    if kind == "search_result":
        texts = [
            c.get("text", "")
            for c in block.get("content") or []
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        head = f"Search result: {title}\nSource: {block.get('source')}"
        return "\n".join([head, *texts])
    if kind == "document":
        source = block.get("source") or {}
        if source.get("type") == "text":
            text = source.get("data", "")
        elif source.get("type") == "content":
            content = source.get("content")
            text = (
                content
                if isinstance(content, str)
                else "\n".join(
                    c.get("text", "")
                    for c in content or []
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            )
        else:
            name = f" {title!r}" if title else ""
            media = source.get("media_type") or source.get("type") or "unknown"
            return f"[document{name} ({media}) not shown: this model reads text only]"
        return f"Document: {title}\n{text}" if title else text
    return f"[{kind} block omitted]"


class AnthropicContentBlock(BaseModel):
    """Content block in message"""

    type: AnthropicBlockType
    text: str | None = None
    # For image content
    source: dict[str, Any] | None = None
    # For tool use/result
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
    # For tool_reference content
    tool_name: str | None = None
    # For thinking content
    thinking: str | None = None
    signature: str | None = None
    # For redacted thinking content (safety-filtered by the API)
    data: str | None = None
    # For tool_addition/tool_removal content
    tool: "AnthropicToolChange | None" = None

    @model_validator(mode="before")
    @classmethod
    def unsupported_block_as_text(cls, data: Any) -> Any:
        """``document``, ``search_result``, server-tool and future block types
        become text instead of failing validation (422)."""
        if isinstance(data, dict) and data.get("type") not in get_args(
            AnthropicBlockType
        ):
            return {"type": "text", "text": block_as_text(data)}
        return data

    @model_validator(mode="after")
    def validate_tool_change(self) -> "AnthropicContentBlock":
        if self.type not in ("tool_addition", "tool_removal"):
            return self
        if self.tool is None:
            raise ValueError(f"tool is required for {self.type} blocks")
        if self.type == "tool_removal" and self.tool.type != "tool_reference":
            raise ValueError("tool_removal only accepts a tool_reference")
        return self


def _reject_tool_changes(content: str | list[AnthropicContentBlock] | None) -> None:
    if content is None or isinstance(content, str):
        return
    for block in content:
        if block.type in ("tool_addition", "tool_removal"):
            raise ValueError(
                f'{block.type} blocks are only allowed in messages with role "system"'
            )


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant", "system"]
    content: str | list[AnthropicContentBlock]

    @model_validator(mode="after")
    def validate_tool_change_role(self) -> "AnthropicMessage":
        if self.role != "system":
            _reject_tool_changes(self.content)
        return self


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]
    strict: bool | None = None
    defer_loading: bool | None = None

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"  # Default to object type
        return v


class AnthropicToolChangeReference(BaseModel):
    """A tool named by a tool_addition or tool_removal block."""

    type: Literal["tool_reference"]
    name: str


class AnthropicToolChangeDefinition(BaseModel):
    """A tool defined by value in a tool_addition block."""

    type: Literal["tool_definition"]
    definition: AnthropicTool


AnthropicToolChange = Annotated[
    AnthropicToolChangeReference | AnthropicToolChangeDefinition,
    Field(discriminator="type"),
]


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition"""

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None

    @model_validator(mode="after")
    def validate_name_required_for_tool(self) -> "AnthropicToolChoice":
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice.name is required when type is 'tool'")
        return self


class AnthropicJsonOutputFormat(BaseModel):
    """JSON output format configuration"""

    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    type: Literal["json_schema"] = "json_schema"


AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
# reasoning_effort sent to the chat template for thinking.type=disabled.
AnthropicDisabledThinkingEffort = Literal["none", AnthropicEffort]
AnthropicDisabledThinkingEffortOption = Literal["auto", AnthropicDisabledThinkingEffort]
AnthropicThinkingDisplay = Literal["summarized", "omitted", "updates"]


class AnthropicOutputConfig(BaseModel):
    """Configuration options for the model's output, such as the output format."""

    effort: AnthropicEffort | None = None
    format: AnthropicJsonOutputFormat | None = None


class AnthropicThinkingConfigEnabled(BaseModel):
    """Extended thinking with a fixed token budget.

    ``display`` is accepted but ignored: reasoning is always returned.
    """

    type: Literal["enabled"]
    budget_tokens: int = Field(ge=1024)
    display: AnthropicThinkingDisplay | None = None


class AnthropicThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"]


class AnthropicThinkingConfigAdaptive(BaseModel):
    """Extended thinking whose depth the model chooses.

    ``display`` is accepted but ignored: reasoning is always returned.
    """

    type: Literal["adaptive"]
    display: AnthropicThinkingDisplay | None = None


AnthropicThinkingConfig = Annotated[
    AnthropicThinkingConfigEnabled
    | AnthropicThinkingConfigDisabled
    | AnthropicThinkingConfigAdaptive,
    Field(discriminator="type"),
]


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    output_config: AnthropicOutputConfig | None = None
    thinking: AnthropicThinkingConfig | None = None
    stop_sequences: (
        Annotated[list[str], Field(max_length=envs.VLLM_MAX_STOP_STRINGS)] | None
    ) = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None

    # vLLM-specific fields that are not in Anthropic spec
    cache_salt: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
    ec_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "ECTransfer parameters used for encoder-cache disaggregated serving."
        ),
    )
    vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None = Field(
        default=None,
        description=(
            "Additional request parameters with (list of) string or "
            "numeric values, used by custom extensions."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("system")
    @classmethod
    def validate_system(cls, v):
        _reject_tool_changes(v)
        return v

    @model_validator(mode="after")
    def validate_thinking_budget(self) -> "AnthropicMessagesRequest":
        if (
            isinstance(self.thinking, AnthropicThinkingConfigEnabled)
            and self.thinking.budget_tokens >= self.max_tokens
        ):
            raise ValueError("thinking.budget_tokens must be less than max_tokens")
        return self


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: (
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]
        | None
    ) = None
    text: str | None = None
    thinking: str | None = None
    partial_json: str | None = None
    signature: str | None = None

    # Message delta
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: "AnthropicMessagesResponse | None" = None
    delta: AnthropicDelta | None = None
    content_block: AnthropicContentBlock | None = None
    index: int | None = None
    error: AnthropicError | None = None
    usage: AnthropicUsage | None = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage | None = None

    # vLLM-specific fields that are not in Anthropic spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )
    ec_transfer_params: dict[str, Any] | None = Field(
        default=None, description="ECTransfer parameters."
    )

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


class AnthropicContextManagement(BaseModel):
    """Context management information for token counting."""

    original_input_tokens: int


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic messages.count_tokens request"""

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None

    # vLLM-specific fields that are not in Anthropic spec
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("system")
    @classmethod
    def validate_system(cls, v):
        _reject_tool_changes(v)
        return v


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic messages.count_tokens response"""

    input_tokens: int
    context_management: AnthropicContextManagement | None = None
