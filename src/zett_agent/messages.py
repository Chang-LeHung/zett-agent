from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar


class MessageRole(StrEnum):
    """Roles understood by the provider-neutral agent protocol."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    AGENT = "agent"


class ImageDetail(StrEnum):
    """Provider-neutral image fidelity requested from a vision-capable model."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class TextContent:
    """One text block in an ordered multimodal message."""

    text: str


@dataclass(frozen=True, slots=True)
class ImageUrlSource:
    """Remote or data URL containing an image."""

    url: str

    def __post_init__(self) -> None:
        value = self.url.strip()
        if not value:
            raise ValueError("Image URL cannot be empty")
        if not value.startswith(("https://", "http://", "data:image/")):
            raise ValueError("Image URL must use HTTP, HTTPS, or an image data URL")


@dataclass(frozen=True, slots=True)
class ImageBytesSource:
    """In-memory encoded image bytes and their MIME type."""

    data: bytes
    media_type: str

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("Image data cannot be empty")
        _validate_image_media_type(self.media_type)


type ImageSource = ImageUrlSource | ImageBytesSource


@dataclass(frozen=True, slots=True)
class ImageContent:
    """One image block in an ordered multimodal message."""

    source: ImageSource
    detail: ImageDetail = ImageDetail.AUTO
    alt_text: str | None = None


type UserContentPart = TextContent | ImageContent
type UserContent = str | list[UserContentPart]


def _validate_image_media_type(media_type: str) -> None:
    if not media_type.strip().lower().startswith("image/"):
        raise ValueError("Image media type must start with 'image/'")


@dataclass(slots=True)
class ToolCall:
    """A complete model request to invoke a named tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class Message:
    """Common base for user, internal agent, assistant, system, and tool roles."""

    role: ClassVar[MessageRole]


@dataclass(slots=True, kw_only=True)
class SystemMessage(Message):
    """Application instructions."""

    role: ClassVar[MessageRole] = MessageRole.SYSTEM
    content: str


@dataclass(slots=True, kw_only=True)
class UserMessage(Message):
    """User text or ordered text/image blocks."""

    role: ClassVar[MessageRole] = MessageRole.USER
    content: UserContent

    @property
    def parts(self) -> list[UserContentPart]:
        """Expand the text shorthand into content blocks."""
        return [TextContent(self.content)] if isinstance(self.content, str) else self.content

    @property
    def text(self) -> str:
        """Extract text without including image payloads."""
        return "\n".join(part.text for part in self.parts if isinstance(part, TextContent))


@dataclass(slots=True, kw_only=True)
class AssistantMessage(Message):
    """Model text, reasoning, and complete tool calls."""

    role: ClassVar[MessageRole] = MessageRole.ASSISTANT
    content: str = ""
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    # Identify the provider that owns signed replay blocks.
    provider: str | None = None
    # Signed thinking blocks must be returned unchanged during tool round trips.
    replay_blocks: tuple[Mapping[str, Any], ...] = ()


@dataclass(slots=True, kw_only=True)
class ToolMessage(Message):
    """One tool result returned to the model."""

    role: ClassVar[MessageRole] = MessageRole.TOOL
    tool_call_id: str
    name: str
    content: str
    success: bool = True


@dataclass(slots=True, kw_only=True)
class AgentMessage(Message):
    """Internal extension input, persisted as agent and sent as user to providers.

    This is an instruction to the running agent, not model-generated output.
    Provider APIs have no agent role; adapters map it to user at the API boundary.
    """

    role: ClassVar[MessageRole] = MessageRole.AGENT
    content: str


type AnyMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage | AgentMessage
