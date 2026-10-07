from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ._compat import StrEnum, TypeAliasType


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

    #: Text at this position in a multimodal content list.
    text: str


@dataclass(frozen=True, slots=True)
class ImageUrlSource:
    """Remote or data URL containing an image."""

    #: Non-empty HTTP, HTTPS, or image data URL.
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

    #: Encoded image file bytes, not decoded pixels or base64 text.
    data: bytes
    #: Image MIME type, such as image/png or image/jpeg.
    media_type: str

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("Image data cannot be empty")
        _validate_image_media_type(self.media_type)


ImageSource = TypeAliasType("ImageSource", ImageUrlSource | ImageBytesSource)


@dataclass(frozen=True, slots=True)
class ImageContent:
    """One image block in an ordered multimodal message."""

    #: Remote URL or local in-memory encoded image bytes.
    source: ImageSource
    #: Requested fidelity, mapped according to provider capabilities.
    detail: ImageDetail = ImageDetail.AUTO
    #: Optional textual description of the image.
    alt_text: str | None = None


UserContentPart = TypeAliasType("UserContentPart", TextContent | ImageContent)
UserContent = TypeAliasType("UserContent", str | list[UserContentPart])


def _validate_image_media_type(media_type: str) -> None:
    if not media_type.strip().lower().startswith("image/"):
        raise ValueError("Image media type must start with 'image/'")


@dataclass(slots=True)
class ToolCall:
    """A complete model request to invoke a named tool.

    Unlike :class:`ToolCallDelta`, ``arguments`` must already be a complete
    mapping suitable for validation and execution.

    Examples:
        Construct the normalized result of a provider tool call::

            call = ToolCall(
                id="call-7",
                name="read_file",
                arguments={"path": "README.md", "start_line": 1},
            )
    """

    #: Invocation identifier used to correlate a ToolMessage result.
    id: str
    #: Registered tool name selected by the model.
    name: str
    #: Complete parsed keyword arguments, not a partial JSON delta.
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class Message:
    """Common base for user, internal agent, assistant, system, and tool roles.

    ``attributes`` carries application-owned information attached to one message.
    Provider adapters ignore it unless they explicitly document another mapping.
    ``persist`` controls whether Raw Log subscribers should store the message.
    ``include_in_messages`` controls whether persisted history is restored into
    ``AgentState.messages`` and therefore sent on later model requests.

    .. note::
        Request-level ``metadata`` and ``tags`` live on :class:`AgentRunContext` and
        are persisted beside Raw Log records. They are intentionally different
        from per-message ``attributes``.

    .. seealso::
        :class:`~zett_agent.extensions.persistence.RawMessageRecord` is the persisted message envelope.
    """

    role: ClassVar[MessageRole]
    #: Application-owned per-message information. Provider adapters do not send
    #: these values to model APIs unless an adapter explicitly defines a mapping.
    attributes: dict[str, Any] = field(default_factory=dict)
    #: Whether the runtime's persistence subscribers should store this message.
    persist: bool = True
    #: Whether restored history should include this message in model context.
    include_in_messages: bool = True


@dataclass(slots=True, kw_only=True)
class SystemMessage(Message):
    """Application instructions placed before conversational messages.

    System messages are recorded in the Raw Log for observability but are not
    restored into later model context by default. Applications rebuild current
    instructions for every request, so persisting the same instruction must not
    accumulate stale copies in ``AgentState.messages``. Set
    ``include_in_messages=True`` when a system message should also be replayed.

    Examples:
        Insert durable behavior during an extension's ``on_state`` hook::

            context.add_message(
                SystemMessage(content="Answer with concise, verifiable steps."),
                index=0,
            )
    """

    role: ClassVar[MessageRole] = MessageRole.SYSTEM
    content: str
    persist: bool = True
    include_in_messages: bool = False


@dataclass(slots=True, kw_only=True)
class UserMessage(Message):
    """User text or ordered text/image blocks.

    Examples:
        Compose a multimodal request::

            message = UserMessage(content=[
                TextContent("Describe this image."),
                ImageContent(ImageUrlSource("https://example.com/photo.png")),
            ])

    Note:
        Image support depends on the selected model. The text property excludes
        image payloads; parts preserves their original order.

    .. seealso::
        :class:`~zett_agent.messages.TextContent`, :class:`~zett_agent.messages.ImageContent`, and
        :class:`~zett_agent.messages.ImageSource` define supported content parts.
    """

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
    """Model text, reasoning, and complete tool calls.

    Attributes:
        content: Final answer text; may be empty for tool-only responses.
        reasoning: Provider-reported reasoning, or None when unavailable.
        tool_calls: Complete invocations, separate from streamed argument deltas.

    .. note::
        ``replay_blocks`` are opaque transport state for the same provider and
        model. They are not portable reasoning text and should not be displayed
        as part of ``content``.

    Examples:
        A tool-only response may have no answer text::

            message = AssistantMessage(
                tool_calls=(ToolCall("call-1", "grep", {"pattern": "TODO"}),),
                provider="anthropic",
                model="claude-example",
            )

    .. seealso::
        :class:`~zett_agent.model.ModelResponse` carries this message with normalized
        usage, and :class:`~zett_agent.messages.ToolMessage` returns tool observations.
    """

    role: ClassVar[MessageRole] = MessageRole.ASSISTANT
    content: str = ""
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    #: Provider namespace for opaque replay data; prevents cross-provider reuse.
    provider: str | None = None
    #: Provider model identifier that produced this response. Together with
    #: provider, it prevents signed replay data from crossing model boundaries.
    model: str | None = None
    #: Opaque provider-authenticated reasoning/tool blocks that must be persisted
    #: and returned structurally unchanged in the next same-provider tool round
    #: trip; adapters may still decode and encode their transport representation.
    #: They are transport state, not user-visible reasoning or portable content.
    replay_blocks: tuple[Mapping[str, Any], ...] = ()


@dataclass(slots=True, kw_only=True)
class ToolMessage(Message):
    """One tool result returned to the model.

    Attributes:
        tool_call_id: Identifier of the assistant call answered by this message.
        name: Invoked tool name.
        content: Serialized result, failure text, or ordered text/image content blocks.
        success: Whether execution succeeded; false also covers skipped calls.
    """

    role: ClassVar[MessageRole] = MessageRole.TOOL
    tool_call_id: str
    name: str
    content: UserContent
    success: bool = True

    @property
    def text(self) -> str:
        """Plain text for search indexes; encoded image data stays in content."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content if isinstance(part, TextContent))


@dataclass(slots=True, kw_only=True)
class AgentMessage(Message):
    """Internal extension input, persisted as agent and sent as user to providers.

    This is an instruction to the running agent, not model-generated output.
    Provider APIs have no agent role; adapters map it to user at the API boundary.

    Examples:
        An extension can request another model iteration without impersonating
        the user::

            await context.publish(
                InternalMessageEvent(AgentMessage(content="Run the failing test and continue."))
            )

    .. seealso::
        :class:`~zett_agent.extensions.internal_message.InternalMessageExtension` owns the request-local
        queue and processing limit for these messages.
    """

    role: ClassVar[MessageRole] = MessageRole.AGENT
    content: str


AnyMessage = TypeAliasType("AnyMessage", SystemMessage | UserMessage | AssistantMessage | ToolMessage | AgentMessage)
