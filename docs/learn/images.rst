Send images to a model
======================

Pass a :class:`~zett_agent.messages.UserMessage` when a prompt includes images.
Use an ordered list of ``TextContent`` and ``ImageContent`` parts instead of
vendor-specific dictionaries. A plain string remains the shorthand for a
text-only request.

Choose a vision-capable model before using this guide. The runtime represents
images, but it does not add vision support to a text-only endpoint.

Use image bytes
---------------

.. code-block:: python

   from pathlib import Path

   from zett_agent.messages import (
       ImageBytesSource,
       ImageContent,
       ImageDetail,
       TextContent,
       UserMessage,
   )

   message = UserMessage(
       content=[
           TextContent("Describe the layout of this screenshot."),
           ImageContent(
               source=ImageBytesSource(
                   data=Path("screenshot.png").read_bytes(),
                   media_type="image/png",
               ),
               detail=ImageDetail.HIGH,
           ),
       ],
   )
   reply = await client.run(message)
   print(reply.content)

Run this inside an async function with a client from :doc:`providers` and an
existing image. ``data`` is an encoded image file's bytes, **not** decoded
pixels or a base64 string. The MIME type must describe the file, such as
``image/png`` or ``image/jpeg``. The runtime checks that bytes are nonempty and
the MIME type starts with ``image/``; it does not decode or validate the image.

For large uploads in an async service, read files outside the request loop or
use a worker thread. Validate upload sizes and allowed formats in your
application before constructing the message.

Use a URL
---------

.. code-block:: python

   from zett_agent.messages import ImageUrlSource

   message = UserMessage(
       content=[
           TextContent("What is shown in this image?"),
           ImageContent(source=ImageUrlSource("https://example.com/image.png")),
       ],
   )

Replace the URL with an image your provider can access. The runtime does not
download it for you or attach your application's authentication headers.
``ImageUrlSource`` accepts HTTP, HTTPS, and image data URLs; it does not accept
a local file path. Use bytes for local files or private images.

Provider differences
--------------------

.. list-table:: Image representation supported by the adapters
   :header-rows: 1
   :widths: 29 71

   * - Adapter
     - What to consider
   * - OpenAI / DeepSeek
     - Accept typed bytes or URLs and render them for the selected protocol. Vision depends on the endpoint and model.
   * - Anthropic
     - Bytes become base64 image blocks; HTTP(S) URLs and image data URLs are represented separately.
   * - Google
     - Bytes become inline image parts; URLs become URI parts. The chosen service must support the URI.
   * - Ollama
     - Use bytes or base64 image data URLs. Remote HTTP(S) image URLs are rejected by the adapter.

``ImageDetail.AUTO`` is the default. ``LOW`` and ``HIGH`` are fidelity hints,
not universal resolution guarantees; only protocols that expose such a setting
can use them. ``alt_text`` stores an optional description on the image object,
but do not rely on it as model-visible text: add a ``TextContent`` part for
instructions or accessibility descriptions the model needs.

Inspect and store multimodal messages
------------------------------------------------------------

``message.parts`` preserves the content parts in order. ``message.text`` joins
only the text parts, which is useful for display labels but is not a complete
transcript. Built-in SQLite storage preserves typed image content. Images can
therefore appear in later requests when history is restored; plan for storage
size, retention, and provider costs accordingly.

Local tools can also return ``ImageContent`` or a list of typed text/image
parts. See :doc:`tools` for result types. MCP content follows its own result
mapping; do not assume every MCP image block becomes a vision input.

Verify the message offline
--------------------------

.. literalinclude:: ../_examples/images.py
   :language: python
   :pyobject: main

Run ``uv run python docs/_examples/images.py`` from a checkout. Expected output:

.. code-block:: text

   Received text and image in order.

The :download:`complete example <../_examples/images.py>` embeds a tiny image
and verifies the typed request with a deterministic model. It tests message
handling, not an LLM's ability to interpret the image.

Next: :doc:`streaming` to display the response, or :doc:`sessions` to decide
what to retain.
