Know why the model stopped
==========================

A request ends when the model has nothing more to ask for. That is not the same as the
model *finishing*. It may have refused, run into its output limit, filled the context
window, or lost its connection halfway through a sentence. Providers report each of
these differently, and a program that only looks for tool calls sees all of them as
"the model answered".

Zett Agent reads every provider's report into one vocabulary and puts it where a host
can see it, instead of dropping it.

Check how it ended
----------------------------------------

An abnormal stop is on the message the run returns:

.. code-block:: python

   message = await agent.run("Summarize this repository")

   if message.stop is not None:
       # Not a normal ending: the text may be missing, partial, or a refusal.
       print(message.stop.reason, message.stop.category, message.stop.message)

``message.stop`` is ``None`` when the model finished or asked for tools, which includes
every message stored before this existed. When it is set it is a
:class:`~zett_agent.stop.StopInfo`:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``reason``
     - The normalized :class:`~zett_agent.stop.StopReason`. Branch on this.
   * - ``raw``
     - The provider's own value, unchanged, such as ``"content_filter"``.
   * - ``category``
     - What kind: the provider's policy category (``"cyber"``, ``"SAFETY"``) or, if it has none, the raw value.
   * - ``message``
     - A sentence for a person: the refusal text, or the provider's explanation. Show it; do not parse it.
   * - ``recommended_model``
     - A model the provider suggests retrying on, when it does.
   * - ``detail``
     - Provider-specific facts with no normalized home, such as an OpenRouter ``native_finish_reason`` or the names of tool calls that were withheld.

Scripts can ask to be told instead of checking:

.. code-block:: python

   message = (await agent.run("...")).raise_for_stop()   # raises AgentStopError

The runtime never raises it for you. A refusal is a normal outcome, so a host that does
not look still gets the conversation back.

Hear about it as it happens
----------------------------------------

A streaming UI gets a ``MODEL_STOPPED`` event right after ``MODEL_COMPLETED``. It is sent
only for an abnormal stop, so there is nothing to filter:

.. code-block:: python

   async for event in client.stream("Summarize this repository"):
       if event.type == AgentEventType.MODEL_STOPPED:
           show_notice(event.stop.describe())    # e.g. "refusal (cyber): ..."

With an :class:`~zett_agent.dispatcher.AgentEventDispatcher`, override
``on_model_stopped_event``. An extension sees the same value as ``response.stop`` in
``after_model``.

``RUN_COMPLETED`` still follows. It means the request ran to its end, and a refusal
ends a request and completes it; ``MODEL_STOPPED`` is how you tell the two apart.

What each reason means
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Reason
     - Meaning
     - What the runtime does
   * - ``END_TURN``
     - The model finished, or hit a stop sequence you set.
     - Nothing; not reported.
   * - ``TOOL_USE``
     - The model asked for tools.
     - Runs them; not reported.
   * - ``MAX_OUTPUT``
     - The output limit was reached. Text, or a tool call's arguments, is cut off.
     - Reports it. A tool call with cut-off arguments is answered with an error, as before.
   * - ``CONTEXT_FULL``
     - The context window filled. The output is cut off.
     - Reports it.
   * - ``REFUSAL``
     - The model, or a safety system in front of it, declined.
     - **Withholds tool calls**, ends the request, reports it.
   * - ``INTERRUPTED``
     - Infrastructure ended the response: a provider error, a dropped stream, no capacity.
     - **Withholds tool calls**, ends the request, reports it.
   * - ``PAUSED``
     - The provider handed control back and wants the turn continued.
     - Continues, up to ``Agent.max_pause_continuations`` times (5); reports it only when that runs out.
   * - ``OTHER``
     - A value this library does not know. ``raw`` has it.
     - Reports it and otherwise carries on.

A new provider value never breaks a run: it becomes ``OTHER`` with the original kept.

Tool calls that come with a refusal are not run
-----------------------------------------------

A refusal can arrive after a complete tool call was already streamed, and an interrupted
stream may end in the middle of one. Running that call would act on output the model or
the provider disowned. So for ``REFUSAL`` and ``INTERRUPTED`` the calls are removed from
the stored message, the tool is never started, and the names of what was withheld are in
``stop.detail["withheld_tool_calls"]``. The provider's own record of the call goes too,
because a ``tool_use`` replayed without a result is rejected by the API.

:download:`Download stop_reasons.py <../_examples/stop_reasons.py>` for a complete,
offline program that streams a partial answer and a tool call, refuses, and shows that
the tool never ran.

.. literalinclude:: ../_examples/stop_reasons.py
   :language: python
   :pyobject: main

Where the reason comes from
----------------------------------------

A provider can say it in four ways, and each is read:

#. **A value** in ``finish_reason`` or ``stop_reason``. Every provider.
#. **A refusal in the content.** OpenAI's ``refusal`` field and content block. The text is
   kept and shown like any other answer, and it becomes ``stop.message``. This is the one
   that used to be lost: such a response finishes with ``stop`` and no text.
#. **A refusal before any output**, which arrives as an error rather than a stop: a blocked
   prompt on Google, a ``content_filter`` ``400`` on Kimi, a moderation or guardrail ``403``
   on OpenRouter, a policy failure on the OpenAI Responses API. These are answered as a
   ``REFUSAL`` with an empty message, so the conversation continues and the host is told
   why. Every other error status still raises as it did.
#. **An error inside a successful stream.** OpenRouter sends one after the ``200`` is
   committed. With output already written it ends the response as ``INTERRUPTED`` and
   keeps the text; with none, it is raised as an error, so the usual retry applies.

What each provider sends
----------------------------------------

These are the values each provider documents, and what they become. A value not listed is
``OTHER``.

.. list-table::
   :header-rows: 1
   :widths: 17 43 40

   * - Provider
     - Values
     - Refusals arrive as
   * - Anthropic
     - ``end_turn``, ``stop_sequence``, ``tool_use``, ``max_tokens``, ``pause_turn``, ``refusal``, ``model_context_window_exceeded``
     - ``stop_reason: "refusal"``, possibly mid-stream, with ``stop_details`` (``category``: ``cyber``, ``bio``, ``frontier_llm``, ``reasoning_extraction``, ``general_harms``; ``explanation``; ``recommended_model``)
   * - OpenAI Chat
     - ``stop``, ``length``, ``tool_calls``, ``content_filter``, ``function_call``
     - ``content_filter``, or the ``refusal`` field
   * - OpenAI Responses
     - ``completed``, or ``incomplete`` with ``max_output_tokens`` or ``content_filter``
     - A ``refusal`` content block, ``incomplete`` + ``content_filter``, or a failure with a policy ``error.code`` (``bio_policy``, ``image_content_policy_violation``)
   * - DeepSeek
     - ``stop``, ``length``, ``content_filter``, ``tool_calls``, ``insufficient_system_resource``, ``aborted``
     - ``content_filter``. ``length`` also covers an over-long conversation, so it is ``MAX_OUTPUT``.
   * - GLM (Z.ai)
     - ``stop``, ``tool_calls``, ``length``, ``sensitive``, ``model_context_window_exceeded``, ``network_error``
     - ``sensitive``
   * - Kimi (Moonshot)
     - ``stop``, ``length``, ``tool_calls``
     - ``HTTP 400`` with ``error.type: "content_filter"``
   * - OpenRouter
     - ``tool_calls``, ``stop``, ``length``, ``content_filter``, ``error``; ``native_finish_reason`` is the upstream value
     - ``content_filter``; a moderation ``403`` (``metadata.reasons``); a guardrail ``403`` (``metadata.patterns``)
   * - OpenCode Zen
     - A gateway: the values of the protocol you choose (Responses, Chat, Messages, or Gemini)
     - As that protocol
   * - Google
     - ``STOP``, ``MAX_TOKENS``, ``SAFETY``, ``RECITATION``, ``BLOCKLIST``, ``PROHIBITED_CONTENT``, ``SPII``, ``IMAGE_SAFETY``, ``IMAGE_PROHIBITED_CONTENT``, ``IMAGE_RECITATION``; others (``OTHER``, ``MALFORMED_FUNCTION_CALL``, ``LANGUAGE``, ...) are ``OTHER``
     - The finish reasons above, or ``prompt_feedback.block_reason`` when the prompt was blocked and there is no candidate
   * - Mistral
     - ``stop``, ``length``, ``model_length``, ``error``, ``tool_calls``
     - No dedicated value
   * - Qwen, Groq, xAI, Ollama
     - Qwen documents ``stop``, ``length`` and ``tool_calls``. Groq's and xAI's pages do not list the values. Ollama's ``done_reason`` is an open string.
     - Not documented by the provider

Gemini reports ``STOP`` whether or not it called a function, so a step that finished with
tool calls is ``TOOL_USE``. ``finishMessage`` is dropped by the Gemini API path of the
``google-genai`` SDK (only Vertex keeps it), so a Google refusal has a category but no
sentence.

Choose whether the model reads a refusal again
----------------------------------------------

By default a refused answer stays in the history and is sent again with every later
request, like any other assistant message. Anthropic advises treating output produced
before a refusal as incomplete and discarding it. Add
:class:`~zett_agent.extensions.refusal_replay.RefusalReplayExtension` to choose:

.. code-block:: python

   from zett_agent.extensions.refusal_replay import RefusalReplay, RefusalReplayExtension

   agent = await create_agent(model, extensions=[RefusalReplayExtension(replay=RefusalReplay.DROP)])

Only the request changes. The message stays in stored history and on screen, so changing
the setting never rewrites a session. Dropping a message can leave two user messages in a
row, and providers differ in whether they accept that, so test it against yours.

Subagents and goals
----------------------------------------

A subagent that was refused still returns a message, and its text can be empty or partial.
The parent model is told: the report starts with a notice, and ``SubAgentResult.stop``
carries the reason. A background task whose child stopped early is still ``completed``,
because the child ended, and says so in ``stop``, in its note, and in the completion notice.

A goal is not evaluated against a refused or interrupted answer. Asking the model to try
again would only meet the same refusal, so the request ends and the host hears why.

What this does not do
----------------------------------------

* **No retry on another model.** A refusal is reported, not worked around. Anthropic
  documents server-side and client-side fallback to a different model; acting on
  ``stop.recommended_model`` is left to the host.
* **No removal of a refused turn from the history** unless you choose it above.
* **No guarantee the provider says why.** A refusal with no category and no text is
  reported as a refusal with nothing to show.

Next: :doc:`streaming` for the full event list, or :doc:`providers` to connect one.

Related API: :class:`~zett_agent.stop.StopInfo`, :class:`~zett_agent.stop.StopReason`,
:class:`~zett_agent.exceptions.AgentStopError`,
:class:`~zett_agent.extensions.refusal_replay.RefusalReplayExtension`.
