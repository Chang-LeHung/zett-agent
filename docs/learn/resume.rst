Resume an interrupted session
=============================

A request can end without finishing. The connection drops and the retries run out,
the user presses stop, the process is killed. The conversation is saved up to that
point, and you want to carry on from there instead of typing "continue" and hoping
the model works out what was missing.

``resume()`` is :meth:`~zett_agent.agent.Agent.run` without a new user message. The
session is restored, its last message says how it was interrupted, and the request
picks up from there.

.. code-block:: python

   try:
       reply = await agent.run("Refactor the parser")
   except ProviderTimeoutError:
       reply = await agent.resume()

Ask before you resume
---------------------

Nothing records that a request did not finish: a process that was killed cannot write
anything. What is left is the history, and its last message is the record. A finished
request ends in an assistant answer that asked for nothing and stopped normally;
every other ending is an interruption.

:meth:`~zett_agent.agent.Agent.resume_point` reads the session and says which, without
changing it. It writes nothing and calls no model.

.. code-block:: python

   point = await agent.resume_point()
   if point is not None:
       show_button("Resume", tooltip=point.description)

It returns ``None`` when there is nothing to resume. ``resume()`` raises
:class:`~zett_agent.exceptions.AgentResumeError` for those cases, with a ``reason``,
so you can also just try it.

What resuming does
------------------

The last message decides. ``kind`` on the :class:`~zett_agent.resume.ResumePoint` says
which row applies.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - History ends in
     - How it got there
     - What resuming does
   * - A user or internal message
     - The model never answered: the connection failed, or the request was cancelled.
     - ``CALL_MODEL``. Calls the model. Nothing is added.
   * - Tool results, all there
     - The tools finished and the next model call failed. Also a cancel during a tool,
       which already answered the call as cancelled.
     - ``CALL_MODEL``. Calls the model.
   * - A step the provider paused
     - The provider handed control back and the pause limit was reached.
     - ``CALL_MODEL``. Calls the model again.
   * - Tool calls with no result
     - The process stopped while tools ran.
     - ``RECOVER_TOOLS``. Adds a result for each missing call saying it was interrupted,
       then calls the model. See below.
   * - A response that stopped as ``INTERRUPTED``
     - The stream was cut by a failure.
     - ``RETRY_RESPONSE``. Keeps the partial response, adds a message telling the model it
       was interrupted and to start that response again, then calls the model.
   * - A response that stopped as ``MAX_OUTPUT``
     - It hit the output limit.
     - ``CONTINUE_RESPONSE``. Keeps the response, adds a message telling the model to
       continue exactly where it stopped, then calls the model.
   * - A normal answer
     - The request finished.
     - Nothing to resume. Raises ``AgentResumeError("completed")``.
   * - A refusal
     - The model declined.
     - Raises ``"refused"``. The same input would be declined again.
   * - A response that stopped as ``CONTEXT_FULL``
     - The context window was full.
     - Raises ``"context_full"``. Sending the same history fills the window again, so
       this is left to you: compact the session or start a shorter one.
   * - Any other stop
     - A value this library does not know.
     - Raises ``"unknown_stop"``.

An empty session raises ``"empty"``, and a history that ends in a tool result that
answers no call raises ``"malformed"``.

Tools are never run again on their own
--------------------------------------

A crash can leave an assistant message that asked for two tool calls with a result
for one. The other may have run, may have run halfway, or may never have started, and
nothing says which. A tool that writes a file or sends a message may already have done
so. Running it again could do it twice.

So the missing call is answered, not repeated. The model is given a failed result:

   The process stopped while this tool was running, so it may or may not have completed,
   and whatever it changes may or may not have happened. Check the current state before
   running it again.

and decides what to do, which is usually to look at the current state and then repeat
the call if it is needed. The result that did arrive is kept, so the model knows the
first tool ran.

That is the same rule that applies when a new message arrives instead of a resume.
Before the new input is appended, any tool call without a result is closed this way,
because a provider rejects a history that has one. Restoring a session used to drop the
whole interrupted batch, including the results that had arrived, so the model forgot
that those tools had run. It now keeps them.

:download:`Download resume.py <../_examples/resume.py>` for a complete, offline
program that resumes after a failure and after a crash, and checks that no tool ran
twice.

.. literalinclude:: ../_examples/resume.py
   :language: python
   :pyobject: main

What to pass again
------------------

Nothing about the interrupted request is stored except its messages. Pass the model,
tools, extensions and settings the interrupted request had, as you would for ``run``.
You may pass a different model, for example when a provider keeps failing. Resuming does
not change whether that model accepts a history that another provider wrote.

Streaming, callbacks and the synchronous agent
----------------------------------------------

``stream_resume()`` yields the same events as ``stream()``, and ``AgentClient`` and
``SyncAgent`` have ``resume``, ``stream_resume`` and ``resume_point`` too. The events
that follow are an ordinary request's: a resumed request ends in ``RUN_COMPLETED``, and
an abnormal stop in the resumed step is reported through ``MODEL_STOPPED`` like any
other, so a response that is cut off again can be resumed again.

What is different:

* There is no new user message, so the ``on_message`` hook is not called and
  ``context.input_message`` is ``None``. The ``before_run`` and ``after_run`` hooks run
  as for any request. The JSONL extension starts the resumed turn at the last user
  message.
* A goal that was armed for the interrupted request is not armed again.
* A session with nothing to resume raises as the request fails, and the error hooks run
  like for any failed request. Ask ``resume_point()`` first to avoid that.

What this does not do
---------------------

* **It does not recover the text that was streaming when the request stopped.** Partial
  output is saved only once the model finishes the step, so what a user saw on screen
  and the model had not completed is lost. After a cut-off response the model is told to
  start it again.
* **It does not retry for you.** The runtime retries a failed request before any output
  arrives; ``resume()`` is for when that is over, and it is the host that calls it.
* **It does not stop two processes resuming the same session.** The check that rejects a
  second request on a session works inside one process.
* **It does not know why the request stopped.** The history says what it was doing, not
  whether the cause was a timeout, a stop button or a crash, so a host that wants to show
  the cause keeps it itself.

Next: :doc:`stop-reasons` for how a response's ending is reported, and :doc:`sessions`
for how history is saved and restored.

Related API: :class:`~zett_agent.resume.ResumePoint`,
:class:`~zett_agent.resume.ResumeKind`,
:class:`~zett_agent.exceptions.AgentResumeError`.
