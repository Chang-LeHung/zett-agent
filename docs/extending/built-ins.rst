Compose built-in extensions
===============================

Start with the capability you need, then choose explicit dependencies. Passing
an extension list replaces optional defaults; InternalMessageExtension and
SteeringExtension remain Agent-owned built-ins and must not be registered again.

.. list-table:: Capability map
   :header-rows: 1
   :widths: 27 40 33

   * - Extension
     - Use when
     - Important constraint
   * - InMemoryMessageAccumulator
     - Retain conversation history inside one process.
     - Share the instance to share its memory; not durable.
   * - SQLiteSessionExtension
     - Restore raw history and compaction checkpoints across restarts.
     - Close the owned pool; use one history restorer.
   * - ToolGuidelinesExtension
     - Include tool snippets and guidelines in model instructions.
     - Include explicitly when supplying a custom extension list.
   * - FileSystemExtension
     - Read/search or edit the current working directory.
     - read_only limits registered tools, not OS permissions.
   * - CodingExtension
     - Add writable filesystem tools and shell execution.
     - Run only with a suitable sandbox/trust policy.
   * - AskUserExtension
     - Let the model ask the UI a question and wait.
     - Route replies with session ID and tool-call ID.
   * - PlanModeExtension
     - Model-selected planning entry/exit with user approval events.
     - UI must implement its external-response protocol.
   * - TodoWriteExtension
     - Track ordered tasks within the current request.
     - Only one processing item; lists are cleared at request end.
   * - CompactionExtension
     - Summarize older complete turns before a model step.
     - Token estimates and model context limits must be configured appropriately.
   * - GoalExtension
     - Continue explicit /goal work after private evaluation.
     - Bounded internal continuations and decision retries.
   * - SubAgentExtension
     - Delegate to configured child profiles.
     - Models, tools, and persistence live in each definition.
   * - SkillExtension
     - Discover skill descriptions and lazily load complete instructions.
     - Treat skill files as external input, not authorization.
   * - McpExtension
     - Register remote or subprocess MCP tools.
     - Trust and resource ownership remain application responsibilities.

Try the examples rather than enabling everything
----------------------------------------------------

Start with :doc:`../examples/streaming-tools` for tools,
:doc:`../examples/sessions` for durability, :doc:`../examples/approval` for user
interaction, and :doc:`../examples/compaction` for context management. The
:doc:`subagent <../examples/subagent>` and :doc:`goal <../examples/goal>` examples
show two distinct delegation patterns.

Browse :doc:`../_generated/group-extensions` for constructor signatures, event
payload models, hook implementations, and links to source.
