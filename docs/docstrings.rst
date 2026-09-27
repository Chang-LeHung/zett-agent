Writing API documentation
=========================

Source of truth
---------------

Public objects are discovered from the modules that define them: the reference
generator walks the runtime package, keeps every public name a module defines
itself, creates one page per name, and groups pages by responsibility. Do not
edit ``docs/_generated``: rebuilds regenerate these files. Maintain API
descriptions in source docstrings and tutorials in hand-written RST pages.

Sphinx, reStructuredText, and docstring style
----------------------------------------------------

These are different layers. Sphinx builds and cross-links the website.
reStructuredText (RST) is the markup used by the handwritten ``.rst`` pages.
Google style structures the Python docstrings; the Napoleon extension converts
those sections into Sphinx markup. The PyData theme controls presentation.

NumPy-style docstrings are another input convention, not a prerequisite for a
NumPy-like website. This project consistently uses Google-style sections rather
than mixing conventions within a class.

Useful RST constructs::

    A section heading
    =================

    :class:`~zett_agent.agent.Agent`
    :meth:`~zett_agent.agent.Agent.stream`
    :doc:`extending/first-extension`

    .. code-block:: python

       reply = await client.run("Hello")

    .. literalinclude:: _examples/first_agent.py
       :language: python
       :pyobject: main

Keep a blank line before directive contents and indent their options and body.
Use literalinclude for tested examples so the website and downloadable program
cannot silently diverge. Cross-reference roles resolve symbols and generate links;
plain backticks alone do not create an API reference.

Diagrams
--------

Use Mermaid for process, state, and sequence diagrams that benefit from real
nodes and connecting lines. Hand-written documentation can use the directive
directly, and Sphinx renders it as a sharp SVG at different viewport sizes::

    .. mermaid::
       :caption: The model-tool loop

       flowchart LR
          user[UserMessage] --> model[model]
          model -- ToolCall --> tool[tool]
          tool -- ToolMessage --> model
          model --> assistant[AssistantMessage]

Public API docstrings use ``.. zett-diagram:: name`` followed by an ASCII version.
The source therefore remains useful in an editor, ``help()``, and a terminal;
the local ``diagrams`` Sphinx extension replaces that block with the registered
Mermaid diagram only while building the website. Update both representations
when their behavior changes.

Keep ``.. code-block:: text`` for output whose exact characters are the subject
of the documentation, such as a terminal transcript, wire format, or ASCII
table. Larger architectural diagrams belong in a hand-written guide.

Google-style docstrings
-----------------------

Use a short imperative summary, optional behavioral details, then relevant
sections. Type hints supply types; docstrings explain meaning, units, defaults,
ownership, and failure conditions.

.. code-block:: python

   async def run(message: str) -> AssistantMessage:
       """Process a request and return its final answer.

       Args:
           message: Current user input, not the complete conversation history.

       Returns:
           Final assistant output after stream cleanup.

       Raises:
           AgentProtocolError: If the runtime is not initialized.

       Examples:
           Send one message::

               reply = await agent.run("Hello")
               print(reply.content)
       """

Use ``Yields`` for async generators, ``Attributes`` for model fields, and ``Note``
or ``Warning`` only where they clarify a real constraint. Omit irrelevant
sections. A code-only example uses an indented literal block after ``::``;
plain Python under an Examples heading alone may be parsed as prose.

Field comments
--------------

.. code-block:: python

   @dataclass
   class Usage:
       """Normalized provider usage."""

       #: Input tokens, including the subset read from cache.
       input_tokens: int = 0

``#:`` makes a source-adjacent field description available to autodoc. Avoid
duplicating a description in both Attributes and a field comment. Document time
units, UTC versus monotonic clocks, nullable values, and subset relationships.

Tools have an additional contract
---------------------------------

``Snippet`` and ``Guidelines`` are consumed by the tool decorator and included in
model guidance. They are not merely website formatting. Preserve their parser
syntax and validate tool-schema tests after editing them. Examples intended for
developers belong under Examples rather than in model-facing guidance.

Build and validate
------------------

From the repository root::

    make docs
    make docs-serve

The build treats documentation warnings as failures. The docs tests check that
every public export has a generated page and that links, example blocks, and
parameter descriptions appear in the built output. Normal builds import the
package but do not instantiate providers, execute examples, or touch session data.
