Install Zett Agent
========================

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install zett-agent

   .. tab-item:: uv

      .. code-block:: bash

         uv add zett-agent

Zett Agent requires Python 3.10 or newer; continuous integration covers 3.10
through 3.14. The library needs no Node.js and no external service to run. Every
built-in provider is a package dependency, so there are no extras to install.

Verify the install
------------------

.. code-block:: bash

   python -c "import zett_agent; print(zett_agent.__version__)"

Run something immediately
-------------------------

:doc:`first-agent` is a complete program with a deterministic model, so it needs
no API key and makes no network request. It runs the real loop — messages,
streaming and session history — with a scripted model in place of a provider.
Try :doc:`../examples/streaming-tools` next for a complete tool round trip.

Work from a checkout
--------------------

To read and modify the source, clone the repository and let ``uv`` create the
environment:

.. code-block:: bash

   git clone https://github.com/Chang-LeHung/zett-agent
   cd zett-agent
   uv sync
   uv run python docs/_examples/first_agent.py

Use ``uv run python your_script.py`` to run code against the checkout instead of
an installed release.

.. note::

   Providers and storage you create are yours to close. Call
   ``await model.aclose()`` and ``await storage.close()`` when finished.

Next: :doc:`first-agent`.
