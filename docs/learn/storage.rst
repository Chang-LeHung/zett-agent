Store sessions in a database
============================

A session is the conversation a later request restores: the original messages, the
latest compaction checkpoint, and a little display metadata. Where that lives is an
application decision, and the runtime ships one schema for SQLite, MySQL, and
PostgreSQL.

Choose a backend
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Backend
     - Use it when
     - What it costs
   * - SQLite
     - One process owns the file, or a desktop app keeps everything local.
     - No server; WAL lets a second process read and write, but a network
       filesystem is not supported.
   * - MySQL
     - Several application processes share conversations and the team already
       runs MySQL.
     - A server, a driver extra, credentials, and ``CREATE`` on the schema the
       first time.
   * - PostgreSQL
     - The same, on PostgreSQL.
     - The same.

The three share one set of tables and one behavior: appends allocate a
per-session sequence, snapshots advance a version and are rejected when stale,
and deleting a session removes its messages and checkpoints.

Install the driver
------------------

SQLite needs nothing beyond the runtime. A server dialect needs its async driver:

.. code-block:: shell

   pip install "zett-agent[mysql]"      # aiomysql
   pip install "zett-agent[postgres]"   # psycopg

Connecting without a URL
------------------------

Pass the parts; the storage builds the URL, fills in the dialect's default port,
and keeps credentials out of the string you log:

.. code-block:: python

   from zett_agent.storage import SQLSessionStorage

   sqlite = SQLSessionStorage("sessions.sqlite3")  # a plain path is a SQLite file
   local = SQLSessionStorage.for_sqlite()  # ~/.zett-agent/sessions.sqlite3

   mysql = SQLSessionStorage.for_mysql(
       host="db.internal",
       user="agent",
       password=os.environ["MYSQL_PASSWORD"],
       database="sessions",
   )
   postgres = SQLSessionStorage.for_postgres(
       host="db.internal",
       user="agent",
       password=os.environ["PGPASSWORD"],
       database="sessions",
   )

A URL works too, and covers anything SQLAlchemy supports:

.. code-block:: python

   storage = SQLSessionStorage("postgresql+psycopg://agent:secret@db.internal:5432/sessions")
   # or, to inspect what would be used, without connecting:
   url = SQLSessionStorage.build_url(driver="postgresql", user="agent", password="secret", database="sessions")

Persist with the extension
--------------------------

.. code-block:: python

   from zett_agent.extensions.sql import SQLSessionExtension
   from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

   persistence = SQLSessionExtension(storage)
   client = await create_agent(
       model,
       config=AgentRunConfig(session_id="project-review"),
       extensions=[persistence, ToolGuidelinesExtension()],
   )

An explicit extension list replaces the Agent defaults, so add a history restorer such
as this one, or session persistence, once. Close the extension when the application
stops; it closes the storage, which releases the pool.

What the storage creates
------------------------

The first awaited call creates three tables in the schema the connection names:
``agent_sessions``, ``raw_messages``, and ``session_snapshots``, with their indexes.
Nothing is created by the constructor, so a storage can be built before an event loop
exists. After that first call, the database user only needs ``SELECT``, ``INSERT``,
``UPDATE``, and ``DELETE`` on those tables.

Two upgrades also run there, on any dialect: a missing ``session_type`` column is
added with ``ALTER TABLE``, and index definitions that an older release did not create
are added one by one. Concurrent first connections are safe because ``CREATE TABLE``
and the index check are idempotent on all three dialects.

Pooling and lifetime
--------------------

SQLite uses one connection per event loop instead of a pool, because a pooled SQLite
connection cannot cross loops; WAL mode and a five-second busy timeout keep a second
process from blocking the first. MySQL and PostgreSQL use a pooled connection with a
pre-ping before reuse:

.. code-block:: python

   postgres = SQLSessionStorage.for_postgres(
       user="agent",
       password=os.environ["PGPASSWORD"],
       database="sessions",
       pool_size=5,  # connections kept per process
       max_overflow=10,  # opened beyond the pool under load
       pool_recycle=1800,  # reconnect before a server idle timeout
   )

Keep credentials out of source
------------------------------

Read the password from the environment or a secret manager. A SQLAlchemy URL keeps the
password in clear text, so do not log ``storage.url`` or a URL you built from one; log
``storage.dialect`` and the host instead. Give the application's database user rights
on its own schema only.

Write your own storage
----------------------

A different database, an object store, or an encrypted file needs only three
operations. :doc:`../extending/storage-adapter` builds one from scratch, with the
invariants each operation must keep.

Related API: :class:`~zett_agent.storage.SQLSessionStorage`,
:class:`~zett_agent.extensions.sql.SQLSessionExtension`,
:class:`~zett_agent.extensions.persistence.SessionStorage`.
