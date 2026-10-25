"""Round trips against a real MySQL and PostgreSQL, when a URL is provided.

Set ``ZETT_TEST_MYSQL_URL`` and/or ``ZETT_TEST_POSTGRES_URL`` to run these; they skip
otherwise, so the default suite needs no database server. Example::

    ZETT_TEST_POSTGRES_URL=postgresql+psycopg://agent:secret@127.0.0.1:5432/zett \\
    ZETT_TEST_MYSQL_URL=mysql+aiomysql://agent:secret@127.0.0.1:3306/zett \\
    uv run pytest tests/test_sql_storage_backends.py
"""

from __future__ import annotations

import os

import pytest

from zett_agent.messages import AssistantMessage, UserMessage
from zett_agent.storage import SQLSessionStorage

URLS = {
    "mysql": os.environ.get("ZETT_TEST_MYSQL_URL"),
    "postgresql": os.environ.get("ZETT_TEST_POSTGRES_URL"),
}


@pytest.mark.parametrize("dialect", sorted(URLS))
async def test_a_session_round_trips_through_the_server(dialect):
    url = URLS[dialect]
    if not url:
        pytest.skip(f"set ZETT_TEST_{dialect.upper()}_URL to run this")
    storage = SQLSessionStorage(url)
    try:
        session = await storage.create_session(title="integration", agent_name="tester", session_type=3)
        assert session.session_type == 3

        first = await storage.append(
            session.session_id, "req-1", UserMessage(content="hello from " + dialect), tags={"suite": "integration"}
        )
        second = await storage.append(session.session_id, "req-2", AssistantMessage(content="hello back"))
        assert (first, second) == (1, 2)

        view = await storage.load(session.session_id)
        assert [record.message.content for record in view.raw_tail] == [
            "hello from " + dialect,
            "hello back",
        ]
        assert view.raw_tail[0].tags == {"suite": "integration"}
        assert await storage.count_messages(session.session_id) == 2

        page = await storage.list_raw_messages(session.session_id, after_sequence=1, limit=10)
        assert [record.sequence for record in page] == [2]

        assert await storage.delete_session(session.session_id) is True
        assert await storage.get_session(session.session_id) is None
    finally:
        await storage.close()


@pytest.mark.parametrize("dialect", sorted(URLS))
async def test_a_message_body_larger_than_a_mysql_text_column_round_trips(dialect):
    """MySQL ``TEXT`` stops at 64 KiB; the schema must use ``LONGTEXT`` for bodies."""
    url = URLS[dialect]
    if not url:
        pytest.skip(f"set ZETT_TEST_{dialect.upper()}_URL to run this")
    body = "x" * (200 * 1024)
    storage = SQLSessionStorage(url)
    session_id: str | None = None
    try:
        session = await storage.create_session(title="large body")
        session_id = session.session_id
        await storage.append(session_id, "req-large", UserMessage(content=body))
        view = await storage.load(session_id)
        assert view.raw_tail[0].message.content == body
    finally:
        if session_id is not None:
            await storage.delete_session(session_id)
        await storage.close()
