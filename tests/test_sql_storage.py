"""The connection and schema shared by SQLite, MySQL, and PostgreSQL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable

from zett_agent import _sql_storage
from zett_agent._sql_storage import AgentSessionModel, Base, SQLiteSessionStorage, SQLSessionStorage


def actions(tmp_path: Path) -> Path:
    return tmp_path / "sessions.sqlite3"


# ------------------------------------------------------------------------ connections


def test_a_plain_string_is_a_sqlite_file(tmp_path):
    storage = SQLSessionStorage(str(actions(tmp_path)))
    assert storage.dialect == "sqlite"
    assert storage.path == actions(tmp_path)
    assert storage.url.database == str(actions(tmp_path))


def test_a_url_is_used_as_given():
    url = "postgresql+psycopg://agent:secret@db.example:5432/sessions"
    built = SQLSessionStorage.build_url(url)
    assert built.get_backend_name() == "postgresql"
    assert built.password == "secret" and built.host == "db.example"


def test_parameters_build_a_mysql_url():
    built = SQLSessionStorage.build_url(
        driver="mysql", user="agent", password="secret", database="sessions", host="db", port=3307
    )
    assert built.get_backend_name() == "mysql"
    assert (built.username, built.host, built.port, built.database) == ("agent", "db", 3307, "sessions")


def test_parameters_build_a_postgres_url_with_the_default_port():
    built = SQLSessionStorage.build_url(driver="postgresql", user="agent", password="secret", database="sessions")
    assert built.get_backend_name() == "postgresql" and built.port == 5432


def test_the_default_database_path_is_under_the_user_directory():
    storage = SQLSessionStorage(driver="sqlite")
    assert storage.path == Path.home() / ".zett-agent" / "sessions.sqlite3"


def test_an_in_memory_database_has_no_file():
    assert SQLSessionStorage("sqlite+aiosqlite:///:memory:").path is None


def test_a_server_dialect_needs_a_database():
    with pytest.raises(ValueError, match="database is required"):
        SQLSessionStorage(driver="mysql", user="agent", password="secret")


def test_an_unknown_driver_is_rejected():
    with pytest.raises(ValueError, match="driver must be one of"):
        SQLSessionStorage(driver="oracle")


@pytest.mark.parametrize(("driver", "extra"), [("mysql", "mysql"), ("postgresql", "postgres")])
def test_a_missing_driver_names_the_extra_to_install(monkeypatch, driver, extra):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError(f"No module named '{driver}'")

    monkeypatch.setattr(_sql_storage, "create_async_engine", missing)
    with pytest.raises(ImportError, match=rf"zett-agent\[{extra}\]"):
        SQLSessionStorage(driver=driver, user="agent", password="secret", database="sessions")


# ----------------------------------------------------------------------------- schema


@pytest.mark.parametrize("dialect", [sqlite.dialect(), mysql.dialect(), postgresql.dialect()])
def test_the_schema_compiles_for_every_dialect(dialect):
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert "CREATE TABLE" in ddl
    sessions = str(CreateTable(AgentSessionModel.__table__).compile(dialect=dialect))
    assert "session_type" in sessions and "VARCHAR(36)" in sessions


def test_mysql_stores_message_bodies_in_longtext():
    """MySQL ``TEXT`` stops at 64 KiB, which a long conversation turn exceeds."""
    ddl = str(CreateTable(Base.metadata.tables["raw_messages"]).compile(dialect=mysql.dialect()))
    assert "LONGTEXT" in ddl
    assert "message_json TEXT" not in ddl


async def test_a_fresh_sqlite_database_gets_the_tables_and_indexes(tmp_path):
    storage = SQLiteSessionStorage(actions(tmp_path))
    try:
        await storage.create_session(title="schema")
    finally:
        await storage.close()
    with sqlite3.connect(actions(tmp_path)) as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        indexes = {row[1] for row in connection.execute("pragma index_list(raw_messages)")}
    assert {"agent_sessions", "raw_messages", "session_snapshots"} <= tables
    assert "ux_raw_session_sequence" in indexes


async def test_an_old_database_without_session_type_is_migrated(tmp_path):
    with sqlite3.connect(actions(tmp_path)) as connection:
        connection.execute(
            "CREATE TABLE agent_sessions ("
            "id VARCHAR(36) PRIMARY KEY, parent_session_id VARCHAR(36), title VARCHAR(200), "
            "agent_name VARCHAR(64), created_at DATETIME, updated_at DATETIME)"
        )
        connection.commit()
    storage = SQLiteSessionStorage(actions(tmp_path))
    try:
        session = await storage.create_session(title="migrated", session_type=7)
        assert session.session_type == 7
    finally:
        await storage.close()
    with sqlite3.connect(actions(tmp_path)) as connection:
        columns = {row[1] for row in connection.execute("pragma table_info(agent_sessions)")}
    assert "session_type" in columns


async def test_the_sync_facade_works_on_the_generic_class(tmp_path):
    storage = SQLSessionStorage.for_sqlite(actions(tmp_path))
    try:
        with storage.sync() as blocking:
            session = blocking.create_session(title="sync")
            assert blocking.get_session(session.session_id) == session
    finally:
        await storage.close()
