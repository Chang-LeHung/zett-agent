from uuid import UUID

from kcs_agent.ids import new_uuid7


def test_new_uuid7_returns_time_ordered_version_seven_ids():
    identifiers = [new_uuid7() for _ in range(3)]
    assert all(UUID(identifier).version == 7 for identifier in identifiers)
    assert identifiers == sorted(identifiers)
