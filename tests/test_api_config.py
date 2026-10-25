import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sqlalchemy") is None,
    reason="API optional dependency is not installed",
)


def test_database_url_is_read_after_module_import(monkeypatch):
    from analysis_loop_v3.api import db

    first = "postgresql+asyncpg://first:first@localhost:5432/first"
    second = "postgresql+asyncpg://second:second@localhost:5432/second"

    monkeypatch.setenv("DATABASE_URL", first)
    assert db._database_url() == first

    monkeypatch.setenv("DATABASE_URL", second)
    assert db._database_url() == second
