import pytest


@pytest.fixture(autouse=True)
def isolated_index(tmp_path, monkeypatch):
    """Every test gets its own empty docs folder and index, and no startup indexing."""
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setenv("DOCS_DIR", str(docs))
    monkeypatch.setenv("INDEX_DB", str(tmp_path / "index.sqlite"))
    monkeypatch.setenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("AUTO_INDEX", "0")
    return docs
