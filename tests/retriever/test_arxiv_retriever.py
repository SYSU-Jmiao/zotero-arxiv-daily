"""Tests for ArxivRetriever."""

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_arxiv_retriever_keywords(config, monkeypatch):
    """ArxivRetriever retrieves papers via keyword search when keywords are set."""
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # Configure keyword search (no category)
    from omegaconf import OmegaConf
    config.source.arxiv.category = None
    OmegaConf.update(config, "source.arxiv.keywords", ["Federated Learning", "Edge Computing"])

    now = datetime.now(tz=timezone.utc)
    fake_keyword_results = [
        SimpleNamespace(
            title=f"Paper on {kw}",
            authors=[SimpleNamespace(name="Test Author")],
            summary=f"Abstract mentioning {kw}.",
            pdf_url=f"https://arxiv.org/pdf/2501.0000{i}",
            entry_id=f"https://arxiv.org/abs/2501.0000{i}",
            published=now,
            source_url=lambda i=i: f"https://arxiv.org/e-print/2501.0000{i}",
        )
        for i, kw in enumerate(["Federated Learning", "Edge Computing"])
    ]

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_keyword_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(fake_keyword_results)
    assert set(p.title for p in papers) == {r.title for r in fake_keyword_results}


def test_arxiv_retriever_deduplicates_category_and_keyword(config, mock_feedparser, monkeypatch):
    """Papers found by both category and keyword search are deduplicated."""
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    from omegaconf import OmegaConf
    OmegaConf.update(config, "source.arxiv.keywords", ["Wireless Communication"])

    now = datetime.now(tz=timezone.utc)

    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    # The first category paper will also be returned by keyword search
    duplicate_entry = new_entries[0]
    dup_pid = duplicate_entry.id.removeprefix("oai:arXiv.org:")
    dup_entry_id = f"https://arxiv.org/abs/{dup_pid}"

    category_results = [
        SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{entry.id.removeprefix('oai:arXiv.org:')}",
            entry_id=f"https://arxiv.org/abs/{entry.id.removeprefix('oai:arXiv.org:')}",
            published=now,
            source_url=lambda pid=entry.id.removeprefix("oai:arXiv.org:"): f"https://arxiv.org/e-print/{pid}",
        )
        for entry in new_entries
    ]
    keyword_results = [
        SimpleNamespace(
            title=duplicate_entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{dup_pid}",
            entry_id=dup_entry_id,
            published=now,
            source_url=lambda: f"https://arxiv.org/e-print/{dup_pid}",
        )
    ]

    calls = []

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            calls.append("results")
            if len(calls) == 1:
                return iter(category_results)
            return iter(keyword_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    # The duplicate should appear only once
    titles = [p.title for p in papers]
    assert titles.count(duplicate_entry.title) == 1
    assert len(papers) == len(new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
