from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.models import SourceSpec
from config import AppConfig
from ingest import loader
from ingest.loader import AllowlistError, fetch_all, fetch_source, render_report

HTML = (
    "<html><body><h1>HDFC Large Cap Fund</h1><a href='/mutual-funds/x'>x</a>"
    "<h2>About HDFC Large Cap Fund</h2><p>Expense ratio 1.03% as on 30 Sep 2026.</p>"
    "<p>The scheme seeks to provide long-term capital appreciation by investing predominantly in "
    "Large-Cap companies. The benchmark is NIFTY 100 Total Return Index and the exit load is 1% if the "
    "units are redeemed within 1 year. The NAV of the scheme was 11,165.78 as on 30 Sep 2026 and the "
    "minimum investment amount is 100 per SIP instalment for the direct plan growth option.</p>"
    "<table><tr><th>Name</th><th>Weight</th></tr><tr><td>ICICI Bank Ltd</td><td>10.05%</td></tr></table>"
    "</body></html>"
)


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", headers: dict | None = None) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeSession:
    def __init__(self, responses: list | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[str] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs) -> FakeResponse:
        self.calls.append(url)
        response = self.responses.pop(0) if self.responses else FakeResponse(200, HTML.encode("utf-8"))
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(loader.time, "sleep", lambda seconds: slept.append(seconds))
    loader._LAST_REQUEST_AT.clear()
    return slept


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch):
    created: list[FakeSession] = []

    def _install(responses: list) -> FakeSession:
        session = FakeSession(responses)
        created.append(session)
        monkeypatch.setattr(loader, "_new_session", lambda: session)
        return session

    _install.created = created  # type: ignore[attr-defined]
    return _install


def test_disallowed_host_raises_before_any_network_call(tmp_config: AppConfig, fake_session, no_network) -> None:
    spec = SourceSpec(
        id="sx",
        url="https://evil.example.com/mutual-funds/x",
        domain="evil.example.com",
        doc_type="scheme_page",
        scheme_name=None,
        scheme_category=None,
        plan=None,
        option=None,
        publisher="test",
    )
    with pytest.raises(AllowlistError):
        fetch_source(spec, tmp_config)
    assert fake_session.created == []


def test_fetch_all_reports_blocked_for_disallowed_host(tmp_config: AppConfig, fake_session) -> None:
    spec = SourceSpec(
        id="sx",
        url="http://groww.in/insecure",
        domain="groww.in",
        doc_type="scheme_page",
        scheme_name=None,
        scheme_category=None,
        plan=None,
        option=None,
        publisher="test",
    )
    results = fetch_all([spec], tmp_config)
    assert [result.status for result in results] == ["blocked"]
    assert fake_session.created == []


def test_cache_reuse_makes_one_network_call(tmp_config: AppConfig, fake_session) -> None:
    spec = tmp_config.sources[0]
    session = fake_session([FakeResponse(200, HTML.encode("utf-8"))])
    first = fetch_source(spec, tmp_config)
    second = fetch_source(spec, tmp_config)
    assert first.status == "ok" and first.cache_hit is False
    assert second.status == "ok" and second.cache_hit is True
    assert len(session.calls) == 1
    assert session.closed is True


def test_force_refetches(tmp_config: AppConfig, fake_session) -> None:
    spec = tmp_config.sources[0]
    session = fake_session(
        [FakeResponse(200, HTML.encode("utf-8")), FakeResponse(200, HTML.encode("utf-8") + b"<!--v2-->")]
    )
    fetch_source(spec, tmp_config)
    refetched = fetch_source(spec, tmp_config, force=True)
    assert refetched.cache_hit is False
    assert len(session.calls) == 2


def test_cached_sha256_matches_file_bytes(tmp_config: AppConfig, fake_session) -> None:
    spec = tmp_config.sources[0]
    fake_session([FakeResponse(200, HTML.encode("utf-8"))])
    fetched = fetch_source(spec, tmp_config)
    cached = fetch_source(spec, tmp_config)
    body = loader.cached_path(spec, tmp_config).read_bytes()
    assert cached.sha256 == hashlib.sha256(body).hexdigest()
    assert cached.sha256 == fetched.sha256


def test_403_is_blocked_without_retry(tmp_config: AppConfig, fake_session) -> None:
    session = fake_session([FakeResponse(403, b"nope", {"Content-Type": "text/html"})])
    result = fetch_source(tmp_config.sources[0], tmp_config)
    assert result.status == "blocked"
    assert result.http_code == 403
    assert len(session.calls) == 1


def test_429_is_retried_at_most_twice_honouring_retry_after(
    tmp_config: AppConfig, fake_session, _no_real_sleep: list[float]
) -> None:
    session = fake_session(
        [
            FakeResponse(429, b"slow down", {"Retry-After": "1"}),
            FakeResponse(429, b"slow down", {"Retry-After": "1"}),
            FakeResponse(429, b"slow down", {"Retry-After": "1"}),
        ]
    )
    result = fetch_source(tmp_config.sources[0], tmp_config)
    assert result.status == "blocked"
    assert result.http_code == 429
    assert len(session.calls) == 3
    assert 1.0 in _no_real_sleep


def test_5xx_is_retried_then_reported_as_error(tmp_config: AppConfig, fake_session) -> None:
    session = fake_session([FakeResponse(503, b"", {}), FakeResponse(503, b"", {}), FakeResponse(503, b"", {})])
    result = fetch_source(tmp_config.sources[0], tmp_config)
    assert result.status == "error"
    assert result.error == "http_503"
    assert len(session.calls) == 3


def test_connection_error_is_reported_as_error(tmp_config: AppConfig, fake_session) -> None:
    import requests

    session = fake_session([requests.ConnectionError("dns"), requests.ConnectionError("dns"), requests.ConnectionError("dns")])
    result = fetch_source(tmp_config.sources[0], tmp_config)
    assert result.status == "error"
    assert result.error is not None and result.error.startswith("ConnectionError")
    assert len(session.calls) == 3


def test_cross_host_redirect_is_blocked(tmp_config: AppConfig, fake_session) -> None:
    session = fake_session(
        [FakeResponse(302, b"", {"Location": "https://evil.example.com/x"}), FakeResponse(200, b"x")]
    )
    result = fetch_source(tmp_config.sources[0], tmp_config)
    assert result.status == "blocked"
    assert result.error == "redirect_off_allowlist"
    assert len(session.calls) == 1


def test_allowlisted_redirect_is_followed(tmp_config: AppConfig, fake_session) -> None:
    spec = tmp_config.sources[0]
    session = fake_session(
        [
            FakeResponse(301, b"", {"Location": "https://groww.in/mutual-funds/moved"}),
            FakeResponse(200, HTML.encode("utf-8")),
        ]
    )
    result = fetch_source(spec, tmp_config)
    assert result.status == "ok"
    assert session.calls[-1] == "https://groww.in/mutual-funds/moved"
    meta = json.loads(loader.meta_path(spec, tmp_config).read_text(encoding="utf-8"))
    assert meta["url"] == "https://groww.in/mutual-funds/moved"


def test_redirect_loop_is_capped(tmp_config: AppConfig, fake_session) -> None:
    session = fake_session(
        [FakeResponse(302, b"", {"Location": "https://groww.in/loop"}) for _ in range(6)]
    )
    result = fetch_source(tmp_config.sources[0], tmp_config)
    assert result.status == "error"
    assert result.error == "too_many_redirects"
    assert len(session.calls) <= loader.MAX_REDIRECTS + 1


def test_noncollectable_links_recorded_and_not_fetched(tmp_config: AppConfig, fake_session) -> None:
    spec = tmp_config.sources[0]
    body = (
        b"<html><body><a href='/ok'>a</a><a href='https://www.google.com/x'>b</a>"
        b"<a href='https://www.google.com/x'>c</a><a href='mailto:a@b.com'>d</a></body></html>"
    )
    session = fake_session([FakeResponse(200, body)])
    fetch_source(spec, tmp_config)
    record = json.loads(loader.noncollectable_path(spec, tmp_config).read_text(encoding="utf-8"))
    assert record["count"] == 2
    assert [link["url"] for link in record["links"]] == ["https://www.google.com/x"]
    assert all("google.com" not in call for call in session.calls)


def test_user_agent_is_descriptive(tmp_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    class RecordingSession(FakeSession):
        def request(self, method: str, url: str, **kwargs) -> FakeResponse:
            seen.update(kwargs.get("headers", {}))
            return FakeResponse(200, HTML.encode("utf-8"))

    monkeypatch.setattr(loader, "_new_session", RecordingSession)
    fetch_source(tmp_config.sources[0], tmp_config)
    assert seen["User-Agent"].startswith("Milestone4-MF-FAQ-Research/")
    assert "Mozilla" not in seen["User-Agent"]


def test_politeness_wait_between_same_host_requests(tmp_config: AppConfig, fake_session, _no_real_sleep) -> None:
    fake_session([FakeResponse(503, b"", {}), FakeResponse(503, b"", {}), FakeResponse(503, b"", {})])
    fetch_source(tmp_config.sources[0], tmp_config)
    assert any(abs(value - loader.MIN_HOST_INTERVAL_SECONDS) < 0.1 for value in _no_real_sleep)


def test_fetch_all_returns_one_result_per_spec(tmp_config: AppConfig, fake_session) -> None:
    session = fake_session([FakeResponse(200, HTML.encode("utf-8")) for _ in range(len(tmp_config.sources))])
    results = fetch_all(tmp_config.sources, tmp_config)
    assert [result.source_id for result in results] == [spec.id for spec in tmp_config.sources]
    assert all(result.status == "ok" for result in results)
    assert len(session.calls) == len(tmp_config.sources)


def test_report_columns_and_totals(tmp_config: AppConfig, fake_session) -> None:
    session = fake_session([FakeResponse(200, HTML.encode("utf-8")), FakeResponse(403, b"", {})])
    results = fetch_all(tmp_config.sources[:2], tmp_config)
    report = render_report(results)
    header = report.splitlines()[0]
    assert "source_id" in header and "cache" in header and "sha256" in header
    assert "totals: 2 requested, 1 ok, 1 blocked, 0 error" in report
    assert "False" in report


def test_report_is_readable_for_full_corpus(tmp_config: AppConfig, fake_session) -> None:
    session = fake_session([FakeResponse(200, HTML.encode("utf-8")) for _ in range(len(tmp_config.sources))])
    fetch_all(tmp_config.sources, tmp_config)
    report = render_report(fetch_all(tmp_config.sources, tmp_config))
    assert "cache" in report.splitlines()[0]
    assert report.splitlines()[-1].startswith("totals: 2 requested")


def test_raw_cache_layout(tmp_config: AppConfig, fake_session) -> None:
    spec = tmp_config.sources[0]
    fake_session([FakeResponse(200, HTML.encode("utf-8"))])
    fetch_source(spec, tmp_config)
    expected = tmp_config.paths.raw_dir / "groww.in" / "s1.html"
    assert loader.cached_path(spec, tmp_config) == expected
    assert expected.is_file()
    assert loader.meta_path(spec, tmp_config).is_file()
    assert expected.read_text(encoding="utf-8") == HTML


def test_meta_records_audit_fields(tmp_config: AppConfig, fake_session) -> None:
    spec = tmp_config.sources[0]
    fake_session([FakeResponse(200, HTML.encode("utf-8"))])
    result = fetch_source(spec, tmp_config)
    meta = json.loads(loader.meta_path(spec, tmp_config).read_text(encoding="utf-8"))
    assert set(meta) == {"url", "http_code", "fetched_at", "sha256", "bytes", "content_type"}
    assert meta["sha256"] == result.sha256
    assert meta["fetched_at"].endswith("Z")
    assert meta["bytes"] == len(HTML.encode("utf-8"))


def test_second_full_run_is_fully_cached(tmp_config: AppConfig, fake_session) -> None:
    session = fake_session([FakeResponse(200, HTML.encode("utf-8")) for _ in range(len(tmp_config.sources))])
    fetch_all(tmp_config.sources, tmp_config)
    assert not any(loader.cached_path(spec, tmp_config).is_file() is False for spec in tmp_config.sources)
    results = fetch_all(tmp_config.sources, tmp_config)
    assert all(result.cache_hit for result in results)
    assert len(session.calls) == len(tmp_config.sources)


def test_ingest_layering_never_imports_app_behaviour_or_llm_sdks() -> None:
    import ingest

    root = Path(ingest.__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "streamlit" not in source
        assert "groq" not in source
        for line in source.splitlines():
            if line.startswith(("import app.", "from app.")):
                assert line.split()[1] == "app.models", f"{path.name}: {line}"


def test_run_ingest_dry_run_exits_zero_on_cache(write_config, fake_session, capsys) -> None:
    import run_ingest

    path = write_config()
    session = fake_session([FakeResponse(200, HTML.encode("utf-8")) for _ in range(2)])
    assert run_ingest.run(path) == 0
    assert run_ingest.run(path) == 0
    out = capsys.readouterr().out
    assert out.count("True") == 2
    assert len(session.calls) == 2


def test_run_ingest_exits_one_when_all_sources_fail(write_config, fake_session, capsys) -> None:
    import run_ingest

    path = write_config()
    fake_session([FakeResponse(403, b"", {}) for _ in range(2)])
    assert run_ingest.run(path) == 1
    assert "did not reach status ok" in capsys.readouterr().out


def test_run_ingest_sources_flag_restricts_run(write_config, fake_session) -> None:
    import run_ingest

    path = write_config()
    session = fake_session([FakeResponse(200, HTML.encode("utf-8"))])
    assert run_ingest.run(path, sources="s2") == 0
    assert session.calls == ["https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"]


def test_run_ingest_rejects_unknown_source_id(write_config) -> None:
    import run_ingest
    from config import ConfigError as CfgError

    path = write_config()
    with pytest.raises(CfgError):
        run_ingest.run(path, sources="s9")
