from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urljoin, urlsplit

import requests

from app.models import FetchResult, SourceSpec
from config import AppConfig, ConfigError

USER_AGENT = "Milestone4-MF-FAQ-Research/0.1 (+contact: milestone4-team@example.invalid)"
REQUEST_TIMEOUT_SECONDS = 30
MIN_HOST_INTERVAL_SECONDS = 2.0
MAX_ATTEMPTS = 3
MAX_REDIRECTS = 3
MAX_RETRY_AFTER_SECONDS = 60

_LOG = logging.getLogger("ingest.loader")
_LAST_REQUEST_AT: dict[str, float] = {}
_ANCHOR = re.compile(r"<a\b[^>]*?\bhref\s*=\s*(\"([^\"]*)\"|'([^']*)'|([^\s>]+))", re.IGNORECASE)


class AllowlistError(ConfigError):
    pass


def cached_path(spec: SourceSpec, cfg: AppConfig) -> Path:
    return cfg.paths.raw_dir / spec.domain / f"{spec.id}.html"


def meta_path(spec: SourceSpec, cfg: AppConfig) -> Path:
    return cfg.paths.raw_dir / spec.domain / f"{spec.id}.meta.json"


def noncollectable_path(spec: SourceSpec, cfg: AppConfig) -> Path:
    return cfg.paths.raw_dir / f"{spec.id}.noncollectable.json"


def fetch_source(spec: SourceSpec, cfg: AppConfig, *, force: bool = False) -> FetchResult:
    """Fetch one allowlisted source, honouring the raw cache unless `force` is set."""
    assert_allowed(spec, cfg)
    if not force:
        cached = _read_cache(spec, cfg)
        if cached is not None:
            return cached
    body, final_url, http_code, fetched_at, status, error = _download(spec.url, cfg)
    if body is None:
        _LOG.info(
            "[load] %s %s http=%s error=%s",
            spec.id,
            status,
            http_code if http_code is not None else "-",
            error or "-",
        )
        return FetchResult(
            source_id=spec.id,
            url=spec.url,
            status=status,
            http_code=http_code,
            bytes=0,
            sha256=None,
            fetched_at=fetched_at,
            cache_hit=False,
            error=error,
        )
    digest = _sha256(body)
    html_path = cached_path(spec, cfg)
    _write_atomic(html_path, body)
    _write_atomic(
        meta_path(spec, cfg),
        json.dumps(
            {
                "url": final_url,
                "http_code": http_code,
                "fetched_at": fetched_at,
                "sha256": digest,
                "bytes": len(body),
                "content_type": "text/html",
            },
            indent=2,
        ).encode("utf-8"),
    )
    _write_noncollectable(spec, cfg, body, final_url)
    _LOG.info("[load] %s ok http=%s bytes=%s", spec.id, http_code, len(body))
    return FetchResult(
        source_id=spec.id,
        url=spec.url,
        status="ok",
        http_code=http_code,
        bytes=len(body),
        sha256=digest,
        fetched_at=fetched_at,
        cache_hit=False,
        error=None,
    )


def fetch_all(specs: Sequence[SourceSpec], cfg: AppConfig, *, force: bool = False) -> list[FetchResult]:
    """Fetch every spec in order; a single failure never aborts the batch."""
    results: list[FetchResult] = []
    for spec in specs:
        try:
            results.append(fetch_source(spec, cfg, force=force))
        except AllowlistError as exc:
            _LOG.info("[load] %s blocked error=%s", spec.id, exc)
            results.append(
                FetchResult(
                    source_id=spec.id,
                    url=spec.url,
                    status="blocked",
                    http_code=None,
                    bytes=0,
                    sha256=None,
                    fetched_at=_utc_now(),
                    cache_hit=False,
                    error=str(exc),
                )
            )
    return results


def render_report(results: Sequence[FetchResult]) -> str:
    """Render the FR-11.1 per-URL table; elapsed covers only this run's network fetches."""
    header = ("source_id", "http", "cache", "bytes", "sha256", "status", "error")
    rows = [
        (
            result.source_id,
            str(result.http_code) if result.http_code is not None else "-",
            "True" if result.cache_hit else "False",
            str(result.bytes),
            result.sha256[:8] if result.sha256 else "-",
            result.status,
            result.error or "-",
        )
        for result in results
    ]
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows)] if rows else [
        len(name) for name in header
    ]
    lines = [" | ".join(name.ljust(width) for name, width in zip(header, widths)).rstrip()]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows)
    ok = sum(1 for result in results if result.status == "ok")
    blocked = sum(1 for result in results if result.status == "blocked")
    errored = sum(1 for result in results if result.status == "error")
    lines.append(
        f"totals: {len(results)} requested, {ok} ok, {blocked} blocked, {errored} error, "
        f"{_elapsed_seconds(results):.1f}s elapsed"
    )
    return "\n".join(lines)


def assert_allowed(spec: SourceSpec, cfg: AppConfig) -> None:
    parts = urlsplit(spec.url)
    if parts.scheme != "https":
        raise AllowlistError(f"scheme_not_allowed:{parts.scheme or 'none'}")
    host = (parts.hostname or "").lower()
    if host not in cfg.allowed_domains:
        raise AllowlistError(f"host_not_allowlisted:{host or 'none'}")


def _download(url: str, cfg: AppConfig) -> tuple[bytes | None, str, int | None, str, str, str | None]:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        host = (urlsplit(current).hostname or "").lower()
        if host not in cfg.allowed_domains:
            return None, current, None, _utc_now(), "blocked", "redirect_off_allowlist"
        if urlsplit(current).scheme != "https":
            return None, current, None, _utc_now(), "blocked", "redirect_scheme_not_allowed"
        response, transport_error = _request_with_retries(current, cfg)
        fetched_at = _utc_now()
        if transport_error is not None:
            return None, current, None, fetched_at, "error", transport_error
        code = int(response.status_code)
        if 300 <= code < 400:
            location = response.headers.get("Location") or response.headers.get("location")
            if not location:
                return None, current, code, fetched_at, "error", "redirect_without_location"
            current = urljoin(current, location)
            continue
        if code == 200:
            return bytes(response.content), current, code, fetched_at, "ok", None
        if code == 403:
            return None, current, code, fetched_at, "blocked", "http_403_forbidden"
        if code == 429:
            return None, current, code, fetched_at, "blocked", "http_429_rate_limited"
        if 400 <= code < 500:
            return None, current, code, fetched_at, "error", f"http_{code}"
        if code >= 500:
            return None, current, code, fetched_at, "error", f"http_{code}"
        return None, current, code, fetched_at, "error", f"http_{code}"
    return None, current, None, _utc_now(), "error", "too_many_redirects"


def _request_with_retries(url: str, cfg: AppConfig) -> tuple[Any, str | None]:
    session = _new_session()
    try:
        for attempt in range(MAX_ATTEMPTS):
            _wait_for_host(url)
            try:
                response = session.request(
                    "GET",
                    url,
                    headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    return None, f"{type(exc).__name__}: {exc}"
                _sleep(_backoff_seconds(attempt, None))
                continue
            if int(response.status_code) in {429} or int(response.status_code) >= 500:
                if attempt == MAX_ATTEMPTS - 1:
                    return response, None
                _sleep(_backoff_seconds(attempt, response.headers.get("Retry-After")))
                continue
            return response, None
    finally:
        session.close()
    return None, "retries_exhausted"


def _new_session() -> requests.Session:
    return requests.Session()


def _wait_for_host(url: str) -> None:
    host = (urlsplit(url).hostname or "").lower()
    now = time.monotonic()
    previous = _LAST_REQUEST_AT.get(host)
    if previous is not None:
        remaining = MIN_HOST_INTERVAL_SECONDS - (now - previous)
        if remaining > 0:
            _sleep(remaining)
    _LAST_REQUEST_AT[host] = time.monotonic()


def _backoff_seconds(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            requested = float(retry_after)
        except (TypeError, ValueError):
            requested = -1.0
        if 0.0 <= requested <= MAX_RETRY_AFTER_SECONDS:
            return requested
    return float(2**attempt) + random.uniform(0.0, 0.5)


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _read_cache(spec: SourceSpec, cfg: AppConfig) -> FetchResult | None:
    html_path = cached_path(spec, cfg)
    meta_file = meta_path(spec, cfg)
    if not html_path.is_file() or not meta_file.is_file():
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    body = html_path.read_bytes()
    digest = _sha256(body)
    _write_noncollectable(spec, cfg, body, str(meta.get("url", spec.url)))
    _LOG.info("[load] %s cache http=%s bytes=%s", spec.id, meta.get("http_code", "-"), len(body))
    return FetchResult(
        source_id=spec.id,
        url=spec.url,
        status="ok",
        http_code=int(meta["http_code"]) if meta.get("http_code") is not None else None,
        bytes=len(body),
        sha256=digest,
        fetched_at=str(meta.get("fetched_at", _utc_now())),
        cache_hit=True,
        error=None,
    )


def _write_noncollectable(spec: SourceSpec, cfg: AppConfig, body: bytes, base_url: str) -> None:
    text = body.decode("utf-8", errors="replace")
    counts: dict[str, int] = {}
    for match in _ANCHOR.finditer(text):
        href = next((group for group in match.groups()[1:] if group), "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        target = urljoin(base_url, href)
        host = (urlsplit(target).hostname or "").lower()
        if host and host not in cfg.allowed_domains:
            counts[target] = counts.get(target, 0) + 1
    _write_atomic(
        noncollectable_path(spec, cfg),
        json.dumps(
            {
                "source_id": spec.id,
                "page_url": base_url,
                "count": sum(counts.values()),
                "links": [{"url": url, "occurrences": n} for url, n in sorted(counts.items())],
            },
            indent=2,
        ).encode("utf-8"),
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _elapsed_seconds(results: Sequence[FetchResult]) -> float:
    """Span of the network fetches performed in this run; cache hits contribute no wall time."""
    stamps: list[float] = []
    for result in results:
        if result.cache_hit:
            continue
        try:
            stamps.append(
                datetime.strptime(result.fetched_at, "%Y-%m-%dT%H:%M:%SZ")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        except (TypeError, ValueError):
            continue
    return max(stamps) - min(stamps) if len(stamps) > 1 else 0.0
