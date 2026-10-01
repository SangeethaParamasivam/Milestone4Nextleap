from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import pytest

from config import REPO_ROOT, ConfigError, get_groq_api_key, get_groq_model, load_config
from tests.conftest import CONFIG_TEXT, yaml_without


def test_loads_five_sources_with_ids_s1_to_s5(real_config) -> None:
    assert [spec.id for spec in real_config.sources] == ["s1", "s2", "s3", "s4", "s5"]
    assert len(real_config.sources) == 5


def test_every_source_host_is_allowlisted(real_config) -> None:
    for spec in real_config.sources:
        assert (urlsplit(spec.url).hostname or "") in real_config.allowed_domains
        assert spec.domain in real_config.allowed_domains
        assert spec.url.startswith("https://")


def test_source_with_foreign_host_raises(write_config) -> None:
    text = CONFIG_TEXT.replace(
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "https://evil.example.com/mutual-funds/hdfc-large-cap-fund-direct-growth",
    )
    with pytest.raises(ConfigError):
        load_config(write_config(text))


def test_non_https_source_raises(write_config) -> None:
    text = CONFIG_TEXT.replace("https://groww.in", "http://groww.in")
    with pytest.raises(ConfigError):
        load_config(write_config(text))


def test_env_overrides_yaml(tmp_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "9")
    load_config.cache_clear()
    assert load_config().retrieval.top_k == 9


def test_env_overrides_yaml_for_temp_config(write_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIN_SCORE", "0.5")
    assert load_config(write_config()).retrieval.min_score == 0.5


def test_env_file_does_not_override_real_env(write_config, repo_dotenv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_dotenv.write_text("TOP_K=2\n", encoding="utf-8")
    monkeypatch.setenv("TOP_K", "9")
    assert load_config(write_config()).retrieval.top_k == 9


def test_env_file_used_when_real_env_absent(write_config, repo_dotenv: Path) -> None:
    repo_dotenv.write_text("TOP_K=2\n", encoding="utf-8")
    assert load_config(write_config()).retrieval.top_k == 2


def test_missing_optional_keys_fall_back_to_defaults(write_config) -> None:
    cfg = load_config(write_config(yaml_without(CONFIG_TEXT, "freshness", "topics", "out_of_corpus")))
    assert cfg.freshness_warn_days == 90
    assert cfg.topics
    assert "parag parflex" not in cfg.out_of_corpus
    assert cfg.paths.raw_dir.name == "raw"


def test_paths_are_absolute_and_under_repo_root(real_config) -> None:
    paths = real_config.paths
    for value in (
        paths.raw_dir,
        paths.chunks_dir,
        paths.persist_dir,
        paths.cache_dir,
        paths.models_dir,
        paths.status_file,
        paths.rejected_file,
    ):
        assert value.is_absolute()
        assert REPO_ROOT in value.parents or value.parent == REPO_ROOT


def test_env_overrides_paths(write_config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PERSIST_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("RAW_DIR", str(tmp_path / "raw-elsewhere"))
    cfg = load_config(write_config())
    assert cfg.paths.persist_dir == tmp_path / "elsewhere"
    assert cfg.paths.raw_dir == tmp_path / "raw-elsewhere"


def test_groq_helpers_read_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_groq_api_key() is None
    assert get_groq_model() == "llama-3.1-8b-instant"
    monkeypatch.setenv("GROQ_API_KEY", "  gsk_example  ")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    assert get_groq_api_key() == "gsk_example"
    assert get_groq_model() == "llama-3.3-70b-versatile"


def test_load_config_is_cached(real_config) -> None:
    assert load_config() is load_config()


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.yaml")


def test_related_topics_parsed_from_retrieval(real_config) -> None:
    assert real_config.related_topics == {
        "objective": ("benchmark",),
        "benchmark": ("objective",),
    }


def test_related_topics_empty_when_absent(tmp_config) -> None:
    assert tmp_config.related_topics == {}
