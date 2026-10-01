from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from config import AppConfig, ConfigError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EMBEDDINGS_PREVIEW_FILE = "embeddings_preview.txt"
PREVIEW_ROWS = 5
PREVIEW_DIMS = 10


def configure_model_env(cfg: AppConfig) -> None:
    """Point the HuggingFace caches inside `paths.models_dir` before sentence-transformers is imported."""
    models_dir = cfg.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(models_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(models_dir / "hub"))
    if cfg.embedding.offline_env and _model_cached(models_dir):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _model_cached(models_dir: Path) -> bool:
    hub = models_dir / "hub"
    if not hub.is_dir():
        return False
    needle = "minilm-l6-v2"
    return any(needle in path.name.lower() for path in hub.iterdir())


@lru_cache(maxsize=1)
def _load(model_name: str, models_dir: str, expected_dim: int) -> "SentenceTransformer":
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device="cpu",
        cache_folder=Path(models_dir),
        model_kwargs={"torch_dtype": torch.float32},
    )
    dimension = int(_embedding_dimension(model))
    if dimension != expected_dim:
        raise ConfigError(f"embedding.model={model_name} gives dim {dimension}, expected {expected_dim}")
    return model


def _embedding_dimension(model: "SentenceTransformer") -> int:
    reader = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(reader())


def get_model(cfg: AppConfig) -> "SentenceTransformer":
    """Process-wide singleton; the offline env vars are set before the first load and reused after it."""
    configure_model_env(cfg)
    model = _load(cfg.embedding.model, str(cfg.paths.models_dir), cfg.embedding.dim)
    if cfg.embedding.offline_env:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return model


def embed_documents(texts: Sequence[str], cfg: AppConfig) -> np.ndarray:
    """`(n, dim)` float32, L2-normalised, one row per input text in the same order."""
    if not texts:
        return np.zeros((0, cfg.embedding.dim), dtype=np.float32)
    model = get_model(cfg)
    vectors = model.encode(
        list(texts),
        batch_size=cfg.embedding.batch_size,
        normalize_embeddings=cfg.embedding.normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (len(texts), cfg.embedding.dim):
        raise ConfigError(f"embed_documents returned shape {array.shape}, expected ({len(texts)}, {cfg.embedding.dim})")
    return array


def embed_query(text: str, cfg: AppConfig) -> np.ndarray:
    """`(dim,)` float32 unit vector for the question, from the same model as the corpus."""
    return embed_documents([text], cfg)[0]


def real_token_count(text: str, cfg: AppConfig) -> int:
    """Tokenizer count with no truncation; the true check behind the chunker's estimate."""
    tokenizer = get_model(cfg).tokenizer
    return len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])


def embeddings_preview_path(cfg: AppConfig) -> Path:
    """`data/embeddings_preview.txt`, next to the persistent store, so no config key is added."""
    return cfg.paths.persist_dir.parent / EMBEDDINGS_PREVIEW_FILE


def render_embeddings_preview(vectors: np.ndarray, texts: Sequence[str], cfg: AppConfig) -> str:
    rows = min(PREVIEW_ROWS, len(vectors))
    lines = [
        f"model: {cfg.embedding.model}",
        f"dim: {cfg.embedding.dim} | dtype: {vectors.dtype} | normalise: {cfg.embedding.normalize}",
        f"vectors: {len(vectors)} | preview_rows: {rows} | preview_dims: {PREVIEW_DIMS}",
        "",
    ]
    for index in range(rows):
        head = " ".join(f"{float(value):+.4f}" for value in vectors[index][:PREVIEW_DIMS])
        snippet = texts[index].replace("\n", " ")[:80]
        lines.append(f"[{index}] l2_norm={float(np.linalg.norm(vectors[index])):.4f} first_{PREVIEW_DIMS}: {head}")
        lines.append(f"     text: {snippet}")
    lines.append("")
    return "\n".join(lines)


def write_embeddings_preview(vectors: np.ndarray, texts: Sequence[str], cfg: AppConfig) -> Path:
    path = embeddings_preview_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(render_embeddings_preview(vectors, texts, cfg).encode("utf-8"))
    os.replace(temp, path)
    return path
