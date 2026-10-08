"""Content-addressed on-disk cache for LLM responses."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def make_cache_key(model: str, temperature: float, prompt_hash: str, normalized_input: str) -> str:
    """SHA-256 over everything that can change the model's answer.

    ``prompt_hash`` must be the hash of the FULL prompt file content, so editing
    a prompt invalidates its cache even if the version label is unchanged.
    """
    payload = json.dumps(
        {"model": model, "temperature": temperature, "prompt": prompt_hash, "input": normalized_input},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """One JSON file per key under ``directory`` (sharded by key prefix)."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached payload or ``None`` (corrupt entries count as misses)."""
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable cache entry %s: %s", path.name, exc)
            return None

    def set(self, key: str, payload: dict[str, Any]) -> None:
        """Atomically write ``payload`` for ``key``."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
