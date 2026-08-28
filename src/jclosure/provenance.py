"""Run manifests, hashing, deterministic seeds, and append-only records."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.config import config_digest, public_config

SCHEMA_VERSION = 2


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )


def set_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = deterministic
    deterministic_error = None
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception as exc:  # pragma: no cover - version dependent
        deterministic_error = f"{type(exc).__name__}: {exc}"
    return {
        "seed": seed,
        "deterministic_requested": deterministic,
        "deterministic_setup_error": deterministic_error,
    }


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = result.stdout.strip() or result.stderr.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_snapshot() -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "compute_capability": [props.major, props.minor],
                }
            )
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpus,
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def git_commit(root: str | Path) -> str | None:
    value = _command_output(["git", "-C", str(Path(root)), "rev-parse", "HEAD"])
    return value if value and not value.startswith("fatal:") else None


def make_run_id(kind: str, digest: str, seed: int) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{timestamp}-{digest[:8]}-s{seed}"


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(value, handle, sort_keys=True, indent=2, default=json_default)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, target)


def json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, default=json_default))
            handle.write("\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def build_manifest(
    *,
    kind: str,
    config: dict[str, Any],
    seed: int,
    repo_root: str | Path,
    command: list[str],
    status: str = "RUNNING",
) -> dict[str, Any]:
    digest = config_digest(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": make_run_id(kind, digest, seed),
        "kind": kind,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "config_digest": digest,
        "config": public_config(config),
        "seed": seed,
        "command": command,
        "git_commit": git_commit(repo_root),
        "environment": environment_snapshot(),
    }
