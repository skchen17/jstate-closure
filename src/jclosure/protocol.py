"""Immutable protocol manifests for confirmatory experiment execution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jclosure.config import config_digest, load_config
from jclosure.phase0 import PROTOCOL_VERSION
from jclosure.provenance import git_commit, sha256_file, write_json_atomic


@dataclass(frozen=True)
class ProtocolFreezeManifest:
    protocol_version: str
    implementation_commit: str | None
    config_path: str
    config_digest: str
    workspace_band: tuple[int, ...]
    positive_control_layer: int
    thresholds: dict[str, Any]
    exclusion_policy: dict[str, Any]
    synonym_rule: str
    files: dict[str, str]
    notes: tuple[str, ...] = ()
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if root.resolve() not in resolved.parents and resolved != root.resolve():
        raise ValueError(f"freeze file escapes repository: {path}")
    return str(resolved.relative_to(root.resolve()))


def build_protocol_freeze(
    *,
    root: str | Path,
    config_path: str | Path,
    workspace_band: Iterable[int],
    positive_control_layer: int,
    tracked_files: Iterable[str | Path],
    thresholds: dict[str, Any],
    exclusion_policy: dict[str, Any],
    notes: Iterable[str] = (),
) -> ProtocolFreezeManifest:
    repo = Path(root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo / config_file
    files: dict[str, str] = {}
    all_files = [config_file, *(Path(path) if Path(path).is_absolute() else repo / path for path in tracked_files)]
    for path in all_files:
        relative = _relative(repo, path)
        files[relative] = sha256_file(path)
    config = load_config(config_file)
    return ProtocolFreezeManifest(
        protocol_version=PROTOCOL_VERSION,
        implementation_commit=git_commit(repo),
        config_path=_relative(repo, config_file),
        config_digest=config_digest(config),
        workspace_band=tuple(int(layer) for layer in workspace_band),
        positive_control_layer=int(positive_control_layer),
        thresholds=dict(thresholds),
        exclusion_policy=dict(exclusion_policy),
        synonym_rule="official-compatible-declared-v2",
        files=files,
        notes=tuple(str(note) for note in notes),
    )


def write_protocol_freeze(path: str | Path, manifest: ProtocolFreezeManifest) -> None:
    write_json_atomic(path, manifest.to_dict())


def verify_protocol_freeze(
    path: str | Path, *, root: str | Path
) -> dict[str, Any]:
    repo = Path(root).resolve()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("confirmatory protocol version is not phase0_protocol_v2")
    for relative, expected in payload.get("files", {}).items():
        observed = sha256_file(repo / relative)
        if observed != expected:
            raise RuntimeError(
                f"frozen protocol file changed: {relative}; expected {expected}, observed {observed}"
            )
    config = load_config(repo / payload["config_path"])
    if config_digest(config) != payload["config_digest"]:
        raise RuntimeError("frozen protocol config digest changed")
    return payload
