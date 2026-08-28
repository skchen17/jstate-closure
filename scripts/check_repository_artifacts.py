#!/usr/bin/env python3
"""Reject tracked large/model/credential-like artifacts before publication."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MAX_BYTES = 95 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".safetensors", ".npy", ".npz"}
CREDENTIAL_PATTERNS = {
    "github_pat": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "huggingface_token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    failures: list[str] = []
    checked = 0
    for relative in sorted(set(result.stdout.splitlines())):
        path = root / relative
        if not path.is_file():
            continue
        checked += 1
        size = path.stat().st_size
        if size > MAX_BYTES:
            failures.append(f"large file ({size} bytes): {relative}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact suffix: {relative}")
        if size <= 10 * 1024 * 1024:
            content = path.read_bytes()
            for name, pattern in CREDENTIAL_PATTERNS.items():
                if pattern.search(content):
                    failures.append(f"credential pattern {name}: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"repository artifact and credential scan passed ({checked} files)")


if __name__ == "__main__":
    main()
