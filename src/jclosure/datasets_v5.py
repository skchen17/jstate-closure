"""Independent frozen H2-replication tasks for protocol v5."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from jclosure.datasets_v4 import (
    ProgramTraceTask,
    generate_program_tasks,
    verify_disjoint_domains,
)
from jclosure.provenance import write_json_atomic

PROTOCOL_V5 = "jstate_peripheral_protocol_v5"


def v4_program_hashes(root: Path) -> set[str]:
    output: set[str] = set()
    for path in sorted((root / "data/v4").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        output.update(str(item["program_hash"]) for item in payload["items"])
    return output


def build_replication_tasks(
    root: Path, config: dict[str, Any]
) -> list[ProgramTraceTask]:
    section = config["h2_replication_v5"]
    seed = int(section["seeds"]["evaluation"])
    horizons = tuple(int(value) for value in section["horizons"])
    blocked = v4_program_hashes(root)
    tasks: list[ProgramTraceTask] = []
    for family_index, (family, count) in enumerate(
        sorted(section["counts_by_family"].items())
    ):
        generated = generate_program_tasks(
            str(family),
            str(section["variants"][family]),
            int(count),
            seed=seed + family_index * 100_003,
            horizons=horizons,
            blocked_program_hashes=blocked,
        )
        blocked.update(value.program_hash for value in generated)
        for value in generated:
            tasks.append(
                replace(
                    value,
                    example_id=value.example_id.replace("v4-", "v5-rep-", 1),
                    template_id=f"v5-rep:{value.template_id}",
                )
            )
    verify_disjoint_domains({"h2_replication": tasks})
    overlap = {value.program_hash for value in tasks} & v4_program_hashes(root)
    if overlap:
        raise RuntimeError("v5 H2 replication overlaps a frozen v4 program")
    return sorted(tasks, key=lambda value: value.example_id)


def write_replication_tasks(
    root: Path, config: dict[str, Any], path: Path
) -> list[ProgramTraceTask]:
    tasks = build_replication_tasks(root, config)
    write_json_atomic(
        path,
        {
            "schema_version": 7,
            "protocol_version": PROTOCOL_V5,
            "domain": "independent_h2_replication",
            "items": [value.to_dict() for value in tasks],
        },
    )
    return tasks


def load_replication_tasks(path: Path) -> list[ProgramTraceTask]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for item in payload["items"]:
        value = dict(item)
        value["semantic_actions"] = tuple(value["semantic_actions"])
        output.append(ProgramTraceTask(**value))
    return output
