"""Generate protocol-v9 reports from frozen machine-readable results."""

from __future__ import annotations

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v9 import verify_freeze
from jclosure.reporting_v9 import build_reports


def main() -> None:
    parser = standard_parser(
        "build conditional sufficiency protocol-v9 reports",
        "configs/sufficiency_v9.yaml",
    )
    args = parser.parse_args()
    context = initialize_context("report-v9", args)
    try:
        freeze = verify_freeze(context.root, context.config)
        outputs = build_reports(context.root, context.config, freeze)
        context.finish(
            "COMPLETED_V9_REPORT",
            source_freeze_digest=freeze["freeze_digest"],
            reports=outputs,
        )
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
