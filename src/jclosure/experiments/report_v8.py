"""Build protocol-v8 reports from machine-readable results."""

from __future__ import annotations

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v8 import verify_freeze
from jclosure.reporting_v8 import build_reports


def main() -> None:
    parser = standard_parser(
        "build persistent-state v8 reports", "configs/persistent_state_v8.yaml"
    )
    args = parser.parse_args()
    context = initialize_context("report-v8", args)
    try:
        freeze = verify_freeze(context.root, context.config)
        outputs = build_reports(context.root)
        context.finish(
            "COMPLETED_REPORT",
            source_freeze_digest=freeze["freeze_digest"],
            reports=outputs,
        )
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
