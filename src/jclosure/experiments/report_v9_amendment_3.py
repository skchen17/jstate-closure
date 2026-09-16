"""Freeze or execute the visual-QA protocol-v9 reporting amendment."""

from __future__ import annotations

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v9 import verify_freeze
from jclosure.protocol_v9_reporting_amendment_1 import verify_amendment
from jclosure.protocol_v9_reporting_amendment_2 import verify_amendment_2
from jclosure.protocol_v9_reporting_amendment_3 import (
    build_amendment,
    verify_amendment_3,
)
from jclosure.reporting_v9_amendment_3 import build_reports


def main() -> None:
    parser = standard_parser(
        "protocol-v9 reporting amendment 3",
        "configs/sufficiency_v9.yaml",
    )
    parser.add_argument("--stage", required=True, choices=("freeze", "report"))
    args = parser.parse_args()
    context = initialize_context("report-v9-amendment-3", args)
    try:
        if args.stage == "freeze":
            amendment = build_amendment(context.root, context.config)
            context.finish("COMPLETED_V9_REPORTING_AMENDMENT_3_FREEZE", **amendment)
            return
        base = verify_freeze(context.root, context.config)
        amendment_1 = verify_amendment(context.root, context.config)
        amendment_2 = verify_amendment_2(context.root, context.config)
        amendment_3 = verify_amendment_3(context.root, context.config)
        outputs = build_reports(
            context.root,
            context.config,
            base,
            amendment_1,
            amendment_2,
            amendment_3,
        )
        context.finish(
            "COMPLETED_V9_REPORTING_AMENDMENT_3",
            base_freeze_digest=base["freeze_digest"],
            amendment_digest=amendment_3["amendment_digest"],
            reports=outputs,
        )
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
