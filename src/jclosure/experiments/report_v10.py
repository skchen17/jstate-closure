"""Generate protocol-v10 reports from frozen machine results."""

from __future__ import annotations

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v10 import verify_candidate_freeze, verify_freeze
from jclosure.reporting_v10 import build_reports


def main() -> None:
    parser = standard_parser(
        "build corrected causal-sufficiency protocol-v10 reports",
        "configs/causal_sufficiency_v10.yaml",
    )
    args = parser.parse_args()
    context = initialize_context("report-v10", args)
    try:
        freeze = verify_freeze(context.root, context.config)
        candidates = verify_candidate_freeze(context.root, context.config)
        reports = build_reports(context.root, context.config, freeze, candidates)
        context.finish(
            "COMPLETED_V10_REPORT",
            source_freeze_digest=freeze["freeze_digest"],
            candidate_freeze_digest=candidates["freeze_digest"],
            reports=reports,
        )
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
