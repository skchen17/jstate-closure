"""CLI wrapper for protocol-v6 machine report generation."""

from __future__ import annotations

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v6 import verify_freeze
from jclosure.reporting_v6 import build


def main() -> None:
    args = standard_parser(
        "Build v6 reports", "configs/peripheral_v6.yaml"
    ).parse_args()
    context = initialize_context("report-v6", args)
    try:
        freeze = verify_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        manifest = build(context.root)
        context.finish("COMPLETED", report_manifest=manifest)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
