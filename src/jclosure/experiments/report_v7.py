from __future__ import annotations

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v7_corrective import verify_freeze
from jclosure.reporting_v7 import build


def main() -> None:
    parser = standard_parser(
        "build protocol-v7 reports", "configs/persistent_channels_v7_corrective.yaml"
    )
    args = parser.parse_args()
    context = initialize_context("report-v7", args)
    try:
        freeze = verify_freeze(context.root, context.config)
        manifest = build(context.root)
        context.finish(
            "COMPLETED_REPORTS",
            source_freeze_digest=freeze["freeze_digest"],
            report_manifest=manifest,
        )
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
