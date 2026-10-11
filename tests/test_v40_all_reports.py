"""Prevent a progress checkpoint from being labelled execution-complete."""
from scripts.build_v40_all_reports import ORDER, build


def test_all_reports_status_uses_complete_report_title(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for name in ORDER:
        (reports / name).write_text("# V40 report\n", encoding="utf-8")
    complete = reports / "V40_COMPLETE_REPORT.md"
    complete.write_text("# V40 — Progress Report (Not Final)\n\nV40 is not yet execution-complete.\n",
                        encoding="utf-8")
    assert "Progress checkpoint" in build(tmp_path).read_text(encoding="utf-8")
    complete.write_text("# V40 — Complete Report\n\nV40 is execution-complete.\n",
                        encoding="utf-8")
    assert "Execution-complete formal" in build(tmp_path).read_text(encoding="utf-8")
