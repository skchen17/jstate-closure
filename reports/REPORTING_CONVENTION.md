# Version reporting convention

Starting with V11, every completed experimental version must keep both:

1. its focused standalone reports, and
2. one canonical single-file bundle at `reports/Vxx_COMPLETE_REPORT.md`.

Generate the bundle only after all reports and machine-readable records for the version are
finalized:

```bash
/home/user/anaconda3/bin/python scripts/build_complete_version_report.py Vxx
```

The bundle must contain the version-specific section from `reports/FINAL_REPORT.md`, the
full text of every `reports/*_Vxx.md` standalone report, and SHA256 indexes for both report
sources and `results/vxx/processed` machine records. Existing standalone reports and prior
version bundles must not be deleted or overwritten by a later version.
