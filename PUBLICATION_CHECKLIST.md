# Publication Checklist

- [ ] Replace no synthetic IDs with real station names.
- [ ] Keep raw engineering and operating data outside the repository.
- [ ] Remove trained weights, normalization files, logs, and generated schedules.
- [ ] Run `python scripts/privacy_scan.py .` and review every finding.
- [ ] Inspect staged files with `git diff --cached --stat` and `git diff --cached`.
- [ ] Inspect the complete Git history before making the repository public.
- [ ] Select and add an open-source license approved by the authors and institution.
- [ ] Confirm that third-party dependencies and data licenses permit redistribution.

