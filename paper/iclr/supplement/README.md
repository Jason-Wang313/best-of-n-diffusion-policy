# Anonymous Supplement README

This supplement is intended for anonymous review. It should be packaged without author names, personal repository links, local absolute paths, acknowledgments, or institution-identifying metadata.

Recommended contents:

- `src/`: implementation of the finite selection law, controller, diagnostics, diffusion samplers, and benchmarks.
- `scripts/`: reproduction, audit, and paper-build scripts.
- `tests/`: unit and regression tests.
- `results/`: audited CSV/JSON tables and generated figures used by the paper.
- `docs/`: claim ledger, readiness notes, theory notes, and validity checklists.
- `paper/iclr/`: LaTeX source for the anonymous paper.

Primary reproduction commands from the repository root:

```bash
python scripts/build_iclr_paper.py
bash scripts/run_claim_audit.sh
python -m pytest -q
```

The package does not contain new experiments beyond the audited artifacts. The paper's promoted claims are valid only when `bash scripts/run_claim_audit.sh` passes.
