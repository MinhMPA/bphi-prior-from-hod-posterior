# Project CLAUDE.md — bphi-prior-from-hod-posterior

## Local LLM reader: prioritize for all reading/summarizing

**Default to the local LM Studio reader for reading and summarizing files in this repo.** This is not a fallback — it is the *first* tool to reach for whenever the task is reading, parsing, extracting, ranking, or compressing.

### Applies to

- Python sources in `python_scripts/`, `notebooks/`, and any `.py`/`.ipynb` under the repo.
- Shell scripts (`*.sh`) including the launchers `bphi_prior_pipeline.sh`, `bphi_prior_pipeline_bfnl.sh`, `HOD_*.sh`, etc.
- LaTeX sources under `paper/` (`main.tex`, `references.bib`, included `.tex` fragments).
- Markdown notes, READMEs, CHANGELOGs, and any documents under `docs/`.
- Large logs, MCMC chain dumps, slurm output, `git diff`/`git log` output, and `ripgrep` results.
- External code that lives outside the repo but is being consulted for context, such as `~/desilike/`, `~/cosmoprimo/`, or `~/jaxPTPolyPol/`.

### How

Route through the existing commands before issuing any `Read`, `cat`, `head`, or `Grep` call that would load more than ~50 lines into Claude's context:

- `qdoc <files...>` — docs, notes, LaTeX, markdown.
- `qapi <files...>` — Python modules, when the goal is API/function extraction.
- `qdiff [base]` — git diff summaries.
- `qgrep` / `qlog` — large grep or log output piped in via stdin.
- `ask-local --paths <files...> --question "<specific question>"` — anything that doesn't fit the above.

After the local reader returns a compact summary, *then* `Read` only the specific file ranges identified as decision-critical (typically <100 lines per file).

### Hard rules

1. **Do not bulk-read `.py`, `.sh`, `.tex`, or `.md` files with the `Read` tool when more than 2 files or more than ~200 lines are needed.** Route through the local reader first.
2. **Do not paste large `git log`, `git diff`, `find`, `grep`, or chain-output blocks into the cloud context.** Pipe through `qdiff`/`qgrep`/`qlog`.
3. **Local reader output is lossy.** Treat its summaries as a map, not the territory. Before editing code, debugging, or making correctness claims, open the specific original file ranges directly.
4. **Cloud Claude remains responsible** for planning, editing, debugging, scientific judgment, and final correctness. The local model never writes code or edits files in this workflow.

### Failure mode to avoid

The local reader was skipped in the 2026-05-21 session despite the user requesting it. The trigger to skip is usually "this file is short, I'll just `Read` it" — but short reads accumulate, and the cumulative context cost is what matters. **When in doubt, route through the local reader.**

## Notation and physics conventions

- `bphi` ≡ $b_\phi$, the local-PNG bias (galaxy abundance response to long-wavelength $\phi$).
- `fnl_loc` ≡ $f_{\rm NL}^{\rm local}$.
- `B_i` ≡ $b_{\phi,i}\,f_{\rm NL}$ (per-bin product amplitude, what the power spectrum directly constrains).
- Mode strings used in the pipeline: `bphi` (sample $b_\phi$ and $f_{\rm NL}$ jointly), `bfnl` (sample only $B_i$, no separate $b_\phi$), `b-p` (fixed-$p$ universality prior).
- Production launcher: `python_scripts/bphi_prior_pipeline.sh` invokes `bphi_prior_pipeline.py` with `--ells 0 2` for all three modes. The legacy `bphi_prior_pipeline_bfnl.py` is monopole-only and not on the production path.
