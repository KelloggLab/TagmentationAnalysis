# Reorganization plan

Goal: `core/` holds code shared between assays. `apps/` holds one folder per
assay, each with its own README explaining how to run that pipeline. Nothing
else at the top level.

## Target layout

    core/                       shared, assay-agnostic
      postprocessHelpers.py       guide/PAM/spacer mapping, alignment, FASTA
      TsvMapHelperFunctions.py    arithmetic on an insertion table
      NGSanalysis_module.py       primer trimming, bwa/samtools wrappers
      NovoCAST_helper.py          predicted insertion site for a spacer
      illumina.py                 NEW: CBCL reader (today copied 3x)
      barcodes.py                 NEW: one-mismatch demultiplexing (today 3x)
      snapgene.py                 was apps/uditas/cbcl_qc/read_snapgene.py

    apps/
      ecoli_profiling/          E. coli tagmentation integration profiling
        map_junction_insertions.py
        filter_donor_backbone.sh
        plot_profile.sh
        README.md
      human_profiling/          UdiTaS in human cells
        run.yaml                  cycle layout + arm sequences for a run
        sample_sheet.csv          barcode -> sample/arm
        runconfig.py              loads and validates the two above
        verify_structure.py
        recheck_inserts.py
        extract_salvage.py
        results/261030/           the three committed result JSONs
        README.md
      chipseq/
        chipseq_pipeline.sh
        README.md

    docs/      tests/
    postprocessHelpers.py  TsvMapHelperFunctions.py  NGSanalysis_module.py
                                three 8-line shims -- see "Notebooks" below

## Deleted

| Folder | Why |
|---|---|
| `legacy/` | Streamlit app + old notebook. You asked to remove it. |
| `tagmentation/` | A 7-line alias shim, nothing else. Dead. |
| `seqsuite/` | Its four modules move to `core/`; the folder goes. |
| `pipeline/` | Contents are E. coli-specific -> `apps/ecoli_profiling/`. |
| `scripts/` | Only `plot_profile.sh`, which reads the E. coli profile format -> `apps/ecoli_profiling/`. |
| `chipseq/` (top level) | -> `apps/chipseq/`. |
| `apps/uditas/` | -> `apps/human_profiling/`, flattened (the `cbcl_qc/` nesting buys nothing for 5 files). |

## Notebooks: the one thing that can break

Your notebooks under `~/Projects/analyze_tagmentation` import this repo
through its symlink, 75 times:

    import TagmentationAnalysis.postprocessHelpers as helper
    from TagmentationAnalysis import TsvMapHelperFunctions as tsvhp
    from TagmentationAnalysis import NGSanalysis_module as ngs

Those work because three 8-line shim files sit at the repo root and forward to
`seqsuite/`. Plan: keep the three shims, repoint each to `core/`, one line each.
Every notebook keeps working with no edit. The alternative -- editing 75 import
sites across ~30 notebooks outside this repo -- is not worth it.

## Packaging

Nothing imports an installed `seqsuite` package; the notebooks reach the code
through the symlink. So drop `packages = [...]` from `pyproject.toml` and `-e .`
from `environment.yml`, leaving pyproject to declare dependencies only. One
manual step afterwards: `pip uninstall seqsuite`.

## Order of work

Each step ends with the full test suite green and one commit.

1. **Delete.** `legacy/`, `tagmentation/`, root `postprocessHelpers` shim stays.
   Smallest step, nothing depends on what goes.
2. **Create `core/`** by `git mv`-ing the four `seqsuite/` modules into it.
   Fix the three root shims, `tests/test_sequence_functions.py`,
   `tests/test_tsv_analysis.py`, `docs/API.md`. Remove `seqsuite/`.
3. **Move the three apps** with `git mv` into `apps/ecoli_profiling/`,
   `apps/human_profiling/`, `apps/chipseq/`. Update the 8 path constants in
   `tests/` that point at `pipeline/` and `apps/uditas/cbcl_qc/`.
4. **Deduplicate the CBCL reader.** `read_header`, `decode_tile`, `read_block`,
   `one_mismatch_map`, `mism`, `rc` and `BASES` are copied verbatim into
   `verify_structure.py`, `recheck_inserts.py` and `extract_salvage.py`.
   Extract to `core/illumina.py` + `core/barcodes.py`; move `read_snapgene.py`
   to `core/snapgene.py`. This is the only step that changes code rather than
   paths, so it goes last among the moves, on a green suite.
5. **Write the three app READMEs** -- how to run that pipeline, what it needs
   installed, what it outputs. Replace `docs/ARCHITECTURE.md` (now stale: it
   plans an `ngslib/` that will not exist) with a short map of the new layout.
6. **Packaging + a final full-suite run.**

## Decisions I'd like your call on

1. **`TsvMapHelperFunctions.py` and `NovoCAST_helper.py` in `core/`?**
   Today only E. coli work uses them, so by the letter of "shared between apps"
   they belong in `apps/ecoli_profiling/`. I put them in `core/` because the
   insertion table they operate on is not E. coli-specific and UdiTaS can
   produce the same table, and because notebooks import them as general tools.
   Say the word and they move into the app instead.
2. **The three result JSONs** (`qc_stats.json`, `recheck_inserts.json`,
   `salvage_summary.json`) are output from run 261030, committed to git. I'd
   keep them under `apps/human_profiling/results/261030/` because the README
   cites their numbers. Deleting them is also defensible.
3. **Still open from before:** `recheck_inserts.py` now reads 220 R2 cycles
   where it used to read 200, so its numbers may no longer match the committed
   `recheck_inserts.json`. Needs a yes/no from you on which is right.
