# Repository layout

Two ideas, and that is all:

- **`core/`** holds functions shared across assays.
- **`apps/<assay>/`** holds the code for one assay, with a README explaining
  how to run that pipeline.

Nothing else sits at the top level except the tests, the docs, and
`environment.yml`, which serves every app.

## `core/`

| Module | What it holds |
|---|---|
| `sequence.py` | Sequence and FASTA primitives: reverse complement, loading, A/T content, local alignment |
| `motifs.py` | Matching a sequence against a motif: IUPAC mismatch counting, and position weight matrices |
| `guides.py` | Locating guide RNAs, PAMs and spacers in a genome |
| `insertions.py` | Analysis of an insertion TSV: on-target percentage, target-site duplications |
| `predict_insertion.py` | Where a CAST is expected to insert for a given spacer |
| `plots.py` | Figures |
| `bbmap_workflow.py` | The older bbmap-based insertion-calling workflow |
| `read_cbcl.py` | Reads Illumina base calls out of a run folder's `.cbcl` files |
| `barcodes.py` | Error-tolerant barcode lookup, reverse complement, mismatch count |
| `read_snapgene.py` | SnapGene `.dna` -> sequence, topology and features |

Every function is documented in [API.md](API.md).

Each module is named for what it does, not for the workflow it came from. The
four original modules were `postprocessHelpers`, `TsvMapHelperFunctions`,
`NovoCAST_helper` and `NGSanalysis_module` -- names that said when the code ran,
or which project it was written for, rather than what it computes. They were
split along the lines above, which also separated figures from the analysis
that feeds them and moved `integration_finder` out of `core/` entirely: its
signature takes the ccdB gene's coordinates, so it is E. coli-specific and now
lives in `apps/ecoli_profiling/ccdb_integration.py`.

`read_cbcl.py`, `barcodes.py` and `read_snapgene.py` import numpy at most,
deliberately: the `apps/human_profiling` scripts run under `python3 -I` and
must read a sequencing run without pandas, matplotlib or Biopython. That is
also why `barcodes.rc` exists alongside `sequence.reverse_complement`.

Two pairs still overlap, kept because notebooks call both:
`guides.sgRNA_finder` and `guides.map_guide_to_genome` both locate a guide
(the first returns bare integers and prints on failure), and
`sequence.fasta_to_string` and `sequence.load_genome` both read a one-record
FASTA.

## `apps/`

| App | Assay | Language | Tests |
|---|---|---|---|
| `ecoli_profiling/` | Where CAST integrates across an *E. coli* genome | Python + bash | 37, end to end against planted insertions |
| `human_profiling/` | UdiTaS: CAST insertion outcomes in human cells | Python | 98, against a synthetic CBCL lane |
| `chipseq/` | Paired-end ChIP-seq against a bacterial genome | bash | none |

The two profiling apps meet at one file format: `human_profiling`'s
`extract_salvage.py` writes exactly the input shape `ecoli_profiling`'s
`map_junction_insertions.py` consumes.

## How notebooks import this

Through the installed package, from any directory:

    from core import guides, insertions, sequence

An editable install (`pip install -e . --no-deps`) is needed once per
environment. Earlier, the lab's notebooks instead reached this repository
through a symlink named `TagmentationAnalysis` and imported it as a package,
which three eight-line forwarding modules at the repository root served. All
86 of those import sites, across 45 notebooks and scripts under
`~/Projects/analyze_tagmentation` and two folders on the Desktop, were
rewritten to the form above, and the forwarders were deleted. Nothing now
depends on the symlinks or on this folder's name.

One wrinkle worth knowing: `core` is a generic top-level import name. If an
environment ever acquires another package called `core`, whichever is found
first on `sys.path` wins. Nothing in the current environments collides.

## What has not been done

- **`apps/chipseq/` has no tests.** It should get a small fixture before it is
  relied on for a figure.
- **One open question on `recheck_inserts.py`:** it now reads 220 R2 cycles
  where it used to read 200, so an adapter between cycles 200 and 220 is found
  rather than counted absent. Its numbers may no longer match the committed
  `results/261030/recheck_inserts.json`.
- **The LE still-donor check is inactive** for want of donor sequence, not for
  want of code. See `apps/human_profiling/README.md`.
