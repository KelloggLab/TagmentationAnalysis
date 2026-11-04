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
| `postprocessHelpers.py` | Sequence utilities: guide/PAM finding, spacer extraction, A/T richness, sequence windows, position weight matrices |
| `TsvMapHelperFunctions.py` | Analysis of an insertion TSV: on-target percentage, target-site duplications, insertion profiles |
| `NovoCAST_helper.py` | Predicted insertion site for a spacer, for novel CAST systems |
| `NGSanalysis_module.py` | Primer trimming and bbmap/samtools wrappers, from the older workflow |
| `illumina.py` | Reads Illumina base calls out of a run folder's `.cbcl` files |
| `barcodes.py` | Error-tolerant barcode lookup, reverse complement, mismatch count |
| `snapgene.py` | SnapGene `.dna` -> FASTA plus a feature table |

Every function in the first four modules is documented in [API.md](API.md).

`illumina.py` and `barcodes.py` import numpy and nothing else, deliberately:
the `apps/human_profiling` scripts run under `python3 -I` and read a
sequencing run without pandas, matplotlib or Biopython. That is also why
`barcodes.rc` exists alongside `postprocessHelpers.reverse_complement` —
the latter is in a module that pulls in all three.

`TsvMapHelperFunctions` and `NovoCAST_helper` are used today only by the
E. coli work, so by the letter of "shared" they could live in that app. They
are here because the insertion table they operate on is not E. coli-specific —
`human_profiling` can produce the same table — and because notebooks import
them as general tools alongside `postprocessHelpers`.

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

    from core import postprocessHelpers as hp
    from core import TsvMapHelperFunctions as tsvhp

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
