# seqsuite

Kellogg Lab tools for sequence-based assays. The code is split in two: shared
functions in `core/`, and one folder per assay under `apps/`, each with its own
README explaining how to run that pipeline.

**Documentation:** [API reference](docs/API.md) for every function in `core/`.
`docs/index.html` is a landing-page version of the same links, for a browser or
GitHub Pages.

## What's in this repository

```
core/                           Functions shared across the apps
  sequence.py                     Sequence and FASTA primitives: reverse complement, A/T content, alignment
  motifs.py                       Matching a sequence against a motif: IUPAC mismatches, PWMs
  guides.py                       Locating guide RNAs, PAMs and spacers in a genome
  insertions.py                   Analysis of an insertion TSV: on-target %, TSD motifs
  predict_insertion.py            Where a CAST is expected to insert for a given spacer
  plots.py                        Figures, kept apart from the analysis
  bbmap_workflow.py               The older bbmap insertion-calling workflow
  read_cbcl.py                    Reads Illumina base calls straight out of .cbcl files
  barcodes.py                     Error-tolerant barcode lookup for demultiplexing
  read_snapgene.py                SnapGene .dna -> sequence, topology and features
apps/                           One folder per assay, each with its own README
  ecoli_profiling/                CAST integration profiling in E. coli
  human_profiling/                UdiTaS: CAST insertion outcomes in human cells
  chipseq/                        Paired-end ChIP-seq
tests/                          Synthetic fixtures and regression tests
environment.yml                 Conda environment with all dependencies, for every app
```

That is the whole top level. Every function has exactly one definition, in
`core/`, and every pipeline step lives with the assay it belongs to.

## The apps

| App | Assay | Start here |
|---|---|---|
| [`apps/ecoli_profiling/`](apps/ecoli_profiling/README.md) | Where CAST integrates across an *E. coli* genome, from tagmentation sequencing | `map_junction_insertions.py` → `filter_donor_backbone.sh` → `plot_profile.sh` |
| [`apps/human_profiling/`](apps/human_profiling/README.md) | UdiTaS: CAST insertion outcomes in human cells, read straight from Illumina `.cbcl` | `verify_structure.py` → `recheck_inserts.py` → `extract_salvage.py` |
| [`apps/chipseq/`](apps/chipseq/README.md) | Paired-end ChIP-seq against a bacterial genome | `chipseq_pipeline.sh --help` |

The two profiling apps meet at one file format: `human_profiling`'s
`extract_salvage.py` writes exactly the input shape `ecoli_profiling`'s
`map_junction_insertions.py` consumes.

## Installation

```bash
git clone git@github.com:KelloggLab/seqsuite.git
cd seqsuite
conda env create -f environment.yml
conda activate seqsuite
```

One environment serves every app. It installs the command-line tools
(cutadapt, bwa, samtools, gnuplot, bbmap) and the Python libraries, and
installs this folder as a package so `from core import sequence`
works from any directory. `apps/chipseq/` needs three extra tools — see its
README.

**Upgrading from an older checkout:** the shared modules used to live in
`seqsuite/`, and before that in a package called `tagmentation`. Re-run the
editable install once in each environment you use, so `core` resolves:

```bash
pip install -e . --no-deps
```

Notebooks import `core` directly — `from core import guides`, and so on.
Older notebooks that imported this folder through a `TagmentationAnalysis`
symlink were rewritten; nothing depends on the symlinks or on the folder's name
any more.

## Analysing results in Python

```python
import pandas as pd
from core import sequence, guides, insertions, motifs, plots

tsv = pd.read_csv("mapped_reads/sample_merged.insertions.5p.nobackbone.tsv", sep="\t")
genome = sequence.load_genome("genome.fasta")
```

Each function has exactly one home, named for what it does rather than for the
workflow it came from.

Useful functions ([full API reference](docs/API.md)):

| Module | Function | What it does |
|---|---|---|
| `insertions` | `percent_within_distance` | % of insertions within ±N bp of a target position |
| | `top_n_insertions_frequencies` | Most frequent insertion positions |
| | `characterize_insertion_from_position` / `_from_spacer` | Relate insertions to the guide and nearby PAMs |
| | `extract_tsd_centered_motifs_5p` | Sequences around each insertion, centered on the target-site duplication |
| | `which`, `between` | Select rows at, or within, a coordinate range |
| `guides` | `map_guide_to_genome` | Find a guide RNA in the genome (either strand) |
| | `extract_spacers`, `find_pam_sites_with_optional_protospacer` | Find PAMs and candidate spacers (IUPAC motifs supported) |
| | `extract_sequence_windows` | Sequence context around a list of positions |
| `sequence` | `reverse_complement`, `load_genome`, `local_align` | The string and FASTA primitives |
| | `at_richness`, `at_sliding_window` | A/T content, whole-sequence or windowed |
| `motifs` | `build_log_odds_pwm`, `score_sequence_pwm` | Build and score position weight matrices |
| | `degenerate_hamming`, `iupac_match` | Count mismatches against an IUPAC motif |
| `plots` | `plot_insertion_profile` | Insertion histogram with the guide position marked |
| | `plot_insertion_histogram`, `plot_integration_profile` | The other two figures |
| `predict_insertion` | `predict_insertion_from_spacer` | Expected insertion site for a spacer, from measured spacing |

Reading a sequencing run, rather than analysing its output, is
`read_cbcl`, `barcodes` and `read_snapgene`; those three import numpy at most,
so the `python3 -I` scripts in `apps/human_profiling/` can use them.

## Tests

```bash
conda activate seqsuite
python -m unittest discover -s tests -v
```

302 tests, about 25 seconds. They build a synthetic 50 kb genome and a donor
plasmid, plant insertions at known positions, and check that
`ecoli_profiling`'s pipeline recovers every one of them and that the backbone
filter drops the backbone-derived reads while sparing the Cargo-derived ones —
plus unit tests for the functions in `core/`. A synthetic Illumina CBCL lane
covers `human_profiling` end to end.

The E. coli end-to-end tests skip cleanly if cutadapt, bwa and samtools are not
on PATH. The UdiTaS mapping tests need `pysam`, which builds its BAM fixtures
in memory rather than running bwa; everything else needs only numpy.
`apps/chipseq/`'s shell steps have no tests, though its plotting module does.
See [tests/README.md](tests/README.md).
