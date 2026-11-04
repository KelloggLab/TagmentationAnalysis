# Architecture: a three-application NGS suite

Target structure for this repository as it grows from one assay to three:
**E. coli genomic tagmentation**, **UdiTaS tagmentation**, and **ChIP-seq**.
Each application gets its own folder of scripts; everything they have in
common lives in one shared library.

This document describes the destination and the route. It is a plan, not a
description of the current code.

## Decisions

| Question | Decision |
|---|---|
| Layout | One repository: a shared library plus `apps/<application>/` |
| ChIP-seq pipeline | Port fully from bash to Python against the shared library |
| Per-experiment settings | A YAML config per run (plus the existing CSV sample sheet) |
| UdiTaS | Build on the shared library, folding in the existing `uditas_hpc` work |

## Why consolidate: what the three applications already share

Every row below is something implemented more than once today, in different
styles, with different bugs.

| Concern | Implemented today in | Belongs in |
|---|---|---|
| Finding and running external tools | `map_junction_insertions.py` `run()`; `filter_donor_backbone.sh` `find_tool()`; `bbmap_bin`/`samtools_bin` arguments threaded through `NGSanalysis_module`; bare PATH calls in `chipseq_pipeline.sh` | `external.py` |
| Making sure a reference is indexed | `ensure_bwa_index()`; the `.bwt`/`.fai` block in `chipseq_pipeline.sh`; the backbone-index cache in `filter_donor_backbone.sh`; `bbmap.sh ref=` in `process_reads()` | `refs.py` |
| Loading a FASTA | `load_genome()`, `load_fasta_as_dict()`, `fasta_to_string()` — three functions, two behaviours, two modules | `refs.py` |
| Trimming reads | `cutadapt_trim()`; `trim_pair()` (Trim Galore); `trim_reads_by_primer()` (hand-written string slicing) | `trim.py` |
| Align → sort → index | `bwa_map()`; `align_dedup_filter()`; the `bwa mem` in `filter_donor_backbone.sh`; bbmap in `process_reads()` | `align.py` |
| Reading alignments, CIGAR arithmetic | `insertion_coord()` + `ref_aln_span()` (pysam); `extract_insertion_sites()` + `aligned_ref_length()` (hand-rolled CIGAR parsing of `samtools view` text) | `bam.py` |
| Knowing the insertion-table columns | `map_junction_insertions.py` writes them; `TsvMapHelperFunctions` reads them; `filter_donor_backbone.sh` looks up `read_id` itself | `tables.py` |
| Plotting | `plot_insertion_histogram()`, `plot_integration_profile()`, `plot_insertion_profile()`, the matplotlib heredoc in `chipseq_pipeline.sh`, and `plot_profile.sh` in gnuplot — five paths, three styles, inconsistent return types | `plots.py` |
| Run statistics and reports | The "Integration Analysis Report" f-string, duplicated verbatim in `tagmentation_app.py` and the notebook; `report_read_counts()`; `print("[info] …")` | `report.py` |
| CAST insertion geometry | `INSERTION_DISTANCE`/`TARGET_SITE_WINDOW` in `NovoCAST_helper.py`, and the same `71`/`10` written inline again in `TsvMapHelperFunctions.characterize_insertion_from_spacer` | `apps/ecoli_tagmentation` |

The deeper overlap is conceptual: **E. coli tagmentation finds donor→genome
junctions, and UdiTaS finds genome→genome junctions.** Both ask where a read
stops matching one reference and starts matching another. One `junctions.py`
serves both.

The existing `uditas_hpc/salvage/extract_salvage.py` already writes per-sample
FASTQs with the primer and transposon end trimmed off — exactly the input
`map_junction_insertions.py` takes. The two applications meet at that file.

## Target layout

```
<repo>/
├── ngslib/                     the shared library — one import root
│   ├── external.py             tool resolution + subprocess running + logging
│   ├── config.py               load/validate run YAML and the CSV sample sheet
│   ├── refs.py                 FASTA loading, bwa index, faidx, genome size
│   ├── illumina.py             run folders, CBCL reading, barcode demultiplexing
│   ├── trim.py                 cutadapt and Trim Galore wrappers
│   ├── umi.py                  UMI extraction, grouping, deduplication
│   ├── align.py                bwa/bbmap → sorted, indexed BAM
│   ├── bam.py                  filtered iteration, CIGAR spans, 5'/3' reference ends
│   ├── junctions.py            breakpoints from primary + supplementary alignments
│   ├── sequence.py             reverse complement, IUPAC, A/T content, windows
│   ├── motifs.py               PWM build/score, degenerate motif scanning
│   ├── targets.py              guide, PAM, spacer and protospacer finding
│   ├── tables.py               insertion/outcome table schemas: read, write, validate
│   ├── plots.py                one house style + the shared figure types
│   └── report.py               run reports as text, JSON and PDF
├── apps/
│   ├── ecoli_tagmentation/
│   ├── uditas/
│   └── chipseq/
├── configs/                    example run configs and sample sheets
├── tests/                      synthetic fixtures + regression tests
├── docs/
└── legacy/                     the bbmap notebook and Streamlit app, frozen
```

### Who uses what

| Module | E. coli | UdiTaS | ChIP-seq |
|---|:--:|:--:|:--:|
| `external`, `config`, `refs`, `report` | ● | ● | ● |
| `trim`, `align`, `bam` | ● | ● | ● |
| `plots`, `tables` | ● | ● | ● |
| `junctions` | ● | ● | |
| `sequence`, `motifs`, `targets` | ● | ● | |
| `illumina`, `umi` | | ● | |

Nothing in `ngslib/` knows which assay is calling it. Anything that encodes
assay biology — donor backbones, target-site duplications, CAST spacing,
peak calling, UdiTaS outcome classes — lives in `apps/`.

## The applications

### `apps/ecoli_tagmentation/`

Genome-wide insertion profiling in E. coli. Mostly a re-homing of today's
`pipeline/`, `scripts/` and `seqsuite/` code.

| Script | From | Notes |
|---|---|---|
| `map_insertions.py` | `pipeline/map_junction_insertions.py` | trimming/alignment/BAM parsing move to `ngslib` |
| `filter_donor_backbone.py` | `pipeline/filter_donor_backbone.sh` | ported to Python; the GenBank Cargo parsing stays here |
| `profile_insertions.py` | `scripts/plot_profile.sh` + `plot_insertion_profile()` | replaces the gnuplot script, so figures match the rest |
| `characterize_insertions.py` | `TsvMapHelperFunctions` | TSD motifs, on-target %, distance from PAM |
| `cast_geometry.py` | `NovoCAST_helper` | the 71 bp / 10 bp constants, defined once |

### `apps/uditas/`

Targeted sequencing of editing outcomes. This folds in the existing
`~/Projects/MD_DNA/uditas_hpc/` work, which is further along than the rest of
the suite and currently lives outside any repository.

| Script | From | Notes |
|---|---|---|
| `qc_run.py` | `cbcl_qc/verify_structure.py`, `recheck_inserts.py` | one QC command with subcommands; CBCL reading moves to `ngslib/illumina.py`. The other nine exploratory scripts were deleted as run-specific — see the history if a diagnostic is wanted back |
| `extract_reads.py` | `cbcl_qc/extract_salvage.py` | demultiplex, trim primer + transposon end, emit per-sample FASTQ |
| `map_junctions.py` | new, but shares `ngslib.junctions` with E. coli | UMI-deduplicated |
| `classify_outcomes.py` | new | indels, large deletions, inversions, translocations between cut sites |
| `qc_report.py` | new — `cbcl_qc/make_report.py` was deleted as run-specific, but its page/table/chart scaffolding is worth reading in the history | uses `ngslib.report` |

Two things must be fixed on the way in: the sample barcodes and LE/RE primers
currently hardcoded in `verify_structure.py` move into the sample sheet, and
the run-specific cycle layout becomes config rather than a constant.

New capability needed in the library: **`umi.py`**, since nothing in the
repository handles UMIs today, and the UdiTaS UMI sits in the 18-cycle I2
read.

### `apps/chipseq/`

Ported from `chipseq_pipeline.sh` to Python, per the decision above.

| Script | Covers |
|---|---|
| `run_pipeline.py` | trim → align → fixmate/markdup → filter, over 1–3 replicates |
| `call_peaks.py` | MACS3 in BAMPE mode, fold-enrichment tracks, bedtools consensus |
| `plot_enrichment.py` | replaces the embedded matplotlib heredoc |

MACS3 and bedtools stay as external command-line tools, invoked through
`ngslib.external`; what gets ported is the orchestration, argument handling
and reporting. The bash version stays in `legacy/` until the Python one
reproduces its output on a real dataset.

## Configuration

Two files per experiment, both committed next to the results:

**`run.yaml`** — everything about the run: reference genome, output directory,
threads, tool overrides, and the app-specific parameters (donor GenBank file,
adapter sequences, MAPQ thresholds, cycle layout).

**`sample_sheet.csv`** — one row per sample. The existing `sample_info.csv`
from `uditas_hpc` is already the right shape, carrying barcodes, primer,
genome, guide sequences and coordinates, donor and genomic background; it
works unchanged for UdiTaS and trims down for the other two applications.

Scripts take `--config run.yaml` and read the sample sheet named in it. The
config file is then the record of how a result was produced — rerunnable, and
shareable with anyone who asks how the numbers were made.

## Migration

Five phases. Each one ends with a working repository, so the work can stop
between any two of them.

| Phase | Work | Risk |
|---|---|---|
| 0 | Create `ngslib/`; move the pure functions (`sequence`, `motifs`, `targets`) with aliases left behind | Low — no I/O, no behaviour change |
| 1 | `external`, `refs`, `align`, `bam`, `trim`; rewire E. coli tagmentation onto them | Medium — verify insertion calls are unchanged on a fixture |
| 2 | `tables`, `plots`, `report`; retire the gnuplot script and the ChIP-seq heredoc | Low — figures change appearance by design |
| 3 | Port ChIP-seq to Python; keep the bash version in `legacy/` until outputs match | Medium — compare peak sets on a real run |
| 4 | Fold in `uditas_hpc`: `illumina`, `umi`, outcome classification | Highest — mostly new code |

**Before Phase 1, build `tests/` with small synthetic fixtures** — a few
thousand reads against a toy genome, with known insertion positions. Every
later phase is then checked by "the same input still gives the same table",
which is the only practical way to refactor analysis code without quietly
changing results.

## Compatibility

Seven folders reach this repository through symlinks named
`TagmentationAnalysis`, and import `TagmentationAnalysis.postprocessHelpers`
and friends. That keeps working if:

- the local folder name stays `TagmentationAnalysis` (symlinks point at a
  path, not at the GitHub repository name, so renaming on GitHub is free);
- the top-level forwarding modules stay until those notebooks are updated;
- `tagmentation/` remains as an alias package over `seqsuite/` (and later
  `ngslib/`) through the migration, and is removed only when nothing imports
  it. It is a four-line `sys.modules` alias, so both
  `from tagmentation import postprocessHelpers` and
  `from seqsuite import postprocessHelpers` return the same module object.

The package was renamed `tagmentation` -> `seqsuite` without touching the
local folder name, exactly because of those symlinks.

## Open questions

1. ~~**Names.**~~ **Decided.** The repository and import root are `seqsuite`
   -- general enough for any sequence-based assay, and checked against PyPI,
   GitHub and the literature before being adopted. The import root moved
   first, with `tagmentation/` left as an alias; the GitHub rename redirects
   old URLs. The **local folder stays `TagmentationAnalysis`**, because seven
   notebook folders symlink to that path. Rename it only after those
   symlinks are repointed.
2. **UdiTaS outcome categories.** Matching the published UDiTaS categories
   makes results comparable with the literature; defining our own fits CAST
   insertion outcomes better. This decides the `classify_outcomes.py` schema.
3. **`tn5_offset=5`** in `NGSanalysis_module.extract_insertion_sites`. For
   CAST junction reads the genomic 5′ end is already the insertion point. This
   should be settled before that logic is reused anywhere.
4. **Where ChIP-seq peak data lands.** The other two applications produce
   per-read tables; ChIP-seq produces intervals. `tables.py` should either
   cover BED/narrowPeak too, or deliberately not.
