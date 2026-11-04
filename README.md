# TagmentationAnalysis

Kellogg Lab tools for mapping where CRISPR-associated transposons (CAST) integrate, using tagmentation sequencing data.

Each read starts in the transposon end (LE or RE) and reads into the flanking genomic DNA. The tools here trim off the transposon sequence, map what's left to the reference genome, and report one insertion coordinate per read. Further tools filter out donor-plasmid artifacts, plot insertion profiles, and analyze sequence preferences around insertion sites (guide/PAM position, A/T richness, target-site duplications, position weight matrices).

**Documentation:** [API reference](docs/API.md) for every function in the `tagmentation` package. `docs/index.html` is a landing page version of the same links, for viewing in a browser or serving with GitHub Pages.

## What's in this repository

```
pipeline/                       Command-line steps, run in order
  map_junction_insertions.py      1. trim transposon end -> map -> per-read insertion TSV
  filter_donor_backbone.sh        2. drop reads that come from the donor plasmid backbone
scripts/
  plot_profile.sh                 Quick genome-wide insertion plot (gnuplot)
tagmentation/                   Python helper functions (import in your own notebooks)
  postprocessHelpers.py           Sequence utilities: guide/PAM finding, A/T richness, spacers, PWMs
  TsvMapHelperFunctions.py        Analysis of insertion TSVs: on-target %, TSD motifs, profiles
  NovoCAST_helper.py              Predicted insertion sites for novel CAST systems
  NGSanalysis_module.py           Functions behind the legacy bbmap workflow
legacy/                         Older bbmap-based workflow (still works, no longer the main path)
  CAST_TagmentationNotebook.ipynb
  tagmentation_app.py             Streamlit web front end
chipseq/
  chipseq_pipeline.sh             Paired-end ChIP-seq pipeline (separate from tagmentation)
apps/
  uditas/                         UdiTaS: CBCL-level library QC and junction-read salvage
tests/                          Synthetic fixtures and regression tests
environment.yml                 Conda environment with all dependencies
```

The `postprocessHelpers.py`, `TsvMapHelperFunctions.py` and `NGSanalysis_module.py` files at the top level only forward to the copies in `tagmentation/`. They're there so older notebooks that use `import TagmentationAnalysis.postprocessHelpers` keep working, and they can be deleted once those notebooks are updated.

## Installation

```bash
git clone git@github.com:KelloggLab/TagmentationAnalysis.git
cd TagmentationAnalysis
conda env create -f environment.yml
conda activate analyze_tagmentation
```

This installs the command-line tools (cutadapt, bwa, samtools, gnuplot, bbmap) and the Python libraries. It also installs this folder as the `tagmentation` Python package, so `from tagmentation import postprocessHelpers` works from any directory.

## Main workflow

### 1. Call insertion sites: `pipeline/map_junction_insertions.py`

Input: a merged (single-end) FASTQ and the reference genome FASTA.

```bash
python pipeline/map_junction_insertions.py \
    --fastq sample_merged.fastq \
    --ref genome.fasta \
    --outdir mapped_reads \
    --donor-seq <transposon-end sequence adjacent to the junction> \
    --donor-side 5p \
    --primary-only
```

This script:
1. Uses **cutadapt** to trim the donor sequence and drop reads that don't contain it (`--error-rate`, default 0.1; `--min-len`, default 20 bp). By default the *whole* `--donor-seq` must match; `--min-overlap` relaxes that if your reads genuinely start part-way into the donor. Don't lower it without reason — cutadapt's own default of 3 bases lets reads with no junction at all through as insertion calls.
2. Maps the remaining genomic sequence with **bwa mem**, then sorts and indexes it with **samtools**. The bwa index is built automatically if it's missing.
3. Calls one insertion coordinate per read from each alignment, taking clipping and indels into account. Alignments with MAPQ below `--min-mapq` (default 20) are skipped.

`--donor-side 5p` means the donor sequence is at the read's 5′ end, and the insertion is called at the read's genomic 5′ end. Use `3p` if the donor is at the 3′ end.

Outputs in `--outdir`, named after the input FASTQ:

| File | Contents |
|---|---|
| `<name>.trimmed.fastq` | Reads with the donor sequence removed |
| `<name>.mapped.sorted.bam` (+ `.bai`) | Alignments to the reference |
| `<name>.insertions.5p.tsv` | One row per read: `read_id, ref, ins0, ins1, strand, mapq, cigar, aln_start0, aln_end0_excl` |

`ins0` is the 0-based insertion coordinate and `ins1` is the 1-based one.

### 2. Remove donor-backbone reads: `pipeline/filter_donor_backbone.sh`

Some reads that survive trimming come from the donor plasmid's backbone (ori, resistance marker, etc.) rather than from genomic DNA. These are leftover donor plasmid or a pre-existing backbone integration, not new insertions. This step aligns the trimmed reads to the backbone and removes any read with at least `-m` bp aligned (default 30).

The donor GenBank file must have a `misc_feature` labelled **Cargo** spanning `1..N`. Everything after position N is treated as backbone. The Cargo itself is deliberately not screened, because real insertions carry Cargo sequence up to the transposon end.

```bash
pipeline/filter_donor_backbone.sh \
    -g donor_plasmid.gb \
    -f mapped_reads/sample_merged.trimmed.fastq \
    -t mapped_reads/sample_merged.insertions.5p.tsv \
    -o mapped_reads/sample_merged.insertions.5p.nobackbone.tsv
```

Run it once per FASTQ/TSV pair. The backbone index is cached next to the GenBank file and reused for later samples. Set `BWA=` or `SAMTOOLS=` to use executables that aren't on your PATH.

### 3. Look at the results

**Quick plot:** `scripts/plot_profile.sh` reads an insertion TSV (position in column 3, strand in column 5) and plots + and − strand counts side by side across the genome. The default bin size is 1000 bp. Use `-f` to plot each bin as a fraction of all reads.

```bash
scripts/plot_profile.sh -f mapped_reads/sample_merged.insertions.5p.nobackbone.tsv 500
```

**Detailed analysis in Python:**

```python
import pandas as pd
from tagmentation import postprocessHelpers as hp
from tagmentation import TsvMapHelperFunctions as tsvhp

tsv = pd.read_csv("mapped_reads/sample_merged.insertions.5p.nobackbone.tsv", sep="\t")
genome = hp.load_genome("genome.fasta")
```

Useful functions ([full API reference](docs/API.md)):

| Module | Function | What it does |
|---|---|---|
| `TsvMapHelperFunctions` | `percent_within_distance` | % of insertions within ±N bp of a target position |
| | `top_n_insertions_frequencies` | Most frequent insertion positions |
| | `plot_insertion_profile` | Insertion histogram with the guide position marked |
| | `characterize_insertion_from_position` / `_from_spacer` | Relate insertions to the guide and nearby PAMs |
| | `extract_tsd_centered_motifs_5p` | Sequences around each insertion, centered on the target-site duplication |
| `postprocessHelpers` | `map_guide_to_genome` | Find a guide RNA in the genome (either strand) |
| | `extract_spacers`, `find_pam_sites_with_optional_protospacer` | Find PAMs and candidate spacers (IUPAC motifs supported) |
| | `extract_sequence_windows`, `at_richness`, `at_sliding_window` | Sequence context and A/T content around sites |
| | `build_log_odds_pwm`, `score_sequence_pwm` | Build and score position weight matrices |
| `NovoCAST_helper` | `predict_insertion_from_spacer` | Predicted insertion site for a spacer (novel CAST systems) |

## Legacy workflow (bbmap)

`legacy/CAST_TagmentationNotebook.ipynb` is the original pipeline. It trims reads at a primer, removes donor reads by their first ten bases, maps with **bbmap**, and reports on-target insertions in a ccdB target gene. `legacy/tagmentation_app.py` is a Streamlit front end for the same steps.

- **Notebook:** open it in Jupyter, edit the parameters in the "Step Three" cell (file paths, primer, donor bases, sgRNA, and the `bbmap_bin`/`samtools_bin` folders), then run every cell.
- **Web app:** `streamlit run legacy/tagmentation_app.py`. It uses `bbmap.sh` and `samtools` from your PATH, or from the folders in `BBMAP_BIN` and `SAMTOOLS_BIN`. Output goes to `TAGMENTATION_OUTPUT_DIR` (default `tagmentation_app_output/`).

## UdiTaS

`apps/uditas/` holds the UdiTaS (UniDirectional Targeted Sequencing) work for
CAST insertion outcomes in human cells. It reads Illumina `.cbcl` files
directly, with no bcl2fastq step, to check library structure and to salvage
junction reads from a donor-dominated run. It is currently an as-is import,
written for one specific run; see [apps/uditas/README.md](apps/uditas/README.md).

## ChIP-seq pipeline

`chipseq/chipseq_pipeline.sh` is a separate end-to-end pipeline for paired-end ChIP-seq with 1–3 replicates. The steps are Trim Galore → bwa → MACS3 BAMPE peak calling → consensus peaks → fold-enrichment plots. It needs extra tools that aren't in `environment.yml`:

```bash
conda install -c bioconda -c conda-forge trim-galore macs3 bedtools
```

Run `chipseq/chipseq_pipeline.sh --help` for usage.

## Tests

```bash
conda activate analyze_tagmentation
python -m unittest discover -s tests -v
```

111 tests, about 18 seconds, no extra dependencies. They build a synthetic
50 kb genome and a donor plasmid, plant insertions at known positions, and
check that `pipeline/map_junction_insertions.py` recovers every one of them and
that `pipeline/filter_donor_backbone.sh` drops the backbone-derived ones while
sparing the Cargo-derived ones — plus unit tests for the helper functions in
`tagmentation/`. The end-to-end tests skip cleanly if cutadapt/bwa/samtools are
not on PATH. See [tests/README.md](tests/README.md).
