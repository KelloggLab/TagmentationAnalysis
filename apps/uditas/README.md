# UdiTaS (apps/uditas)

UniDirectional Targeted Sequencing for CAST insertion outcomes in human cells.

**This folder is a verbatim import**, copied from `~/Projects/MD_DNA/uditas_hpc`
on 2026-11-03 so the work is under version control before it gets refactored.
Nothing has been rewritten, renamed or rewired yet. The intended destination is
in [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): these scripts collapse
into five commands over a shared `ngslib/`, and the CBCL reading moves to
`ngslib/illumina.py`.

Written for one run: **`261030_VH00889_534_AAJLG3JM5`** (NextSeq 2000, P1,
1 lane, R1 301 / I1 8 / I2 18 / R2 301; cycles R1 1–301, I1 302–309,
I2 310–327, R2 328–628), 10.3 M clusters over 3 sampled tiles. Eight samples:
two cell lines (FT, 293) × two i5 (N501, N507) × two i7, where the i7 selects
the transposon arm — N701/N703 carry primer 251 (RE) and N702/N704 primer 253
(LE). `extract_salvage.py` pools the two i5 replicates, so its output is four
groups: `{293T,FT}_{LE,RE}`. The run-specific cycle layout and the sample
barcodes are **hardcoded**, mostly in `cbcl_qc/verify_structure.py`; moving
them into the sample sheet is the first fix on the way to a reusable pipeline.

## What happened in this run

Most of the library turned out to be donor plasmid rather than insertion
junctions: the Tn5 mosaic end sits too close to the LE/RE primer in **72% of
assigned reads**, leaving no genomic flank. It is worse on the LE arm (88.5%)
than on RE (66–69%). `extract_salvage.py` recovers what is left — **792,246
reads kept of 8,475,767 assigned (9.3%)**:

| Group | Assigned | Adapter too close | Kept | UMI-collapsed |
|---|---:|---:|---:|---:|
| `293T_LE` | 1,027,883 | 88.5% | 117,569 | 113,769 |
| `293T_RE` | 3,576,491 | 68.7% | 323,607 | 312,166 |
| `FT_LE` | 906,600 | 88.7% | 101,780 | 96,566 |
| `FT_RE` | 2,964,793 | 65.8% | 249,290 | 243,122 |

`Run261030_library_QC.pdf` is the shareable write-up of that finding, and
`donor_layout.py` / `design_helper.py` exist to redesign the arm primers and a
blocking clamp so the next library avoids it.

## Layout

```
bcl2fastq_uditas.lsf        LSF job: BCL -> Undetermined FASTQs (16 cores, 12 h)
sample_info.csv             Sample sheet: i7/i5 barcodes, primer, genome, guides, donor
Run261030_library_QC.pdf    The QC report that came out of make_report.py
cbcl_qc/                    Reads .cbcl directly, with no bcl2fastq step
salvage/                    Output dir for extract_salvage.py (FASTQs not tracked)
```

### `cbcl_qc/` scripts

All of these decode cycles straight out of `L001_1.cbcl` (surface 1 only) for
the first N tiles, so they *sample* the run rather than scanning it. Run each
with `python3 -I`, passing the lane directory. numpy only, except
`make_report.py`, which needs matplotlib.

| Script | What it answers |
|---|---|
| `verify_structure.py` | Consensus R2 prefix per sample; holds the cycle layout, barcodes and primers that the others import |
| `verify2.py` | Primer match rate, no-call rate, where the mosaic end appears |
| `verify3.py` | Top distinct R2 starts — is the LE/RE end one variant or a mixture? |
| `verify4.py` | What sits between the primer and the mosaic end |
| `verify5.py` | Are the no-mosaic-end reads diverse (genomic) or constant (donor)? |
| `cbcl_index_peek.py` | i7/i5 orientation, and where the UMI sits in the 18-cycle I2 |
| `recheck_inserts.py` | Primer→adapter distance again, tolerating sequencing errors |
| `donor_fraction.py` | First pass at the donor fraction, inferring the terminus from the data |
| `donor_fraction2.py` | Same, but taking donor sequence and terminus from the SnapGene map |
| `donor_layout.py` | Places the LE/RE primers on the donor map; picks a blocking site |
| `design_helper.py` | Sequences for redesigned arm primers and a blocking clamp |
| `read_snapgene.py` | SnapGene `.dna` → FASTA + feature table |
| `inspect_long.py` | Are long-insert survivors real genomic flank? (complexity, cross-cell-line) |
| `extract_salvage.py` | The useful one: demultiplex, drop early-adapter reads, trim primer + transposon end, write per-sample/arm FASTQ with the UMI in the read name |
| `make_report.py` | `qc_stats.json` → the QC PDF |

## What is not tracked

The gzipped FASTQs in `salvage/` (36 MB) and any FASTA/`.dna` donor maps are
gitignored — see `.gitignore`. The small JSON summaries (`qc_stats.json`,
`recheck_inserts.json`, `salvage_summary.json`) and the PDF *are* tracked: they
are the record of what this run showed, they cost 200 KB, and regenerating them
needs the raw CBCLs on the lab share. They also serve as the regression
reference for the refactor — the ported code should reproduce these numbers.

`extract_salvage.py` writes exactly the input shape
`pipeline/map_junction_insertions.py` consumes, which is where this application
and E. coli tagmentation meet.
