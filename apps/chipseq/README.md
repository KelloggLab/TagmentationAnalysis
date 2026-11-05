# ChIP-seq (`apps/chipseq`)

End-to-end ChIP-seq for 1–3 biological replicates of true paired-end FASTQ
against a bacterial reference genome.

R1 and R2 are aligned **together** as read pairs (bwa mem paired mode) and
MACS3 runs in BAMPE mode, so each properly-paired fragment's own aligned span
is used as the fragment. No fragment-size guessing (`--extsize`) is needed:
the true insert size comes straight from the alignment.

## How to run the pipeline

```bash
conda activate seqsuite
./chipseq_pipeline.sh --help
```

The steps it performs:

0. Adapter/quality trim each R1/R2 pair together (Trim Galore, `--paired`) — on by default
1. Build the BWA index, if not already present
2. Align each replicate's R1+R2 as a pair (bwa mem, paired-end)
3. Fixmate, sort, **mark** duplicates (marked, not removed), filter to properly-paired and MAPQ
4. Call peaks per replicate with MACS3 (`-f BAMPE`, `--keep-dup auto`, bedgraph output)
5. Build fold-enrichment tracks (`macs3 bdgcmp`)
6. Build a consensus peak set — peaks present in **all** replicates — with `bedtools intersect` (2 reps) or `multiinter` (3 reps)
7. Generate a genome-wide fold-enrichment plot for each replicate, **one per contig** ([`plot_enrichment.py`](plot_enrichment.py))

## The enrichment plots

Step 7 is `plot_enrichment.py`, a module rather than part of the shell script,
so it can be run on its own against an existing bedGraph and so it can be
tested:

```bash
python3 -I plot_enrichment.py \
    --fai genome.fasta.fai \
    --peaks results/consensus/consensus_peaks.bed \
    --rep results/macs3_rep1/rep1_FE.bdg "Replicate 1" results/plots/rep1
```

Contig lengths come from the reference's `.fai`, and **each contig is drawn on
its own axis**. With a single-contig reference the files are `<prefix>.png`
and `.pdf`; with more than one they are `<prefix>.<contig>.png` / `.pdf`, one
pair per contig that has coverage.

This matters if your reference has a plasmid in it. Until 2026-11-05 this step
was an inline Python block in the shell script that read the `chrom` column and
then ignored it: every contig was plotted from zero on a single axis whose
width was the *sum* of all contig lengths, so a chromosome and a plasmid were
drawn on top of each other across a doubled x-axis. The figure looked
plausible and was wrong. **Any figure from a multi-contig run made before that
date should be regenerated.** Single-contig output is unchanged, and byte-for-byte
identical for a reference 1 Mb or larger.

The axis is labelled in Mb for a contig of 1 Mb or more and in kb below that,
which is the one other difference: the old block always used Mb, so a 120 kb
plasmid got an axis of zeroes.

## Requirements

Beyond `environment.yml`, this pipeline needs tools that are **not** in it:

```bash
conda install -c bioconda trim-galore macs3 bedtools
```

plus `bwa` and `samtools`, which are in `environment.yml`.

## Tests

`tests/test_chipseq_plots.py` — 38 tests covering `plot_enrichment.py`: the
`.fai` and bedGraph readers, per-contig axis limits, the multi-contig file
naming, peak shading staying on its own contig, the axis units, and the
command line the pipeline calls. It needs only pandas and matplotlib, so it
never skips. Six of them read `chipseq_pipeline.sh` itself and assert the
script and the module still agree about how it is invoked.

Steps 0-6 still have no coverage: they are bash orchestrating trim_galore,
bwa, samtools, macs3 and bedtools, and testing them needs a synthetic FASTQ
fixture run through the real tools. That is the next thing worth adding here,
and it would skip without the tools installed the way
`tests/test_insertion_pipeline.py` does.
