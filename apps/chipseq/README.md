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
7. Generate a genome-wide fold-enrichment plot for each replicate

## Requirements

Beyond `environment.yml`, this pipeline needs tools that are **not** in it:

```bash
conda install -c bioconda trim-galore macs3 bedtools
```

plus `bwa` and `samtools`, which are in `environment.yml`.

## Tests

None yet. This is the one app in the repository with no test coverage; it
should get a small fixture before it is relied on for a figure.
