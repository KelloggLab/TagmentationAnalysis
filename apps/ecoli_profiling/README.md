# E. coli integration profiling (`apps/ecoli_profiling`)

Where does a CRISPR-associated transposon (CAST) integrate across an *E. coli*
genome? Each read starts inside the transposon end (LE or RE) and reads outward
into flanking genomic DNA. These steps trim the transposon sequence away, map
what is left, and report one insertion coordinate per read.

## How to run the pipeline

Activate the environment first; it is defined once at the top level and covers
every app.

```bash
conda env create -f ../../environment.yml   # first time only
conda activate seqsuite
```

### 1. Call insertion sites

```bash
python map_junction_insertions.py \
    --fastq      reads.fastq.gz \
    --donor      donor.fasta \
    --reference  genome.fasta \
    --donor-side 5p \
    --outdir     mapped_reads
```

cutadapt trims the donor/transposon sequence, bwa mem maps the remainder, and
the insertion coordinate is read off each alignment with CIGAR-aware arithmetic.
`--donor-side 5p` takes the reference 5' end of the alignment as the insertion
point, `3p` the 3' end. The output TSV has one row per read:

    read_id  ref  ins0  ins1  strand  mapq  cigar  aln_start0  aln_end0_excl

`ins0` is 0-based (BED/Python style) and `ins1` is 1-based (SAM style). Use
`--min-overlap` to relax the default requirement that the whole donor match.

### 2. Remove donor-backbone reads

```bash
./filter_donor_backbone.sh \
    -g donor.gb \
    -f mapped_reads/reads.trimmed.fastq \
    -t mapped_reads/reads.insertions.5p.tsv \
    -o mapped_reads/reads.insertions.5p.nobackbone.tsv
```

After trimming, a read's remaining sequence is supposed to be unique genomic
flank. If it instead matches the donor plasmid **backbone** — ori, oriT,
resistance marker, primer sites, everything outside the Cargo — the read is
free plasmid or a pre-existing backbone integration, not new signal, and it is
dropped.

This screens against the backbone only, never the whole plasmid: a genuine new
integration legitimately carries Cargo sequence right up to the LE/RE boundary
and must still be called. The donor GenBank file therefore needs a
`misc_feature` annotated `Cargo` at location `1..N`; everything after `N` is
treated as backbone.

### 3. Plot the profile

```bash
./plot_profile.sh -f mapped_reads/reads.insertions.5p.nobackbone.tsv 500
```

Plots + and − strand counts side by side across the genome with gnuplot, in
bins of 500 bp (default 1000). `-f` plots each bin as a fraction of all reads
instead of a raw count. The input is any whitespace-delimited file with the
position in column 3 and the strand in column 5, which is what step 1 writes.

## Analysing the result

The insertion TSV is the input to the shared functions in
[`core/`](../../core): `insertions.py` for on-target percentage and
target-site duplications, `plots.py` for the profile figure, and
`guides.py`/`sequence.py`/`motifs.py` for
guide/PAM placement, A/T richness and position weight matrices. See
[docs/API.md](../../docs/API.md).

`ccdb_integration.py` lives here rather than in `core/`: its `integration_finder`
takes the ccdB gene's coordinates and scores how far each insertion fell from
the PAM, which only means anything for this assay's landing site. Import it as

```python
from apps.ecoli_profiling import ccdb_integration
```

`apps/human_profiling/extract_salvage.py` writes exactly the input shape
`map_junction_insertions.py` consumes, which is where this app and the human
UdiTaS work meet.

## Requirements

`cutadapt`, `bwa`, `samtools` and `gnuplot` on your PATH, plus `pysam`. All
four come from `environment.yml`.

## Tests

`tests/test_insertion_pipeline.py` runs step 1 end to end against a synthetic
dataset with planted insertions and checks every one is recovered;
`tests/test_backbone_filter.py` covers what step 2 drops and spares. Both need
the real tools installed, and skip without them.
