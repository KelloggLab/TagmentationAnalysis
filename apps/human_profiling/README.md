# Human integration profiling (`apps/human_profiling`)

UdiTaS — UniDirectional Targeted Sequencing — for CAST insertion outcomes in
human cells. These scripts read Illumina base calls **straight out of a run
folder's `.cbcl` files**, with no bcl2fastq step, so library structure can be
checked before committing to a conversion.

Every script decodes `L001_1.cbcl` (surface 1 only) for the first N tiles, so
they *sample* the run rather than scanning it. They need numpy and nothing else.

## How to run the pipeline

### 1. Describe the run

Two files define everything run-specific, and no cycle number or barcode is
hardcoded in the Python:

- **`run.yaml`** — the cycle layout, the Tn5 mosaic end, and each transposon
  arm's primer, donor sequence and transposon-end length.
- **`sample_sheet.csv`** — one row per library: `library, sample, arm,
  i5_index, i7_index`.

`runconfig.py` loads and validates them, and refuses an inconsistent pair:
mixed barcode lengths, a UMI plus i5 that does not add up to the I2 length, an
i7 that is not the I1 length, duplicate index pairs or library names, or a
sheet naming an arm the config does not define.

Both files as committed describe run `261030_VH00889_534_AAJLG3JM5`. For a new
run, copy them, edit, and point at the copy:

```bash
export UDITAS_RUN_CONFIG=/path/to/my_run.yaml
```

A `.json` file works too, and takes the same shape — handy because reading JSON
needs no PyYAML.

### 2. Check that the library has the structure you expect

```bash
LANE=<run>/Data/Intensities/BaseCalls/L001
python3 -I verify_structure.py $LANE 1 60
```

Decodes I1/I2/R2, demultiplexes, and reports the consensus R2 prefix per
sample. Run this first: if the barcodes or the cycle layout in `run.yaml` are
wrong, everything downstream is wrong, and this is where it shows.

### 3. Measure how much of the library is usable

```bash
python3 -I recheck_inserts.py $LANE 2
```

Reports the distance from the arm primer to the Tn5 adapter, tolerating
sequencing errors. When that distance is short there is no genomic flank
between them, and the read carries no insertion information.

### 4. Extract the junction reads

```bash
python3 -I extract_salvage.py $LANE 2
```

Demultiplexes, drops reads whose adapter is too close, trims the primer and
transposon end, and writes a FASTQ per sample/arm with the UMI in the read
name. It pools the two i5 replicates, so eight libraries become four groups.

### 5. Map the junction reads and call insertion sites

```bash
python3 map_junction_insertions.py \
    --fastq salvage/293T_RE.fastq.gz \
    --ref   /path/to/GRCh38.primary_assembly.fa \
    --outdir mapped/293T_RE
```

`bwa mem` into `samtools sort`, then one insertion coordinate per read and a
UMI-collapsed count per site. Two outputs:

    <sample>.insertions.tsv   read_id umi ref ins0 ins1 strand mapq cigar aln_start0 aln_end0_excl
    <sample>.sites.tsv        ref ins0 ins1 strand reads umis

`ins0` is 0-based (BED/Python) and `ins1` is 1-based (SAM). `reads` is how many
reads landed on a site; `umis` is how many distinct molecules, which is the
number to report. The sorted BAM is kept, since re-mapping is the expensive
step.

**This is a separate script from the E. coli one, not a flag on it.** The two
assays differ in four ways that each change the pipeline:

| | `ecoli_profiling` | here |
|---|---|---|
| Trimming | cutadapt, in the script | already done by `extract_salvage.py` |
| `--donor-seq` / `--donor-side` | required | neither exists; the insertion is always the read's 5' genomic base |
| Duplicates | one row per read | collapsed by UMI from the read name |
| BWA index | built if absent | must already exist, never built |
| Default MAPQ | 20, secondaries kept | 30, secondaries and alt contigs dropped |

Do **not** feed step 4's FASTQ to `apps/ecoli_profiling/map_junction_insertions.py`.
It requires `--donor-seq` and runs cutadapt with `--discard-untrimmed`, so it
would look for a donor sequence that is no longer in the read and discard
every one of them — an empty result, not an error.

#### Why the human genome changes the mapping step

The reference is ~3.1 Gb against E. coli's 4.6 Mb, and the flank is only
`min_flank` (40) nt. So:

- **The index is never built here.** `bwa index` on GRCh38 is about an hour of
  CPU and ~5 GB of output. A missing index is an error naming the five files,
  not a job that starts because a path was mistyped.
- **`bwa mem` is piped into `samtools sort`.** The E. coli script stages an
  intermediate `.sam`; at this scale that is a very large temporary file for
  something read once.
- **The genome is never loaded into Python.** Nothing in the mapping step needs
  reference sequence. In particular `core.sequence.load_genome` must not be
  pointed at GRCh38: it returns only the first record and builds two copies of
  the string on the way. For sequence windows around sites, use
  `pysam.FastaFile(ref).fetch(chrom, start, end)`, which reads off the `.fai`
  in constant memory.
- **Sites are counted in a sliding buffer**, flushed as the BAM's coordinate
  order moves past them, so memory scales with pile-up depth and not with the
  number of loci in the genome.
- **The filters are stricter and each one's toll is printed.** A 40 nt flank
  multimaps in a repetitive genome, so the drop counts are the result's main
  caveat rather than a footnote.

### Shared code these scripts use

The CBCL reading lives in [`core/read_cbcl.py`](../../core/read_cbcl.py) and the
barcode matching in [`core/barcodes.py`](../../core/barcodes.py), so the three
scripts above share one reader rather than carrying a copy each. Both import
numpy and nothing else, which is what keeps `python3 -I` enough to run them.

A SnapGene `.dna` file can be converted to FASTA plus a feature table with
`python3 -I ../../core/read_snapgene.py <in.dna> <out_prefix>`.

## What run 261030 showed

Most of the library was donor plasmid rather than insertion junctions: the Tn5
mosaic end sits too close to the LE/RE primer in **72% of assigned reads**,
leaving no genomic flank. It is worse on LE (88.5%) than on RE (66–69%).
`extract_salvage.py` recovered **792,246 reads of 8,475,767 assigned (9.3%)**:

| Group | Assigned | Adapter too close | Kept | UMI-collapsed |
|---|---:|---:|---:|---:|
| `293T_LE` | 1,027,883 | 88.5% | 117,569 | 113,769 |
| `293T_RE` | 3,576,491 | 68.7% | 323,607 | 312,166 |
| `FT_LE` | 906,600 | 88.7% | 101,780 | 96,566 |
| `FT_RE` | 2,964,793 | 65.8% | 249,290 | 243,122 |

The run itself: NextSeq 2000, P1, 1 lane, R1 301 / I1 8 / I2 18 / R2 301 —
cycles R1 1–301, I1 302–309, I2 310–327, R2 328–628 — 10.3 M clusters over 3
sampled tiles. Eight libraries: two cell lines (FT, 293T) × two i5 (N501,
N507) × two i7, where the i7 selects the arm — N701/N703 carry primer 251 (RE)
and N702/N704 primer 253 (LE).

The machine-readable record of this run -- `qc_stats.json`,
`recheck_inserts.json` and `salvage_summary.json` -- used to be tracked under
`results/261030/` and was removed on 2026-11-05. The numbers above are the
summary worth keeping; the JSON itself is recoverable from git history
(`git show 7261953:apps/human_profiling/results/261030/qc_stats.json`), but
regenerating it needs the raw CBCLs from the lab share.

## Known issue

The still-donor check on the **LE** arm is inactive. It compares a read's
flank against the donor sequence past the transposon end, and needs at least
`still_donor_min_tail` (12) bases to do so — but `run.yaml` records only 7
bases past LE's `end_len`. `runconfig.Arm.still_donor_active` reports this, and
two tests in `tests/test_uditas_runconfig.py` pin the behaviour from both
sides. Supplying more LE donor sequence from the plasmid map switches the check
on with no code change.

## What is not tracked

Gzipped FASTQs and any FASTA/`.dna` donor maps are gitignored — see
`.gitignore`. The raw run lives on the lab share.

## Tests

`tests/test_uditas_cbcl.py` (51 tests) runs these scripts and `core.read_cbcl`
against a synthetic Illumina lane, including `extract_salvage.py` end to end;
`tests/test_uditas_runconfig.py` (32) covers config loading and every
rejection; `tests/test_core_snapgene.py` (17) covers the `.dna` parser. None of
them needs anything beyond numpy.

`tests/test_uditas_mapping.py` (53) covers step 5. Most of it builds BAMs with
pysam directly, so it needs neither bwa nor a human genome: the insertion
coordinate on both strands, UMI collapsing, every filter, the flush window,
and that the script never builds an index or imports a genome loader. Seven
run the script end to end against a 50 kb toy genome with planted insertions,
checking each one is recovered at its exact position from both orientations;
those skip without `bwa` and `samtools` on PATH.
