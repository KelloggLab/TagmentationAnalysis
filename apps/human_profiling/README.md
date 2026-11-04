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

The output is exactly the input shape
[`apps/ecoli_profiling/map_junction_insertions.py`](../ecoli_profiling) consumes,
which is where this app and the E. coli work meet.

### Shared code these scripts use

The CBCL reading lives in [`core/illumina.py`](../../core/illumina.py) and the
barcode matching in [`core/barcodes.py`](../../core/barcodes.py), so the three
scripts above share one reader rather than carrying a copy each. Both import
numpy and nothing else, which is what keeps `python3 -I` enough to run them.

A SnapGene `.dna` file can be converted to FASTA plus a feature table with
`python3 -I ../../core/snapgene.py <in.dna> <out_prefix>`.

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

`results/261030/` holds the machine-readable record: `qc_stats.json`,
`recheck_inserts.json` and `salvage_summary.json`. They are tracked because
regenerating them needs the raw CBCLs from the lab share, they cost 150 KB, and
they are the regression reference for any future change to these scripts.

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

`tests/test_uditas_cbcl.py` (51 tests) runs these scripts and `core.illumina`
against a synthetic Illumina lane, including `extract_salvage.py` end to end;
`tests/test_uditas_runconfig.py` (32) covers config loading and every
rejection; `tests/test_core_snapgene.py` (17) covers the `.dna` parser. None of
them needs anything beyond numpy.
