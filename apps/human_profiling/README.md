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

The CBCL reading lives in [`core/read_cbcl.py`](../../core/read_cbcl.py) and the
barcode matching in [`core/barcodes.py`](../../core/barcodes.py), so the three
scripts above share one reader rather than carrying a copy each. Both import
numpy and nothing else, which is what keeps `python3 -I` enough to run them.

A SnapGene `.dna` file can be converted to FASTA plus a feature table with
`python3 -I ../../core/read_snapgene.py <in.dna> <out_prefix>`.

## Tests

`tests/test_uditas_cbcl.py` (51 tests) runs these scripts and `core.read_cbcl`
against a synthetic Illumina lane, including `extract_salvage.py` end to end;
`tests/test_uditas_runconfig.py` (32) covers config loading and every
rejection; `tests/test_core_snapgene.py` (17) covers the `.dna` parser. None of
them needs anything beyond numpy.
