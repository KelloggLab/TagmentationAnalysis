# CBCL QC scripts

Read index and read cycles straight out of an Illumina run folder's `.cbcl`
files, with no bcl2fastq step, so library structure can be checked before
committing to a conversion.

Four scripts are kept, the ones whose logic is not specific to one run. The
diagnostics that answered questions about run `261030_VH00889_534_AAJLG3JM5`
in particular were deleted; they are in this repository's history, and the
`apps/uditas/README.md` run summary records what they found.

What is still run-specific here, and what the reorganization needs to lift out
into a config and a sample sheet:

- the cycle layout (R1 1-301, I1 302-309, I2 310-327, R2 328-628), repeated in
  `verify_structure.py`, `recheck_inserts.py` and `extract_salvage.py`
- `SAMPLES` — the eight i7/i5 barcode pairs
- `PRIMER` / `DONOR` / `END_LEN` — the LE and RE arm sequences

Run each with `python3 -I`, passing the lane directory:

    LANE=<run>/Data/Intensities/BaseCalls/L001
    python3 -I verify_structure.py $LANE 1 60   # consensus R2 start per sample
    python3 -I recheck_inserts.py  $LANE 2      # primer->adapter distance
    python3 -I extract_salvage.py  $LANE 2      # per-sample/arm FASTQ

| Script | What it does | Run-specific? |
|---|---|---|
| `verify_structure.py` | Decode I1/I2/R2, demultiplex, consensus R2 prefix per sample. Holds the CBCL reader the others copy. | constants only |
| `recheck_inserts.py` | Primer→adapter distance, tolerating sequencing errors | constants only |
| `extract_salvage.py` | Demultiplex, drop early-adapter reads, trim primer + transposon end, write per-sample/arm FASTQ with the UMI in the read name | constants only |
| `read_snapgene.py` | SnapGene `.dna` → FASTA + feature table | no — fully general |

Only `L001_1.cbcl` (surface 1) is read, and only the first N tiles, so these
sample the run rather than scanning it. Needs numpy only.

Covered by `tests/test_uditas_cbcl.py` and `tests/test_uditas_snapgene.py`,
which run them against a synthetic CBCL lane.
