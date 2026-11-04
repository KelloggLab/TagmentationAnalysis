# CBCL QC scripts

Read index and read cycles straight out of an Illumina run folder's `.cbcl`
files, with no bcl2fastq step, so library structure can be checked before
committing to a conversion. Written for run
`261030_VH00889_534_AAJLG3JM5` (NextSeq 2000, R1 301 / I1 8 / I2 18 / R2 301;
cycle layout R1 1-301, I1 302-309, I2 310-327, R2 328-628).

Sample barcodes and the LE/RE primers are hardcoded in `verify_structure.py`.
Run each with `python3 -I`, passing the lane directory:

    LANE=<run>/Data/Intensities/BaseCalls/L001
    python3 -I verify_structure.py $LANE 1 60   # consensus R2 start per sample
    python3 -I verify2.py          $LANE 1 150  # primer match + mosaic-end position
    python3 -I verify3.py          $LANE 1      # top distinct R2 starts
    python3 -I verify4.py          $LANE 1      # what sits between primer and ME
    python3 -I verify5.py          $LANE 1      # diversity of the no-ME reads
    python3 -I cbcl_index_peek.py  $LANE 302 8 310 18 2   # index orientation

Only `L001_1.cbcl` (surface 1) is read, and only the first N tiles, so these
sample the run rather than scanning it. Needs numpy only.
