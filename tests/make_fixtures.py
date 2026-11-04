#!/usr/bin/env python3
"""Build a small synthetic tagmentation dataset with known insertion positions.

The point is to have an input whose correct output is known exactly, so that
refactoring can be checked by "the same input still gives the same table".

What it writes into <outdir>:

    toy_genome.fasta    50 kb of random sequence, one contig named `toy_chr`
    toy_reads.fastq     junction reads built from that genome, plus decoys
    truth.json          what the pipeline is expected to report

Read construction, for an insertion at 0-based genomic position `pos`:

    + strand   DONOR_END + genome[pos : pos+FLANK]
    - strand   DONOR_END + revcomp(genome[pos-FLANK+1 : pos+1])

In both cases the genomic base adjacent to the donor junction is `pos`, so
`map_junction_insertions.py --donor-side 5p` should report ins0 == pos. On the
plus strand that is the alignment's reference_start; on the minus strand it is
reference_end - 1. Every read carries its own truth in its name
(`ins_<pos>_<strand>_<n>`), so each output row can be checked individually
rather than only in aggregate.

usage: python make_fixtures.py <outdir>
"""

import json
import random
import sys
from pathlib import Path

# A synthetic transposon end. Arbitrary, but fixed: it must not occur in the
# toy genome, or reads would be trimmed in the wrong place.
DONOR_END = "ACGTTGCATCAGGTCCTTAAGGCAT"

GENOME_LEN = 50_000
GENOME_NAME = "toy_chr"
FLANK = 70
SEED = 20261103

# A 20 bp guide placed at a known spot, for the on-target analysis tests.
GUIDE_POS = 12_000
GUIDE_LEN = 20

# (position, strand, n_reads). Positions are 0-based.
PLANTED = [
    (12_071, "+", 50),   # hotspot, 71 bp downstream of the guide start
    (12_071, "-", 10),   # same position, other strand
    (12_500, "+", 25),   # second hotspot
    (30_000, "+", 5),
    (30_123, "-", 5),
    (41_000, "+", 1),    # singleton
]

# Reads that should NOT produce an insertion call, and why.
#   no_donor     no transposon end -> cutadapt --discard-untrimmed drops it
#   short_flank  only 12 bp of genome after the end -> below cutadapt -m 20
#   unmappable   70 bp of poly-A -> unmapped, or mapped below --min-mapq
DECOYS = [("no_donor", 8), ("short_flank", 5), ("unmappable", 4)]

QUAL = "I"  # Q40 everywhere; base quality is not what these tests exercise

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def make_genome(rng):
    """Random sequence, regenerated until it contains no donor-end lookalike."""
    while True:
        g = "".join(rng.choice("ACGT") for _ in range(GENOME_LEN))
        # The donor end must be absent, and so must anything within 2
        # mismatches of it, since cutadapt runs at --error-rate 0.1 over 25 bp.
        if not _has_near_match(g, DONOR_END, max_mismatch=2):
            return g


def _has_near_match(genome, probe, max_mismatch):
    m = len(probe)
    for i in range(len(genome) - m + 1):
        mm = 0
        for a, b in zip(genome[i:i + m], probe):
            if a != b:
                mm += 1
                if mm > max_mismatch:
                    break
        else:
            return True
    return False


def junction_read(genome, pos, strand):
    if strand == "+":
        flank = genome[pos:pos + FLANK]
    else:
        flank = revcomp(genome[pos - FLANK + 1:pos + 1])
    assert len(flank) == FLANK, f"flank off the end of the genome at {pos}"
    return DONOR_END + flank


def build(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    genome = make_genome(rng)

    # Keep the planted sites distinguishable: no two within a read length of
    # each other, or a single alignment could be attributed to either.
    positions = sorted({p for p, _, _ in PLANTED})
    for a, b in zip(positions, positions[1:]):
        assert b - a > FLANK, f"planted sites {a} and {b} are too close to tell apart"

    fasta = outdir / "toy_genome.fasta"
    with fasta.open("w") as fh:
        fh.write(f">{GENOME_NAME}\n")
        for i in range(0, len(genome), 60):
            fh.write(genome[i:i + 60] + "\n")

    records = []  # (name, sequence)

    for pos, strand, n in PLANTED:
        seq = junction_read(genome, pos, strand)
        for i in range(n):
            records.append((f"ins_{pos}_{strand}_{i}", seq))

    for kind, n in DECOYS:
        for i in range(n):
            if kind == "no_donor":
                start = 45_000 + i * FLANK
                seq = genome[start:start + FLANK]
            elif kind == "short_flank":
                seq = DONOR_END + genome[20_000 + i:20_012 + i]
            elif kind == "unmappable":
                seq = DONOR_END + "A" * FLANK
            else:
                raise AssertionError(kind)
            records.append((f"decoy_{kind}_{i}", seq))

    # Deterministic shuffle, so read order in the FASTQ is not the order the
    # truth table happens to be written in.
    rng.shuffle(records)

    fastq = outdir / "toy_reads.fastq"
    with fastq.open("w") as fh:
        for name, seq in records:
            fh.write(f"@{name}\n{seq}\n+\n{QUAL * len(seq)}\n")

    truth = {
        "genome_fasta": fasta.name,
        "reads_fastq": fastq.name,
        "genome_name": GENOME_NAME,
        "genome_len": GENOME_LEN,
        "donor_end": DONOR_END,
        "flank": FLANK,
        "seed": SEED,
        "guide": {
            "pos": GUIDE_POS,
            "sequence": genome[GUIDE_POS:GUIDE_POS + GUIDE_LEN],
        },
        "expected_insertions": [
            {"pos": p, "strand": s, "n_reads": n} for p, s, n in PLANTED
        ],
        "expected_insertion_rows": sum(n for _, _, n in PLANTED),
        "decoys": [{"kind": k, "n_reads": n} for k, n in DECOYS],
        "total_reads": len(records),
    }
    (outdir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")

    return truth


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip().splitlines()[-1])
    truth = build(sys.argv[1])
    print(f"wrote {truth['total_reads']} reads "
          f"({truth['expected_insertion_rows']} with a known insertion) "
          f"to {sys.argv[1]}")


if __name__ == "__main__":
    main()
