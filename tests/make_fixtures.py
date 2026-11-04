#!/usr/bin/env python3
"""Build a small synthetic tagmentation dataset with known insertion positions.

The point is to have an input whose correct output is known exactly, so that
refactoring can be checked by "the same input still gives the same table".

What it writes into <outdir>:

    toy_genome.fasta    50 kb of random sequence, one contig named `toy_chr`
    toy_donor.gb        1.5 kb donor plasmid: Cargo 1..600, backbone 601..1500
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

Two stretches of the donor plasmid are spliced into the toy genome, which is
what makes `filter_donor_backbone.sh` testable: reads from those regions map
to the genome (so they appear in the insertion table) while also matching the
donor. It mimics a real situation -- a pre-existing donor integration in the
reference strain. Backbone-derived reads should be filtered out; Cargo-derived
reads should survive, because a genuine new integration carries Cargo sequence
right up to the transposon end.

usage: python make_fixtures.py <outdir>
"""

import json
import random
import sys
from pathlib import Path

# A synthetic transposon end. Arbitrary, but fixed: it must not occur in the
# toy genome or in the donor plasmid, or reads would be trimmed in the wrong
# place.
DONOR_END = "ACGTTGCATCAGGTCCTTAAGGCAT"

GENOME_LEN = 50_000
GENOME_NAME = "toy_chr"
FLANK = 70
SEED = 20261103

# A 20 bp guide placed at a known spot, for the on-target analysis tests.
GUIDE_POS = 12_000
GUIDE_LEN = 20

# Donor plasmid: Cargo occupies 1..CARGO_LEN, the backbone the rest. The
# backbone filter screens only the backbone.
CARGO_LEN = 600
BACKBONE_LEN = 900
DONOR_NAME = "toy_donor"

# Where stretches of the donor are spliced into the genome, and which part of
# the donor they come from. Both are 400 bp.
HOMOLOGY = {
    "backbone": {"genome_start": 25_000, "donor_slice": (100, 500)},
    "cargo": {"genome_start": 35_000, "donor_slice": (100, 500)},
}
HOMOLOGY_LEN = 400

# (position, strand, n_reads, origin). Positions are 0-based. `origin` says
# what the flanking sequence actually is, which is what the backbone filter is
# expected to act on:
#   genome            ordinary unique genomic DNA  -> always kept
#   backbone          70 bp of donor backbone      -> filtered out
#   backbone_partial  40 bp of backbone, then genome -> filtered at -m 30,
#                     kept at -m 50, so it pins the threshold
#   cargo             70 bp of donor Cargo         -> deliberately kept
PLANTED = [
    (12_071, "+", 50, "genome"),   # hotspot, 71 bp downstream of the guide
    (12_071, "-", 10, "genome"),   # same position, other strand
    (12_500, "+", 25, "genome"),   # second hotspot
    (25_100, "+", 10, "backbone"),
    (25_360, "+", 8, "backbone_partial"),
    (30_000, "+", 5, "genome"),
    (30_123, "-", 5, "genome"),
    (35_100, "+", 10, "cargo"),
    (41_000, "+", 1, "genome"),    # singleton
]

# How much of the flank is donor backbone, for the partial site above.
PARTIAL_BACKBONE_BP = 40

# Reads that should NOT produce an insertion call, and why.
#   no_donor     no transposon end -> cutadapt --discard-untrimmed drops it
#   short_flank  only 12 bp of genome after the end -> below cutadapt -m 20
#   unmappable   70 bp of poly-A -> unmapped, or mapped below --min-mapq
DECOYS = [("no_donor", 8), ("short_flank", 5), ("unmappable", 4)]

QUAL = "I"  # Q40 everywhere; base quality is not what these tests exercise

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def _has_near_match(seq, probe, max_mismatch):
    """True if `probe` occurs in `seq` within max_mismatch substitutions."""
    m = len(probe)
    for i in range(len(seq) - m + 1):
        mm = 0
        for a, b in zip(seq[i:i + m], probe):
            if a != b:
                mm += 1
                if mm > max_mismatch:
                    break
        else:
            return True
    return False


def _clean_random(rng, length):
    """Random sequence containing nothing resembling the transposon end.

    Two mismatches, because cutadapt runs at --error-rate 0.1 over 25 bp.
    """
    while True:
        s = "".join(rng.choice("ACGT") for _ in range(length))
        if not _has_near_match(s, DONOR_END, max_mismatch=2):
            return s


def build_donor(rng):
    """Returns (donor_sequence, cargo, backbone)."""
    donor = _clean_random(rng, CARGO_LEN + BACKBONE_LEN)
    return donor, donor[:CARGO_LEN], donor[CARGO_LEN:]


def build_genome(rng, cargo, backbone):
    """Random genome with one backbone stretch and one Cargo stretch spliced in."""
    genome = _clean_random(rng, GENOME_LEN)

    for part, source in (("backbone", backbone), ("cargo", cargo)):
        spec = HOMOLOGY[part]
        a, b = spec["donor_slice"]
        chunk = source[a:b]
        assert len(chunk) == HOMOLOGY_LEN, part
        start = spec["genome_start"]
        genome = genome[:start] + chunk + genome[start + HOMOLOGY_LEN:]

    assert len(genome) == GENOME_LEN
    # Splicing cannot have introduced a transposon-end lookalike, but check,
    # because a false one here would silently corrupt every expectation.
    assert not _has_near_match(genome, DONOR_END, max_mismatch=2)
    return genome


def junction_read(genome, pos, strand):
    if strand == "+":
        flank = genome[pos:pos + FLANK]
    else:
        flank = revcomp(genome[pos - FLANK + 1:pos + 1])
    assert len(flank) == FLANK, f"flank off the end of the genome at {pos}"
    return DONOR_END + flank


def write_genbank(path, donor):
    """A minimal GenBank file of the shape filter_donor_backbone.sh parses:
    a misc_feature at 1..N labelled Cargo, then an ORIGIN block."""
    lines = [
        f"LOCUS       {DONOR_NAME}         {len(donor)} bp    DNA     circular SYN 03-NOV-2026",
        "DEFINITION  Synthetic donor plasmid for the test suite.",
        "ACCESSION   .",
        "KEYWORDS    .",
        "SOURCE      synthetic DNA construct",
        "FEATURES             Location/Qualifiers",
        f"     misc_feature    1..{CARGO_LEN}",
        '                     /label="Cargo"',
        f"     misc_feature    {CARGO_LEN + 1}..{len(donor)}",
        '                     /label="Backbone"',
        "ORIGIN",
    ]
    for i in range(0, len(donor), 60):
        block = donor[i:i + 60].lower()
        groups = " ".join(block[j:j + 10] for j in range(0, len(block), 10))
        lines.append(f"{i + 1:>9} {groups}")
    lines.append("//")
    path.write_text("\n".join(lines) + "\n")


def build(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    donor, cargo, backbone = build_donor(rng)
    genome = build_genome(rng, cargo, backbone)

    # Keep the planted sites distinguishable: no two within a read length of
    # each other, or a single alignment could be attributed to either.
    positions = sorted({p for p, _, _, _ in PLANTED})
    for a, b in zip(positions, positions[1:]):
        assert b - a > FLANK, f"planted sites {a} and {b} are too close to tell apart"

    # The donor-derived sites must really sit in the spliced-in regions, or the
    # backbone-filter expectations below are meaningless.
    for pos, _, _, origin in PLANTED:
        if origin == "backbone":
            assert genome[pos:pos + FLANK] in backbone, pos
        elif origin == "cargo":
            assert genome[pos:pos + FLANK] in cargo, pos
        elif origin == "backbone_partial":
            head = genome[pos:pos + PARTIAL_BACKBONE_BP]
            assert head in backbone, pos
            assert genome[pos:pos + FLANK] not in backbone, pos

    fasta = outdir / "toy_genome.fasta"
    with fasta.open("w") as fh:
        fh.write(f">{GENOME_NAME}\n")
        for i in range(0, len(genome), 60):
            fh.write(genome[i:i + 60] + "\n")

    gb = outdir / "toy_donor.gb"
    write_genbank(gb, donor)

    records = []  # (name, sequence)

    for pos, strand, n, _ in PLANTED:
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

    expected = [{"pos": p, "strand": s, "n_reads": n, "origin": o}
                for p, s, n, o in PLANTED]
    filtered_out = sum(e["n_reads"] for e in expected
                       if e["origin"] in ("backbone", "backbone_partial"))

    truth = {
        "genome_fasta": fasta.name,
        "reads_fastq": fastq.name,
        "donor_genbank": gb.name,
        "genome_name": GENOME_NAME,
        "genome_len": GENOME_LEN,
        "donor_end": DONOR_END,
        "flank": FLANK,
        "seed": SEED,
        "guide": {
            "pos": GUIDE_POS,
            "sequence": genome[GUIDE_POS:GUIDE_POS + GUIDE_LEN],
        },
        "donor": {
            "length": len(donor),
            "cargo_end": CARGO_LEN,
            "backbone_len": BACKBONE_LEN,
            "backbone": backbone,
        },
        "homology": {k: dict(v, length=HOMOLOGY_LEN) for k, v in HOMOLOGY.items()},
        "partial_backbone_bp": PARTIAL_BACKBONE_BP,
        "expected_insertions": expected,
        "expected_insertion_rows": sum(e["n_reads"] for e in expected),
        "expected_backbone_filtered": filtered_out,
        "decoys": [{"kind": k, "n_reads": n} for k, n in DECOYS],
        "total_reads": len(records),
    }
    (outdir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")

    return truth


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python make_fixtures.py <outdir>")
    truth = build(sys.argv[1])
    print(f"wrote {truth['total_reads']} reads "
          f"({truth['expected_insertion_rows']} with a known insertion, "
          f"{truth['expected_backbone_filtered']} of them donor-backbone) "
          f"to {sys.argv[1]}")


if __name__ == "__main__":
    main()
