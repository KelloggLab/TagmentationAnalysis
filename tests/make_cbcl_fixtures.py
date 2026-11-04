#!/usr/bin/env python3
"""Write a synthetic Illumina CBCL lane, so the UdiTaS code can be tested.

`apps/human_profiling/` reads base calls straight out of `.cbcl` files with no
bcl2fastq step, which is the reason none of it had tests: there was no input to
give it. This builds that input.

The format, exactly as the scripts parse it:

    header:  <HIBBI   version, header_size, bits_per_basecall,
                      bits_per_qscore, n_qscore_bins
             <II * n_qscore_bins       bin -> q-score mapping
             <I                        n_tiles
             <IIII * n_tiles           tile_number, n_clusters,
                                       uncompressed_size, compressed_size
    data:    at byte `header_size`, one gzip member per tile, concatenated in
             tile order. Each decompresses to packed nibbles, two base calls
             per byte, low nibble first.

    nibble:  bits 0-1  base index into "ACGT"
             bits 2-3  q-score bin; bin 0 means no-call, which the readers
                       turn into "N"

One file per cycle, at `C<cycle>.1/L001_1.cbcl`, and the scripts read only the
cycles they need, so this writes only the index and R2 cycles rather than all
628 of a real run.

usage: python make_cbcl_fixtures.py <outdir>
"""

import gzip
import json
import random
import struct
import sys
from pathlib import Path

# Cycle layout of run 261030_VH00889_534_AAJLG3JM5, which every script in
# apps/human_profiling/ reads from its config.
I1_START, I1_LEN = 302, 8
I2_START, I2_LEN = 310, 18
R2_START = 328
R2_LEN = 220          # extract_salvage.py reads the most: N_R2 = 220
UMI_LEN = 10

# Taken from extract_salvage.py. Kept here deliberately rather than imported,
# so a test can catch the constants changing underneath it.
SAMPLES = [
    ("FT", "RE", "TAGATCGC", "AGCGGAAT"),
    ("FT", "LE", "TAGATCGC", "GATCATGC"),
    ("FT", "RE", "AAGGAGTA", "AGCGGAAT"),
    ("FT", "LE", "AAGGAGTA", "GATCATGC"),
    ("293T", "RE", "TAGATCGC", "AAGACGGA"),
    ("293T", "LE", "TAGATCGC", "CGAGTCCT"),
    ("293T", "RE", "AAGGAGTA", "AAGACGGA"),
    ("293T", "LE", "AAGGAGTA", "CGAGTCCT"),
]
PRIMER = {"RE": "AGCTGACGCAAGAATGAGGGT", "LE": "GCATAAGCTTGCCGTTGCGG"}
END_LEN = {"RE": 11, "LE": 19}
DONOR = {
    "RE": "TGTAGTGTTGTTGACGGCAAAGTTATGGTTCCAGTAACGGCAAAGTTATGGGT",
    "LE": "CAAAGTTATGGGTAAAGTCACAATCC",
}
ME = "AGATGTGTATAAGAGACAG"
MIN_FLANK = 40

# bin -> q-score. Bin 0 is the no-call bin; the readers map it to "N".
QBINS = ((0, 2), (1, 14), (2, 21), (3, 37))
GOOD_BIN = 3          # the bin good base calls are written in
SEED = 20261104

BASE_INDEX = {b: i for i, b in enumerate("ACGT")}


def rc(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def encode_nibbles(seq):
    """One nibble per base; "N" becomes bin 0, which decodes back to "N"."""
    out = []
    for ch in seq:
        if ch == "N":
            out.append(0)
        else:
            out.append(BASE_INDEX[ch] | (GOOD_BIN << 2))
    return out


def pack_nibbles(nibbles):
    """Two base calls per byte, low nibble first."""
    padded = list(nibbles)
    if len(padded) % 2:
        padded.append(0)
    return bytes(padded[i] | (padded[i + 1] << 4)
                 for i in range(0, len(padded), 2))


def write_cbcl(path, tiles):
    """tiles: list of lists of equal-length base strings, one list per tile.

    Each inner list is the base call at this cycle for every cluster in that
    tile, so each element is a single character.
    """
    blobs, meta = [], []
    for i, calls in enumerate(tiles):
        raw = pack_nibbles(encode_nibbles("".join(calls)))
        blob = gzip.compress(raw)
        meta.append((1101 + i, len(calls), len(raw), len(blob)))
        blobs.append(blob)

    header_size = 12 + 8 * len(QBINS) + 4 + 16 * len(meta)
    head = struct.pack("<HIBBI", 1, header_size, 2, 2, len(QBINS))
    for b, q in QBINS:
        head += struct.pack("<II", b, q)
    head += struct.pack("<I", len(meta))
    for t in meta:
        head += struct.pack("<IIII", *t)
    assert len(head) == header_size, (len(head), header_size)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(head)
        for blob in blobs:
            fh.write(blob)


def write_lane(lane_dir, reads_per_tile):
    """reads_per_tile: list of tiles, each a list of (i1, i2, r2) strings.

    Writes one CBCL per cycle across the index and R2 ranges.
    """
    lane_dir = Path(lane_dir)
    for reads in reads_per_tile:
        for i1, i2, r2 in reads:
            assert len(i1) == I1_LEN and len(i2) == I2_LEN, "index length"
            assert len(r2) == R2_LEN, f"R2 must be {R2_LEN} nt, got {len(r2)}"

    ranges = [(I1_START, I1_LEN, 0), (I2_START, I2_LEN, 1), (R2_START, R2_LEN, 2)]
    for start, length, field in ranges:
        for offset in range(length):
            tiles = [[read[field][offset] for read in reads]
                     for reads in reads_per_tile]
            write_cbcl(lane_dir / f"C{start + offset}.1" / "L001_1.cbcl", tiles)
    return lane_dir


def pad(seq, rng):
    """Fill an R2 read out to the full cycle count with random bases."""
    if len(seq) > R2_LEN:
        return seq[:R2_LEN]
    return seq + "".join(rng.choice("ACGT") for _ in range(R2_LEN - len(seq)))


def build_read(arm, kind, rng, flank=None):
    """Build one R2 read of a given kind, and say what should become of it.

    The structure extract_salvage.py assumes is

        PRIMER[arm] + DONOR[arm][:END_LEN[arm]] + genomic flank + adapter

    and it slices the flank at a fixed offset, start = len(PRIMER) + END_LEN,
    without checking the primer is really there. `kind` picks which of its four
    outcomes the read should land in.
    """
    start = len(PRIMER[arm]) + END_LEN[arm]
    head = PRIMER[arm] + DONOR[arm][:END_LEN[arm]]
    assert len(head) == start

    if kind == "kept":
        flank = flank or "".join(rng.choice("ACGT") for _ in range(60))
        return pad(head + flank + ME, rng), flank
    if kind == "adapter_too_close":
        # Mosaic end immediately after the transposon end: no genomic flank.
        return pad(head + ME, rng), None
    if kind == "still_donor":
        # Donor sequence continues past the terminus, so it is not a junction.
        tail = DONOR[arm][END_LEN[arm]:]
        filler = "".join(rng.choice("ACGT") for _ in range(30))
        return pad(head + tail + filler + ME, rng), None
    if kind == "donor_tail_short":
        # Donor sequence continues past the terminus, but only by the few
        # bases DONOR["LE"] actually records. extract_salvage.py skips its
        # still-donor check unless it has >= 12 bases to compare, so on the LE
        # arm this read is NOT caught -- it is reported as a salvaged junction.
        tail = DONOR[arm][END_LEN[arm]:]
        filler = "".join(rng.choice("ACGT") for _ in range(60))
        return pad(head + tail + filler + ME, rng), tail + filler
    if kind == "too_short_after_trim":
        # Flank is long enough, but a no-call truncates it below MIN_FLANK.
        short = "".join(rng.choice("ACGT") for _ in range(10))
        rest = "".join(rng.choice("ACGT") for _ in range(49))
        return pad(head + short + "N" + rest + ME, rng), None
    raise AssertionError(kind)


def build(outdir):
    outdir = Path(outdir)
    rng = random.Random(SEED)

    # Sample 0 is FT/RE, sample 1 is FT/LE -- one of each arm.
    by_arm = {}
    for samp, arm, i5, i7 in SAMPLES:
        by_arm.setdefault(arm, (samp, arm, i5, i7))

    def indexes(i5, i7, umi):
        return rc(i7), umi + rc(i5)

    reads = []          # (i1, i2, r2)
    expected = {}       # "SAMPLE_ARM" -> expected counters
    kept_flanks = {}

    for arm in ("RE", "LE"):
        samp, _, i5, i7 = by_arm[arm]
        key = f"{samp}_{arm}"
        exp = {"assigned": 0, "kept": 0, "adapter_too_close": 0,
               "still_donor": 0, "too_short_after_trim": 0}

        # Three reads at one site with two distinct UMIs, so UMI collapsing
        # has something to collapse: 3 reads -> 1 site -> 2 unique UMIs.
        shared = "".join(rng.choice("ACGT") for _ in range(60))
        for umi in ["AAAAACCCCC", "AAAAACCCCC", "GGGGGTTTTT"]:
            r2, flank = build_read(arm, "kept", rng, flank=shared)
            i1, i2 = indexes(i5, i7, umi)
            reads.append((i1, i2, r2))
            exp["assigned"] += 1
            exp["kept"] += 1
        kept_flanks[key] = shared

        # Two more kept reads at distinct sites.
        for umi in ("ACGTACGTAC", "TGCATGCATG"):
            r2, flank = build_read(arm, "kept", rng)
            i1, i2 = indexes(i5, i7, umi)
            reads.append((i1, i2, r2))
            exp["assigned"] += 1
            exp["kept"] += 1

        # The still-donor check needs >= 12 bases of donor past the terminus to
        # compare against. DONOR["RE"] supplies 20, DONOR["LE"] only 7, so the
        # check is live on RE and dead on LE. The fixture reflects that rather
        # than pretending both arms behave the same.
        mix = [("adapter_too_close", 4), ("too_short_after_trim", 2)]
        if len(DONOR[arm][END_LEN[arm]:]) >= 12:
            mix.append(("still_donor", 2))
        else:
            mix.append(("donor_tail_short", 2))

        for kind, n in mix:
            for _ in range(n):
                r2, _ = build_read(arm, kind, rng)
                i1, i2 = indexes(i5, i7, "TTTTTAAAAA")
                reads.append((i1, i2, r2))
                exp["assigned"] += 1
                # A donor_tail_short read slips through as a kept junction,
                # which is the point of including it.
                exp["kept" if kind == "donor_tail_short" else kind] += 1

        # One read whose i7 carries a single substitution: the one-mismatch
        # map should still assign it.
        r2, _ = build_read(arm, "kept", rng)
        i1, i2 = indexes(i5, i7, "CCCCCGGGGG")
        i1 = ("T" if i1[0] != "T" else "A") + i1[1:]
        reads.append((i1, i2, r2))
        exp["assigned"] += 1
        exp["kept"] += 1

        expected[key] = exp

    # Reads that belong to no listed sample: never assigned, never counted.
    n_unassigned = 3
    for _ in range(n_unassigned):
        r2, _ = build_read("RE", "kept", rng)
        reads.append(("TTTTTTTT", "ACGTACGTAC" + "GGGGGGGG", r2))

    rng.shuffle(reads)

    # Split across two tiles, to exercise the per-tile loop.
    half = len(reads) // 2
    tiles = [reads[:half], reads[half:]]

    lane = outdir / "Data" / "Intensities" / "BaseCalls" / "L001"
    write_lane(lane, tiles)

    truth = {
        "lane_dir": str(lane),
        "n_tiles": len(tiles),
        "clusters_per_tile": [len(t) for t in tiles],
        "total_clusters": len(reads),
        "n_unassigned": n_unassigned,
        "cycles": {"i1": [I1_START, I1_LEN], "i2": [I2_START, I2_LEN],
                   "r2": [R2_START, R2_LEN]},
        "qbins": {str(b): q for b, q in QBINS},
        "good_bin": GOOD_BIN,
        "still_donor_check_active": {
            arm: len(DONOR[arm][END_LEN[arm]:]) >= 12 for arm in ("RE", "LE")
        },
        "donor_bases_past_terminus": {
            arm: len(DONOR[arm][END_LEN[arm]:]) for arm in ("RE", "LE")
        },
        "expected_groups": expected,
        "shared_site_flank": kept_flanks,
        "seed": SEED,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cbcl_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return truth


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python make_cbcl_fixtures.py <outdir>")
    truth = build(sys.argv[1])
    print(f"wrote {truth['total_clusters']} clusters over "
          f"{truth['n_tiles']} tiles to {truth['lane_dir']}")
    for key, exp in truth["expected_groups"].items():
        print(f"  {key}: {exp}")


if __name__ == "__main__":
    main()
