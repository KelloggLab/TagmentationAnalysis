"""Decode index-read cycles straight from CBCL files (first N tiles) and tally
barcodes, to check i7/i5 orientation and where the UMI sits in I2.

usage: python -I cbcl_index_peek.py <BaseCalls/L001> <i1_start> <i1_len> <i2_start> <i2_len> [ntiles]
"""
import gzip
import struct
import sys
from collections import Counter

import numpy as np

BASES = np.frombuffer(b"ACGT", dtype=np.uint8)


def read_header(f):
    version, header_size, bits_base, bits_q, nbins = struct.unpack("<HIBBI", f.read(12))
    f.read(8 * nbins)
    (ntiles,) = struct.unpack("<I", f.read(4))
    tiles = [struct.unpack("<IIII", f.read(16)) for _ in range(ntiles)]
    (nonpf_excluded,) = struct.unpack("<B", f.read(1))
    return header_size, tiles, nonpf_excluded


def decode_cycle(path, ntiles):
    with open(path, "rb") as f:
        header_size, tiles, excl = read_header(f)
        f.seek(header_size)
        out = []
        for tile, nclust, _usize, csize in tiles[:ntiles]:
            raw = np.frombuffer(gzip.decompress(f.read(csize)), dtype=np.uint8)
            nib = np.empty(raw.size * 2, dtype=np.uint8)
            nib[0::2] = raw & 0x0F
            nib[1::2] = raw >> 4
            nib = nib[:nclust]
            seq = BASES[nib & 0x3].copy()
            seq[(nib >> 2) == 0] = ord("N")
            out.append(seq)
    return np.concatenate(out), excl


def read_index(lane_dir, start, length, ntiles):
    cols, excl = [], None
    for c in range(start, start + length):
        s, excl = decode_cycle(f"{lane_dir}/C{c}.1/L001_1.cbcl", ntiles)
        cols.append(s)
    return np.stack(cols, axis=1), excl


def rc(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def main():
    lane_dir = sys.argv[1]
    i1s, i1l, i2s, i2l = map(int, sys.argv[2:6])
    ntiles = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    i1, excl = read_index(lane_dir, i1s, i1l, ntiles)
    i2, _ = read_index(lane_dir, i2s, i2l, ntiles)
    n = i1.shape[0]
    print(f"clusters decoded: {n}  (non-PF excluded flag={excl})")

    i7 = {"AGCGGAAT": "701", "GATCATGC": "702", "AAGACGGA": "703", "CGAGTCCT": "704"}
    i5 = {"TAGATCGC": "501", "AAGGAGTA": "507"}

    def tally(arr, lo, hi, label, expect):
        c = Counter(r[lo:hi].tobytes().decode() for r in arr)
        print(f"\n{label} positions {lo+1}-{hi}: top barcodes")
        for bc, k in c.most_common(8):
            tag = ""
            if bc in expect:
                tag = f"<- {expect[bc]} as given"
            elif rc(bc) in expect:
                tag = f"<- {expect[rc(bc)]} REVERSE-COMPLEMENTED"
            print(f"  {bc}  {k:>8}  {100*k/n:5.1f}%  {tag}")

    tally(i1, 0, 8, "I1", i7)
    tally(i2, 0, 8, "I2", i5)
    tally(i2, i2l - 8, i2l, "I2", i5)

    # per-position diversity in I2: barcode positions are low-entropy, UMI high
    print("\nI2 per-position top-base fraction (high = fixed barcode, ~0.25-0.4 = random UMI):")
    for p in range(i2l):
        col = i2[:, p]
        frac = max((col == b).mean() for b in b"ACGT")
        print(f"  pos {p+1:>2}: {frac:.2f}")


if __name__ == "__main__":
    main()
