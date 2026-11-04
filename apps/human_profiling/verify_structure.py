"""Decode I1, I2 and the start of R2 from CBCL for one tile, demultiplex in
memory, and show the consensus R2 prefix per sample.

Cycle layout for 261030_VH00889_534_AAJLG3JM5: R1 1-301, I1 302-309,
I2 310-327, R2 328-628.

usage: python -I verify_structure.py <BaseCalls/L001> [ntiles] [n_r2_cycles]
"""
import gzip
import os
import struct
import sys
from collections import Counter

import numpy as np

BASES = np.frombuffer(b"ACGT", dtype=np.uint8)

# -I drops the script's directory from sys.path; add just that one back.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runconfig  # noqa: E402

CFG = runconfig.load()

I1_START, I1_LEN = CFG.cycles.i1_start, CFG.cycles.i1_len
I2_START, I2_LEN = CFG.cycles.i2_start, CFG.cycles.i2_len
R2_START = CFG.cycles.r2_start
UMI_LEN = CFG.cycles.umi_len

# (library, i5, i7), the library name being how this script labels its output.
SAMPLES = [(s.library, s.i5, s.i7) for s in CFG.samples]
# 3' target-specific portion of each LE/RE PCR primer, keyed by arm. This used
# to be keyed by the primer number in the library name ("251"/"253"); the arm
# is what it always meant.
PRIMERS = {name: a.primer for name, a in CFG.arms.items()}
# The arm each library was amplified with, for labelling.
ARM_OF = {s.library: s.arm for s in CFG.samples}


def rc(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def read_header(f):
    _ver, header_size, _bb, _bq, nbins = struct.unpack("<HIBBI", f.read(12))
    f.read(8 * nbins)
    (ntiles,) = struct.unpack("<I", f.read(4))
    tiles = [struct.unpack("<IIII", f.read(16)) for _ in range(ntiles)]
    return header_size, tiles


def decode_cycle(path, ntiles):
    with open(path, "rb") as f:
        header_size, tiles = read_header(f)
        f.seek(header_size)
        out = []
        for _tile, nclust, _usize, csize in tiles[:ntiles]:
            raw = np.frombuffer(gzip.decompress(f.read(csize)), dtype=np.uint8)
            nib = np.empty(raw.size * 2, dtype=np.uint8)
            nib[0::2] = raw & 0x0F
            nib[1::2] = raw >> 4
            nib = nib[:nclust]
            seq = BASES[nib & 0x3].copy()
            seq[(nib >> 2) == 0] = ord("N")
            out.append(seq)
    return np.concatenate(out)


def read_block(lane_dir, start, length, ntiles):
    return np.stack(
        [decode_cycle(f"{lane_dir}/C{c}.1/L001_1.cbcl", ntiles)
         for c in range(start, start + length)], axis=1)


def one_mismatch_map(barcodes):
    """barcode variant -> canonical barcode, dropping ambiguous variants."""
    exact = set(barcodes)
    m = {bc: bc for bc in barcodes}
    ambiguous = set()
    for bc in barcodes:
        for i in range(len(bc)):
            for b in "ACGTN":
                v = bc[:i] + b + bc[i + 1:]
                if v in m and m[v] != bc:
                    ambiguous.add(v)
                else:
                    m.setdefault(v, bc)
    for v in ambiguous - exact:
        m.pop(v, None)
    return m


def main():
    lane_dir = sys.argv[1]
    ntiles = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_r2 = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    i1 = read_block(lane_dir, I1_START, I1_LEN, ntiles)
    i2 = read_block(lane_dir, I2_START, I2_LEN, ntiles)
    r2 = read_block(lane_dir, R2_START, n_r2, ntiles)
    n = i1.shape[0]
    print(f"clusters: {n}   R2 cycles decoded: {n_r2}\n")

    # Indexes come off reverse-complemented; barcode sits after the UMI in I2.
    i7_map = one_mismatch_map([rc(i7) for _, _, i7 in SAMPLES])
    i5_map = one_mismatch_map([rc(i5) for _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): name for name, i5, i7 in SAMPLES}

    per_sample = {name: Counter() for name, _, _ in SAMPLES}
    assigned = 0
    for k in range(n):
        bc7 = i7_map.get(i1[k].tobytes().decode())
        bc5 = i5_map.get(i2[k, UMI_LEN:].tobytes().decode())
        if bc7 is None or bc5 is None:
            continue
        name = combo.get((bc7, bc5))
        if name is None:
            continue
        assigned += 1
        per_sample[name][r2[k].tobytes().decode()] += 1

    print(f"assigned to a listed sample: {assigned} ({100*assigned/n:.1f}%)\n")

    for name, _, _ in SAMPLES:
        c = per_sample[name]
        tot = sum(c.values())
        primer = name.split("_")[2]
        print(f"--- {name}  ({tot} reads, primer {primer} "
              f"{'RE' if primer == '251' else 'LE'})")
        if not tot:
            print("    none\n")
            continue
        # Consensus across all reads of this sample, per position
        cons = "".join(
            Counter(seq[p] for seq in c.elements()).most_common(1)[0][0]
            for p in range(n_r2))
        frac = [max(Counter(seq[p] for seq in c.elements()).values()) / tot
                for p in range(n_r2)]
        print(f"    consensus: {cons}")
        print("    identity:  " + "".join(
            "*" if f > 0.9 else ("+" if f > 0.5 else ".") for f in frac))
        # Where does the constant transposon end stop?
        const_len = 0
        for f in frac:
            if f <= 0.9:
                break
            const_len += 1
        print(f"    constant prefix: {const_len} nt -> {cons[:const_len]}")
        pr = PRIMERS[primer]
        print(f"    primer 3' end:   ...{pr[-12:]}  (primer in read? "
              f"{'yes' if pr[-12:] in cons else 'no'})\n")


if __name__ == "__main__":
    main()
