"""Decode I1, I2 and the start of R2 from CBCL for one tile, demultiplex in
memory, and show the consensus R2 prefix per sample.

Cycle layout for 261030_VH00889_534_AAJLG3JM5: R1 1-301, I1 302-309,
I2 310-327, R2 328-628.

usage: python -I verify_structure.py <BaseCalls/L001> [ntiles] [n_r2_cycles]
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
# -I drops the script's directory from sys.path; add it back so `runconfig`
# resolves, plus the repository root so `core` does too, with or without an
# editable install.
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.abspath(os.path.join(HERE, "..", "..")))

import runconfig  # noqa: E402
from core.barcodes import one_mismatch_map, rc  # noqa: E402
from core.illumina import read_tiles  # noqa: E402

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


def main():
    lane_dir = sys.argv[1]
    ntiles = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_r2 = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    i1, _ = read_tiles(lane_dir, I1_START, I1_LEN, ntiles)
    i2, _ = read_tiles(lane_dir, I2_START, I2_LEN, ntiles)
    r2, _ = read_tiles(lane_dir, R2_START, n_r2, ntiles)
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
        arm = ARM_OF[name]
        print(f"--- {name}  ({tot} reads, {arm} arm)")
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
        pr = PRIMERS[arm]
        print(f"    primer 3' end:   ...{pr[-12:]}  (primer in read? "
              f"{'yes' if pr[-12:] in cons else 'no'})\n")


if __name__ == "__main__":
    main()
