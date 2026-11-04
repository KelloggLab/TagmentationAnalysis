"""Are the reads WITHOUT an early mosaic end genuine genomic flank (diverse) or
more donor sequence (constant)? Diversity decides if this run is salvageable.

usage: python -I verify5.py <BaseCalls/L001> [ntiles]
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify_structure import (I1_LEN, I1_START, I2_LEN, I2_START, PRIMERS,
                              R2_START, SAMPLES, UMI_LEN, one_mismatch_map,
                              rc, read_block)

ME = "AGATGTGTATAAGAGACAG"
PROBES = (ME[:15], rc(ME)[:15])
N_R2 = 150
WIN_OFF, WIN_LEN = 10, 30   # window of putative genomic flank past the primer


def main():
    lane_dir = sys.argv[1]
    ntiles = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    i1 = read_block(lane_dir, I1_START, I1_LEN, ntiles)
    i2 = read_block(lane_dir, I2_START, I2_LEN, ntiles)
    r2 = read_block(lane_dir, R2_START, N_R2, ntiles)

    i7_map = one_mismatch_map([rc(i7) for _, _, i7 in SAMPLES])
    i5_map = one_mismatch_map([rc(i5) for _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): name for name, i5, i7 in SAMPLES}

    longw = {n: Counter() for n, _, _ in SAMPLES}
    totals = Counter()

    for k in range(i1.shape[0]):
        bc7 = i7_map.get(i1[k].tobytes().decode())
        bc5 = i5_map.get(i2[k, UMI_LEN:].tobytes().decode())
        if bc7 is None or bc5 is None:
            continue
        name = combo.get((bc7, bc5))
        if name is None:
            continue
        totals[name] += 1
        seq = r2[k].tobytes().decode()
        if any(p in seq for p in PROBES):
            continue
        plen = len(PRIMERS[name.split("_")[2]])
        w = seq[plen + WIN_OFF: plen + WIN_OFF + WIN_LEN]
        if len(w) == WIN_LEN and "N" not in w:
            longw[name][w] += 1

    print(f"{'sample':<18} {'assigned':>9} {'no-ME':>8} {'%':>6} "
          f"{'distinct':>9} {'top%':>6} {'top5%':>7}")
    for name, _, _ in SAMPLES:
        c = longw[name]
        tot = sum(c.values())
        asg = totals[name]
        if not tot:
            print(f"{name:<18} {asg:>9} {0:>8}")
            continue
        top = c.most_common(5)
        print(f"{name:<18} {asg:>9} {tot:>8} {100*tot/asg:>5.1f}% "
              f"{len(c):>9} {100*top[0][1]/tot:>5.1f}% "
              f"{100*sum(v for _, v in top)/tot:>6.1f}%")

    print("\nMost common 30-nt windows in the no-ME reads (pooled):")
    pooled = Counter()
    for n, _, _ in SAMPLES:
        pooled.update(longw[n])
    ptot = sum(pooled.values()) or 1
    for seq, k in pooled.most_common(8):
        print(f"  {100*k/ptot:5.2f}%  {seq}")
    singles = sum(1 for v in pooled.values() if v == 1)
    print(f"\n  distinct windows: {len(pooled)}   seen once: {singles} "
          f"({100*singles/max(len(pooled),1):.0f}% of distinct)")


if __name__ == "__main__":
    main()
