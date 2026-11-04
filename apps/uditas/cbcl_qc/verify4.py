"""Diagnose the short products: what sits between the LE/RE primer and the Tn5
mosaic end, and is it one dominant species or many?

usage: python -I verify4.py <BaseCalls/L001> [ntiles]
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


def main():
    lane_dir = sys.argv[1]
    ntiles = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    i1 = read_block(lane_dir, I1_START, I1_LEN, ntiles)
    i2 = read_block(lane_dir, I2_START, I2_LEN, ntiles)
    r2 = read_block(lane_dir, R2_START, N_R2, ntiles)

    i7_map = one_mismatch_map([rc(i7) for _, _, i7 in SAMPLES])
    i5_map = one_mismatch_map([rc(i5) for _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): name for name, i5, i7 in SAMPLES}

    inserts = {n: Counter() for n, _, _ in SAMPLES}
    lengths = {n: Counter() for n, _, _ in SAMPLES}
    after = {n: Counter() for n, _, _ in SAMPLES}
    which = {n: Counter() for n, _, _ in SAMPLES}

    for k in range(i1.shape[0]):
        bc7 = i7_map.get(i1[k].tobytes().decode())
        bc5 = i5_map.get(i2[k, UMI_LEN:].tobytes().decode())
        if bc7 is None or bc5 is None:
            continue
        name = combo.get((bc7, bc5))
        if name is None:
            continue
        seq = r2[k].tobytes().decode()
        plen = len(PRIMERS[name.split("_")[2]])
        for tag, probe in zip(("ME", "ME_rc"), PROBES):
            p = seq.find(probe)
            if p == -1 or p < plen:
                continue
            which[name][tag] += 1
            inserts[name][seq[plen:p]] += 1
            lengths[name][p - plen] += 1
            after[name][seq[p + 15:p + 35]] += 1
            break

    for name, _, _ in SAMPLES:
        c = inserts[name]
        tot = sum(c.values())
        if not tot:
            continue
        end = "RE" if name.split("_")[2] == "251" else "LE"
        print(f"=== {name} ({end})  {tot} reads with a mosaic end")
        print(f"    orientation: {dict(which[name])}")
        print(f"    distinct inserts: {len(c)}")
        for seq, k in c.most_common(5):
            print(f"      {100*k/tot:5.1f}%  len{len(seq):>3}  {seq if seq else '(empty)'}")
        ln = lengths[name]
        common_len = ", ".join(f"{L}nt:{100*v/tot:.0f}%" for L, v in ln.most_common(4))
        print(f"    top insert lengths: {common_len}")
        aft = after[name].most_common(1)[0]
        print(f"    after the ME: {aft[0]}  ({100*aft[1]/tot:.0f}%)\n")


if __name__ == "__main__":
    main()
