"""Top distinct R2 start sequences per sample, to see whether the LE/RE end is
one variant or a mixture.

usage: python -I verify3.py <BaseCalls/L001> [ntiles]
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify_structure import (I1_LEN, I1_START, I2_LEN, I2_START, PRIMERS,
                              R2_START, SAMPLES, UMI_LEN, one_mismatch_map,
                              rc, read_block)


def main():
    lane_dir = sys.argv[1]
    ntiles = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    width = 24

    i1 = read_block(lane_dir, I1_START, I1_LEN, ntiles)
    i2 = read_block(lane_dir, I2_START, I2_LEN, ntiles)
    r2 = read_block(lane_dir, R2_START, width, ntiles)

    i7_map = one_mismatch_map([rc(i7) for _, _, i7 in SAMPLES])
    i5_map = one_mismatch_map([rc(i5) for _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): name for name, i5, i7 in SAMPLES}

    starts = {name: Counter() for name, _, _ in SAMPLES}
    for k in range(i1.shape[0]):
        bc7 = i7_map.get(i1[k].tobytes().decode())
        bc5 = i5_map.get(i2[k, UMI_LEN:].tobytes().decode())
        if bc7 is None or bc5 is None:
            continue
        name = combo.get((bc7, bc5))
        if name is not None:
            starts[name][r2[k].tobytes().decode()] += 1

    # One representative per primer is enough to see the pattern.
    for name in ("293_507_251_703", "293_507_253_704"):
        c = starts[name]
        tot = sum(c.values()) or 1
        primer = PRIMERS[name.split("_")[2]]
        print(f"=== {name}   primer {primer}")
        for seq, k in c.most_common(6):
            marks = "".join(
                " " if i < len(primer) and b == primer[i] else "^"
                for i, b in enumerate(seq))
            print(f"  {seq}  {100*k/tot:5.2f}%")
            print(f"  {marks}")
        print(f"  distinct starts: {len(c)} over {tot} reads\n")


if __name__ == "__main__":
    main()
