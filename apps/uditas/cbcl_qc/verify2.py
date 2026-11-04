"""Follow-up stats on R2 structure: primer match rate, no-call rate, and where
the Tn5 mosaic end appears (= end of the genomic insert).

usage: python -I verify2.py <BaseCalls/L001> [ntiles] [n_r2_cycles]
"""
import os
import sys
from collections import Counter

# -I drops the script's directory from sys.path; add just that one back.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify_structure import (I1_LEN, I1_START, I2_LEN, I2_START, PRIMERS,
                              R2_START, SAMPLES, UMI_LEN, one_mismatch_map,
                              rc, read_block)

ME = "AGATGTGTATAAGAGACAG"          # Tn5 mosaic end
ME_RC = rc(ME)


def mismatches(a, b):
    """Mismatches between equal-length strings, ignoring no-calls."""
    return sum(1 for x, y in zip(a, b) if x != y and x != "N" and y != "N")


def find_either(hay, needles, minlen=15):
    for nd in needles:
        probe = nd[:minlen]
        i = hay.find(probe)
        if i != -1:
            return i
    return -1


def main():
    lane_dir = sys.argv[1]
    ntiles = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_r2 = int(sys.argv[3]) if len(sys.argv) > 3 else 120

    i1 = read_block(lane_dir, I1_START, I1_LEN, ntiles)
    i2 = read_block(lane_dir, I2_START, I2_LEN, ntiles)
    r2 = read_block(lane_dir, R2_START, n_r2, ntiles)

    i7_map = one_mismatch_map([rc(i7) for _, _, i7 in SAMPLES])
    i5_map = one_mismatch_map([rc(i5) for _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): name for name, i5, i7 in SAMPLES}

    stats = {name: Counter() for name, _, _ in SAMPLES}
    me_pos = {name: Counter() for name, _, _ in SAMPLES}

    for k in range(i1.shape[0]):
        bc7 = i7_map.get(i1[k].tobytes().decode())
        bc5 = i5_map.get(i2[k, UMI_LEN:].tobytes().decode())
        if bc7 is None or bc5 is None:
            continue
        name = combo.get((bc7, bc5))
        if name is None:
            continue
        seq = r2[k].tobytes().decode()
        s = stats[name]
        s["reads"] += 1
        s["Ns_in_first20"] += seq[:20].count("N")

        primer = PRIMERS[name.split("_")[2]]
        obs = seq[:len(primer)]
        d = mismatches(obs, primer)
        s["primer_exact"] += d == 0
        s["primer_le2"] += d <= 2
        s["primer_le4"] += d <= 4

        i = find_either(seq, [ME, ME_RC])
        if i != -1:
            s["has_ME"] += 1
            me_pos[name][i] += 1

    print(f"{'sample':<18} {'reads':>8} {'%assigned':>9} {'primer=0':>9} "
          f"{'primer<=2':>10} {'N/20nt':>7} {'hasME':>7} {'medME':>6}")
    total = sum(stats[n]["reads"] for n, _, _ in SAMPLES)
    for name, _, _ in SAMPLES:
        s = stats[name]
        r = s["reads"]
        if not r:
            print(f"{name:<18} {0:>8}")
            continue
        mp = sorted(me_pos[name].elements())
        med = mp[len(mp) // 2] if mp else -1
        print(f"{name:<18} {r:>8} {100*r/total:>8.1f}% "
              f"{100*s['primer_exact']/r:>8.1f}% {100*s['primer_le2']/r:>9.1f}% "
              f"{s['Ns_in_first20']/r:>7.2f} {100*s['has_ME']/r:>6.1f}% {med:>6}")

    print("\nTn5 mosaic-end start position in R2 (insert length before ME), "
          "pooled percentiles:")
    allpos = sorted(p for name, _, _ in SAMPLES for p in me_pos[name].elements())
    if allpos:
        for q in (5, 25, 50, 75, 95):
            print(f"  p{q:<3} {allpos[int(len(allpos)*q/100)]}")
        print(f"  reads with ME in first {n_r2} nt: {len(allpos)}")


if __name__ == "__main__":
    main()
