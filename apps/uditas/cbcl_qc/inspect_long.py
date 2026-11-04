"""Look at the reads that survive error-tolerant adapter detection with a long
insert, and decide whether they are real genomic flank.

Two tests: sequence complexity (a no-signal read is all one base), and whether
the same flank turns up in both cell lines (it cannot, if it is genomic).

usage: python3 -I inspect_long.py <BaseCalls/L001> <ntiles>
"""
import sys, os
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from recheck_inserts import (SAMPLES, PRIMER, rc, read_block, one_mismatch_map,
                             earliest_adapter, I1_START, I1_LEN, I2_START,
                             I2_LEN, R2_START, N_R2, UMI_LEN)

MIN_INS = 100

def main():
    lane, ntiles = sys.argv[1], int(sys.argv[2])
    i7m = one_mismatch_map([rc(i7) for _, _, _, i7 in SAMPLES])
    i5m = one_mismatch_map([rc(i5) for _, _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): (s, a) for s, a, i5, i7 in SAMPLES}

    kept = {}          # (sample, arm) -> Counter of 40-nt flank keys
    lowcx = Counter()
    tot = Counter()
    for t in range(ntiles):
        i1 = read_block(lane, I1_START, I1_LEN, t)
        i2 = read_block(lane, I2_START, I2_LEN, t)
        r2 = read_block(lane, R2_START, N_R2, t)
        print(f"tile {t}", file=sys.stderr)
        first = earliest_adapter(r2)
        for k in range(i1.shape[0]):
            b7 = i7m.get(i1[k].tobytes().decode())
            b5 = i5m.get(i2[k, UMI_LEN:].tobytes().decode())
            if b7 is None or b5 is None:
                continue
            g = combo.get((b7, b5))
            if g is None:
                continue
            p = int(first[k])
            ins = (p - PRIMER[g[1]]) if p < 10**6 else N_R2 - PRIMER[g[1]]
            if ins < MIN_INS:
                continue
            seq = r2[k].tobytes().decode()
            flank = seq[PRIMER[g[1]]: PRIMER[g[1]] + 100]
            tot[g] += 1
            c = Counter(flank)
            # one base over 60% of the window, or an N run, is no signal
            if c.most_common(1)[0][1] > 60 or c["N"] > 10:
                lowcx[g] += 1
                continue
            kept.setdefault(g, Counter())[flank[:40]] += 1

    print(f"\n{'group':<10} {'long reads':>11} {'low-complexity':>15} {'informative':>12}")
    for g in sorted(tot):
        print(f"{g[0]+'_'+g[1]:<10} {tot[g]:>11} "
              f"{100*lowcx[g]/tot[g]:>14.1f}% {sum(kept.get(g,{}).values()):>12}")

    for arm in ("RE", "LE"):
        ft = kept.get(("FT", arm), Counter())
        tt = kept.get(("293T", arm), Counter())
        shared = set(ft) & set(tt)
        sf = sum(ft[f] for f in shared); st = sum(tt[f] for f in shared)
        print(f"\n{arm}: {len(shared)} flanks in both; "
              f"{100*sf/max(sum(ft.values()),1):.1f}% of FT, "
              f"{100*st/max(sum(tt.values()),1):.1f}% of 293T informative reads")
        for f in sorted(shared, key=lambda x: -(ft[x]+tt[x]))[:5]:
            print(f"   FT {ft[f]:>5} / 293T {tt[f]:>5}  {f}")
        print("  FT-only top:")
        for f, n in [(f, n) for f, n in ft.most_common(40) if f not in shared][:5]:
            print(f"   {n:>5}  {f}")

if __name__ == "__main__":
    main()
