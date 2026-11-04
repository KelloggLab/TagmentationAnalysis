"""Measure what fraction of the library is provably donor.

For each assigned read, walk the sequence after the LE/RE primer against the
known donor continuation and see how far it agrees. Agreement that runs past
the transposon end means the molecule was donor plasmid, not an insertion
junction, because past the end a real junction carries genomic sequence.

usage: python3 -I donor_fraction.py <BaseCalls/L001> <ntiles>
"""
import sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from recheck_inserts import (SAMPLES, PRIMER, rc, read_block, one_mismatch_map,
                             earliest_adapter, I1_START, I1_LEN, I2_START,
                             I2_LEN, R2_START, N_R2, UMI_LEN)

# Sequence the read sees after the primer, taken from the data itself: the
# transposon end, then whatever the donor plasmid carries beyond it.
DONOR = {
    "RE": "TGTAGTGTTGTTGACGGCAAAGTTATGGTTCCAGTAACGGCAAAGTTATGGGT",
    "LE": "CAAAGTTATGGGTAAAGTCACAATCCCAATGGCGCGCCGAGCTTGGCGTAATCATGGTC",
}
# Bases of transposon end past the primer; beyond this the donor and a real
# junction part company.
END_LEN = {"RE": 12, "LE": 19}
PAST = 15      # bases of donor agreement past the end that settle it
BUDGET = 3     # mismatches tolerated while walking


def agree_len(M, plen, ref):
    """How far each read agrees with ref, starting at column plen."""
    a = np.frombuffer(ref.encode(), dtype=np.uint8)
    seg = M[:, plen:plen + len(a)]
    bad = np.cumsum(seg != a, axis=1)
    over = bad > BUDGET
    n = np.where(over.any(axis=1), over.argmax(axis=1), len(a))
    return n


def main():
    lane, ntiles = sys.argv[1], int(sys.argv[2])
    i7m = one_mismatch_map([rc(i7) for _, _, _, i7 in SAMPLES])
    i5m = one_mismatch_map([rc(i5) for _, _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): (s, a) for s, a, i5, i7 in SAMPLES}

    stat = {}
    for t in range(ntiles):
        i1 = read_block(lane, I1_START, I1_LEN, t)
        i2 = read_block(lane, I2_START, I2_LEN, t)
        r2 = read_block(lane, R2_START, N_R2, t)
        print(f"tile {t}", file=sys.stderr)
        adapt = earliest_adapter(r2)
        agree = {arm: agree_len(r2, PRIMER[arm], DONOR[arm]) for arm in DONOR}

        for k in range(i1.shape[0]):
            b7 = i7m.get(i1[k].tobytes().decode())
            b5 = i5m.get(i2[k, UMI_LEN:].tobytes().decode())
            if b7 is None or b5 is None:
                continue
            g = combo.get((b7, b5))
            if g is None:
                continue
            arm = g[1]
            d = stat.setdefault(g, Counter())
            d["assigned"] += 1
            need = END_LEN[arm] + PAST
            if int(agree[arm][k]) >= need:
                d["donor"] += 1
            elif int(adapt[k]) - PRIMER[arm] < need:
                d["too_short_to_tell"] += 1
            else:
                d["not_donor"] += 1

    print(f"\n{'group':<10} {'assigned':>9} {'donor':>8} {'too short':>11} "
          f"{'not donor':>11}")
    tot = Counter()
    for g in sorted(stat):
        d = stat[g]; a = d["assigned"]; tot.update(d)
        print(f"{g[0]+'_'+g[1]:<10} {a:>9} {100*d['donor']/a:>7.1f}% "
              f"{100*d['too_short_to_tell']/a:>10.1f}% "
              f"{100*d['not_donor']/a:>10.1f}%")
    a = tot["assigned"]
    print(f"\npooled: {100*tot['donor']/a:.1f}% provably donor, "
          f"{100*tot['too_short_to_tell']/a:.1f}% too short to tell, "
          f"{100*tot['not_donor']/a:.1f}% not donor")
    print(f"donor : not-donor = {tot['donor']/max(tot['not_donor'],1):.0f} : 1")


if __name__ == "__main__":
    main()
