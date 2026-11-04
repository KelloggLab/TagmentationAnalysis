"""Donor fraction, measured against the real plasmid map.

The first pass guessed where the transposon end stops by watching per-position
agreement fall away. With a donor-dominated library that guess is circular, and
for the RE arm it was badly wrong: the primer sits 62 bases inside the end, not
12. This takes both the expected donor sequence and the terminus straight from
the SnapGene map, so a read only counts as donor if it keeps matching past the
terminus, where a real junction would have turned into genomic DNA.

usage: python3 -I donor_fraction2.py <donor.fasta> <BaseCalls/L001> <ntiles>
"""
import sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from recheck_inserts import (SAMPLES, PRIMER, rc, read_block, one_mismatch_map,
                             earliest_adapter, I1_START, I1_LEN, I2_START,
                             I2_LEN, R2_START, N_R2, UMI_LEN)

# From donor_layout.py, 1-based on the map:
#   RE primer 3' end at 121, reading down; RE terminus at 59   -> 62 nt of end
#   LE primer 3' end at 2168, reading up;  LE terminus at 2190 -> 22 nt of end
PRIMER3 = {"RE": 121, "LE": 2168}
TERMINUS = {"RE": 59, "LE": 2190}
STRAND = {"RE": -1, "LE": +1}
PAST = 15      # agreement past the terminus that settles it as donor
FLANK = 30     # genomic flank a read must reach to be worth mapping
BUDGET = 3
REFLEN = 180


def load(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">")).upper()


def donor_ref(seq, arm):
    """What the read sees after the primer, if the template is the plasmid."""
    L = len(seq)
    dbl = seq + seq
    i = PRIMER3[arm]                      # 1-based 3' end == 0-based index of next base
    if STRAND[arm] > 0:
        return dbl[i:i + REFLEN]
    return rc(dbl[i - REFLEN + L: i + L])[:REFLEN] if i < REFLEN else rc(seq[i - REFLEN:i])


def agree_len(M, plen, ref):
    # the reference can outrun the sequenced window, so trim it to fit
    a = np.frombuffer(ref[:M.shape[1] - plen].encode(), dtype=np.uint8)
    bad = np.cumsum(M[:, plen:plen + len(a)] != a, axis=1)
    over = bad > BUDGET
    return np.where(over.any(axis=1), over.argmax(axis=1), len(a))


def main():
    fasta, lane, ntiles = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seq = load(fasta)
    end_len = {a: abs(TERMINUS[a] - PRIMER3[a]) for a in PRIMER3}
    refs = {a: donor_ref(seq, a) for a in PRIMER3}
    for a in refs:
        print(f"{a}: {end_len[a]} nt of transposon end, then "
              f"{refs[a][end_len[a]:end_len[a]+24]} on the plasmid", file=sys.stderr)

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
        agree = {a: agree_len(r2, PRIMER[a], refs[a]) for a in refs}

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
            el = end_len[arm]
            reach = int(adapt[k]) - PRIMER[arm]          # bases before the adapter
            if int(agree[arm][k]) >= el + PAST:
                d["donor"] += 1
            elif reach < el + PAST:
                d["never_left_the_end"] += 1
            elif reach < el + FLANK:
                d["too_little_flank"] += 1
            else:
                d["candidate"] += 1

    print(f"\n{'group':<10} {'assigned':>9} {'donor':>8} {'stuck in end':>13} "
          f"{'thin flank':>12} {'candidate':>11}")
    tot = Counter()
    for g in sorted(stat):
        d = stat[g]; a = d["assigned"]; tot.update(d)
        print(f"{g[0]+'_'+g[1]:<10} {a:>9} {100*d['donor']/a:>7.2f}% "
              f"{100*d['never_left_the_end']/a:>12.2f}% "
              f"{100*d['too_little_flank']/a:>11.2f}% "
              f"{100*d['candidate']/a:>10.2f}%")
    a = tot["assigned"]
    print(f"\npooled: {100*tot['donor']/a:.2f}% donor, "
          f"{100*tot['never_left_the_end']/a:.2f}% never left the transposon end, "
          f"{100*tot['too_little_flank']/a:.2f}% too little flank, "
          f"{100*tot['candidate']/a:.2f}% candidate")


if __name__ == "__main__":
    main()
