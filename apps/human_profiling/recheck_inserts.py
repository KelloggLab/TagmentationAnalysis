"""Re-measure the distance from the LE/RE primer to the Tn5 adapter, tolerating
sequencing errors.

The first pass used an exact 15-mer probe for the mosaic end, which misses any
adapter carrying an error and any adapter truncated at the read start. This
scans for the full mosaic end, its reverse complement, the mosaic-end core and
the Nextera adapter body, each with a mismatch budget, and takes the earliest
hit. The resulting insert length is the honest one.

usage: python3 -I recheck_inserts.py <BaseCalls/L001> <ntiles> <out.json>
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# -I drops the script's directory from sys.path; add it back so `runconfig`
# resolves, plus the repository root so `core` does too, with or without an
# editable install.
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.abspath(os.path.join(HERE, "..", "..")))

import runconfig  # noqa: E402
from core.barcodes import one_mismatch_map, rc  # noqa: E402
from core.read_cbcl import read_tile  # noqa: E402

CFG = runconfig.load()

I1_START, I1_LEN = CFG.cycles.i1_start, CFG.cycles.i1_len
I2_START, I2_LEN = CFG.cycles.i2_start, CFG.cycles.i2_len
R2_START, N_R2 = CFG.cycles.r2_start, CFG.cycles.r2_len
UMI_LEN = CFG.cycles.umi_len

SAMPLES = [(s.sample, s.arm, s.i5, s.i7) for s in CFG.samples]
# Primer *lengths* here, not sequences: this script only needs to know where
# the insert starts. Derived, so it cannot disagree with the sequences.
PRIMER = {name: len(a.primer) for name, a in CFG.arms.items()}
ME = CFG.mosaic_end


# (sequence, mismatch budget). The core and the adapter body catch a mosaic end
# that is cut off by the start of the read.
PROBES = [(ME, 2), (rc(ME), 2),
          ("TATACACATCT", 1), ("AGATGTGTATA", 1),
          ("GACGCTGCCGACGA", 2), ("TCGTCGGCAGCGTC", 2)]


def earliest_adapter(M):
    """First column where any adapter probe matches within budget, per row."""
    n, L = M.shape
    first = np.full(n, 10**6, dtype=np.int32)
    for seq, budget in PROBES:
        ref = np.frombuffer(seq.encode(), dtype=np.uint8)
        k = len(ref)
        for o in range(L - k + 1):
            hit = (M[:, o:o + k] != ref).sum(axis=1) <= budget
            np.minimum(first, np.where(hit, o, 10**6), out=first)
    return first


def main():
    lane, ntiles, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    i7m = one_mismatch_map([rc(i7) for _, _, _, i7 in SAMPLES])
    i5m = one_mismatch_map([rc(i5) for _, _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): (s, a) for s, a, i5, i7 in SAMPLES}

    hist = {"RE": Counter(), "LE": Counter()}
    per_group = {}
    total = 0

    for t in range(ntiles):
        i1, _ = read_tile(lane, I1_START, I1_LEN, t)
        i2, _ = read_tile(lane, I2_START, I2_LEN, t)
        r2, _ = read_tile(lane, R2_START, N_R2, t)
        total += i1.shape[0]
        print(f"tile {t}: {i1.shape[0]} clusters", file=sys.stderr)
        first = earliest_adapter(r2)

        for k in range(i1.shape[0]):
            bc7 = i7m.get(i1[k].tobytes().decode())
            bc5 = i5m.get(i2[k, UMI_LEN:].tobytes().decode())
            if bc7 is None or bc5 is None:
                continue
            g = combo.get((bc7, bc5))
            if g is None:
                continue
            key = f"{g[0]}_{g[1]}"
            d = per_group.setdefault(key, Counter())
            d["assigned"] += 1
            p = int(first[k])
            ins = (p - PRIMER[g[1]]) if p < 10**6 else None
            if ins is None:
                d["no_adapter_in_200"] += 1
                hist[g[1]][">170"] += 1
            else:
                ins = max(ins, 0)
                hist[g[1]][min(ins, 170)] += 1
                if ins < 30:
                    d["lt30"] += 1
                elif ins < 100:
                    d["b30_99"] += 1
                else:
                    d["ge100"] += 1

    asg = sum(d["assigned"] for d in per_group.values())
    out = {"tiles": ntiles, "total_clusters": total, "assigned": asg,
           "groups": {}, "hist": {}}
    print(f"\n{'group':<10} {'assigned':>9} {'<30nt':>7} {'30-99':>7} "
          f"{'>=100':>7} {'no adapter':>11}")
    for key, d in sorted(per_group.items()):
        a = d["assigned"]
        out["groups"][key] = {k: d[k] for k in
                              ("assigned", "lt30", "b30_99", "ge100",
                               "no_adapter_in_200")}
        print(f"{key:<10} {a:>9} {100*d['lt30']/a:>6.1f}% "
              f"{100*d['b30_99']/a:>6.1f}% {100*d['ge100']/a:>6.1f}% "
              f"{100*d['no_adapter_in_200']/a:>10.1f}%")
    for arm, c in hist.items():
        out["hist"][arm] = {str(k): v for k, v in sorted(
            c.items(), key=lambda x: (x[0] == ">170", x[0]))}
    tot_lt30 = sum(d["lt30"] for d in per_group.values())
    tot_long = sum(d["ge100"] + d["no_adapter_in_200"] for d in per_group.values())
    out["pct_lt30"] = 100 * tot_lt30 / asg
    out["pct_ge100_or_none"] = 100 * tot_long / asg
    print(f"\npooled: {out['pct_lt30']:.1f}% under 30 nt, "
          f"{out['pct_ge100_or_none']:.1f}% at 100 nt or more")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
