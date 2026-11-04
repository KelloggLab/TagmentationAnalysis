"""Sequences needed to redesign the arm primers and a blocking clamp.

usage: python3 -I design_helper.py <donor.fasta>
"""
import sys

PRIMER3 = {"RE": 121, "LE": 2168}
TERMINUS = {"RE": 59, "LE": 2190}
STRAND = {"RE": -1, "LE": +1}


def rc(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def gc(s):
    return 100 * (s.count("G") + s.count("C")) / len(s)


def tm(s):
    """Nearest-neighbour is overkill here; Wallace-corrected is close enough."""
    return 64.9 + 41 * (s.count("G") + s.count("C") - 16.4) / len(s)


def main():
    seq = "".join(l.strip() for l in open(sys.argv[1]) if not l.startswith(">")).upper()
    L = len(seq)
    dbl = seq + seq

    for arm in ("RE", "LE"):
        t, s = TERMINUS[arm], STRAND[arm]
        print(f"\n===== {arm}  terminus at {t}, read runs "
              f"{'5->3 up' if s > 0 else '3->5 down'} the map")
        if s > 0:
            inside = seq[t - 60:t]          # last 60 nt inside the element
            outside = dbl[t:t + 40]         # first 40 nt of backbone
        else:
            inside = rc(seq[t - 1:t + 59])
            outside = rc(dbl[t - 41 + L:t - 1 + L]) if t < 41 else rc(seq[t - 41:t - 1])
        print(f"  last 60 nt inside the element (read orientation):\n    {inside}")
        print(f"  first 40 nt outside, i.e. where a junction diverges:\n    {outside}")
        print(f"  clamp across the boundary (14 nt): {inside[-7:] + outside[:7]}")
        print(f"  clamp across the boundary (18 nt): {inside[-9:] + outside[:9]}")

        print("  candidate primers, 3' end this far inside the terminus:")
        for back in (20, 25, 30):
            for ln in (20, 22, 24):
                p = inside[len(inside) - back - ln: len(inside) - back]
                if len(p) < ln:
                    continue
                print(f"    {back:>2} nt inside, {ln} nt: {p}  "
                      f"GC {gc(p):.0f}%  Tm~{tm(p):.0f}C")
            print()


if __name__ == "__main__":
    main()
