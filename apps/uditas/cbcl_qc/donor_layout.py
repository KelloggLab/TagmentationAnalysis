"""Place the LE/RE primers on the donor map and work out what a donor read
looks like, so the observed contaminant sequences can be checked against the
map and a blocking restriction site can be chosen.

usage: python3 -I donor_layout.py <donor.fasta>
"""
import sys

PRIMER = {"RE": "AGCTGACGCAAGAATGAGGGT", "LE": "GCATAAGCTTGCCGTTGCGG"}
# What the reads actually showed after the primer, measured from the CBCLs.
OBSERVED = {
    "RE": "TGTAGTGTTGTTGACGGCAAAGTTATGGTTCCAGTAACGGCAAAGTTATGGGT",
    "LE": "CAAAGTTATGGGTAAAGTCACAATCCCAATGGCGCGCCGAGCTTGGCGTAATCATGGTC",
}
ENDS = {"RE": (59, 274), "LE": (2059, 2190)}

ENZYMES = {
    "AscI": "GGCGCGCC", "NotI": "GCGGCCGC", "FseI": "GGCCGGCC",
    "PacI": "TTAATTAA", "PmeI": "GTTTAAAC", "SbfI": "CCTGCAGG",
    "SwaI": "ATTTAAAT", "SrfI": "GCCCGGGC", "AsiSI": "GCGATCGC",
    "SgrAI": "CRCCGGYG", "RsrII": "CGGWCCG", "SexAI": "ACCWGGT",
    "MluI": "ACGCGT", "BssHII": "GCGCGC", "NruI": "TCGCGA",
    "SacII": "CCGCGG", "XhoI": "CTCGAG", "SalI": "GTCGAC",
    "KpnI": "GGTACC", "SphI": "GCATGC", "NheI": "GCTAGC",
    "AvrII": "CCTAGG", "SpeI": "ACTAGT", "BamHI": "GGATCC",
    "EcoRI": "GAATTC", "HindIII": "AAGCTT", "PstI": "CTGCAG",
    "AflII": "CTTAAG", "BstBI": "TTCGAA", "ClaI": "ATCGAT",
}
IUPAC = {"R": "AG", "Y": "CT", "W": "AT", "S": "CG", "K": "GT", "M": "AC",
         "N": "ACGT"}


def rc(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def matches(site, window):
    return all(b == w or (b in IUPAC and w in IUPAC[b])
               for b, w in zip(site, window))


def find_site(seq, site):
    n = len(site)
    return [i for i in range(len(seq) - n + 1) if matches(site, seq[i:i + n])]


def load(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">")).upper()


def main():
    seq = load(sys.argv[1])
    L = len(seq)
    dbl = seq + seq          # circular

    for arm in ("RE", "LE"):
        p = PRIMER[arm]
        fwd = [i for i in range(L) if dbl[i:i + len(p)] == p]
        rev = [i for i in range(L) if dbl[i:i + len(p)] == rc(p)]
        print(f"\n===== {arm} primer {p}")
        print(f"  forward hits {fwd}   reverse hits {rev}")
        s, e = ENDS[arm]
        print(f"  {arm} feature {s}..{e}")

        for i in fwd:
            tail = dbl[i + len(p): i + len(p) + 140]
            print(f"  + strand at {i+1}, 3' end {i+len(p)}, reads toward higher "
                  f"coordinates")
            print(f"    next 60: {tail[:60]}")
            print(f"    observed match: {tail.startswith(OBSERVED[arm][:30])}")
        for i in rev:
            tail = rc(dbl[max(0, i - 140): i])
            print(f"  - strand at {i+1}, 3' end {i+1}, reads toward lower "
                  f"coordinates")
            print(f"    next 60: {tail[:60]}")
            print(f"    observed match: {tail.startswith(OBSERVED[arm][:30])}")

    print("\n===== transposon ends in plasmid context")
    for arm, (s, e) in ENDS.items():
        print(f"  {arm} {s}..{e}  outside-left {seq[max(0,s-31):s-1]}")
        print(f"  {arm} {s}..{e}  outside-right {seq[e:e+30]}")

    print("\n===== unique/rare cutters, by position")
    for name, site in sorted(ENZYMES.items()):
        hits = sorted(set(find_site(dbl[:L + len(site)], site)))
        hits = [h % L + 1 for h in hits if h < L]
        if 1 <= len(hits) <= 3:
            print(f"  {name:<8} {site:<9} {hits}")


if __name__ == "__main__":
    main()
