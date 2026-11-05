"""Demultiplexing helpers: error-tolerant barcode lookup, and the two string
operations that go with it.

`sequence.reverse_complement` does the same job as `rc` here, but that module
imports Biopython. These functions stay
dependency-free so the `python3 -I` scripts in `apps/human_profiling` can read
a sequencing run with numpy alone.
"""
from __future__ import annotations

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def rc(seq):
    """Reverse complement of an upper-case sequence; N maps to N."""
    return seq.translate(_COMPLEMENT)[::-1]


def mism(a, b):
    """Mismatches between two sequences, ignoring no-calls on either side.

    Compares only up to the shorter of the two.
    """
    return sum(1 for x, y in zip(a, b) if x != y and x != "N" and y != "N")


def one_mismatch_map(barcodes):
    """Map every one-substitution variant of each barcode to that barcode.

    Lets a barcode with one sequencing error still be assigned. A variant
    reachable from two different barcodes is ambiguous and is dropped, unless
    it is itself one of the barcodes, in which case the exact match wins.
    """
    exact = set(barcodes)
    m = {bc: bc for bc in barcodes}
    ambiguous = set()
    for bc in barcodes:
        for i in range(len(bc)):
            for b in "ACGTN":
                v = bc[:i] + b + bc[i + 1:]
                if v in m and m[v] != bc:
                    ambiguous.add(v)
                else:
                    m.setdefault(v, bc)
    for v in ambiguous - exact:
        m.pop(v, None)
    return m
