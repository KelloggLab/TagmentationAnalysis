"""Locating guide RNAs, PAMs and spacers in a genome.

`map_guide_to_genome` is the one to reach for: it searches both strands and
returns a list of dicts with coordinates, strand and alignment score.
`sgRNA_finder` answers the same question with a cruder interface -- a 4-tuple
of bare integers, and a printed message instead of an exception when the guide
is absent -- and is kept because existing notebooks call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from Bio.Seq import Seq

from .motifs import iupac_match, pam_matches_at
from .sequence import reverse_complement


@dataclass(frozen=True)
class SpacerHit:
    strand: Literal["+","-"]
    pam_start: int
    pam_end: int
    spacer_start: int
    spacer_end: int
    pam_dna: str
    spacer_top: str   # DNA by default


def map_guide_to_genome(guide, genome):
    """
    Map a guide RNA (DNA alphabet) to a genome sequence.

    Parameters
    ----------
    guide : str
        Guide RNA sequence (DNA alphabet: A/C/G/T)
    genome : str
        Genome sequence (DNA alphabet)

    Returns
    -------
    hits : list of dict
        Each hit contains:
        - start: 0-based start index in genome
        - end:   end index (exclusive)
        - strand: '+' or '-'
        - sequence: matched genome sequence
    """
    guide = guide.upper()
    genome = genome.upper()

    rc_guide = reverse_complement(guide)
    guide_len = len(guide)

    hits = []

    # Search forward strand
    start = 0
    while True:
        idx = genome.find(guide, start)
        if idx == -1:
            break
        hits.append({
            "start": idx,
            "end": idx + guide_len,
            "strand": "+",
            "sequence": genome[idx:idx + guide_len]
        })
        start = idx + 1

    # Search reverse strand
    start = 0
    while True:
        idx = genome.find(rc_guide, start)
        if idx == -1:
            break
        hits.append({
            "start": idx,
            "end": idx + guide_len,
            "strand": "-",
            "sequence": genome[idx:idx + guide_len]
        })
        start = idx + 1

    return hits


def sgRNA_finder(sgRNA, genome, PAM_len, natural_system):
    sgRNA = sgRNA.upper()
    genome = genome.upper()
    
    if natural_system:
        if genome.find(sgRNA) != -1:
            return genome.find(sgRNA), genome.find(sgRNA), genome.find(sgRNA) - PAM_len, False
        elif genome.find(str(Seq(sgRNA).reverse_complement())) != -1:
            return genome.find(str(Seq(sgRNA).reverse_complement())), genome.find(str(Seq(sgRNA).reverse_complement()))  + len(sgRNA), genome.find(str(Seq(sgRNA).reverse_complement())) + len(sgRNA) + PAM_len, True
        else:
            print("sgRNA Sequence Not Found in Genome")
    
    else:
        if genome.find(sgRNA) != -1:
            return genome.find(sgRNA), genome.find(sgRNA) + len(sgRNA), genome.find(sgRNA) + len(sgRNA) + PAM_len, False
        elif genome.find(str(Seq(sgRNA).reverse_complement())) != -1:
            return genome.find(str(Seq(sgRNA).reverse_complement())), genome.find(str(Seq(sgRNA).reverse_complement())), genome.find(str(Seq(sgRNA).reverse_complement())) - PAM_len, True
        else:
            print("sgRNA Sequence Not Found in Genome")


def extract_sequence_windows(positions, genome, window_bp):
    """
    Extract genomic sequence windows around given positions.

    Parameters
    ----------
    positions : list of int
        Genomic coordinates (0-based).
    genome : str
        Genome sequence.
    window_bp : int
        Number of base pairs to include on each side of the position.

    Returns
    -------
    list of dict
        Each entry contains:
        - position: original genomic position
        - start: window start (clipped)
        - end: window end (exclusive)
        - sequence: extracted sequence
    """
    genome = genome.upper()
    genome_len = len(genome)

    windows = []

    for pos in positions:
        if pos < 0 or pos >= genome_len:
            continue  # skip invalid positions

        start = max(0, pos - window_bp)
        end = min(genome_len, pos + window_bp + 1)

        windows.append({
            "position": pos,
            "start": start,
            "end": end,
            "sequence": genome[start:end]
        })

    return windows


def extract_spacers(
    sequence: str,
    pam: str,
    spacer_len: int = 20,
    spacer_side: Literal["5prime", "3prime"] = "5prime",
    search_strands: Literal["+", "-", "both"] = "both",
    return_rna: bool = False,   # <-- DNA is now default
) -> List[SpacerHit]:
    """
    Extract all possible spacer sequences adjacent to a PAM.
    by default sequences always returned as top strand sequence, 
    so reverse complement needs to be applied to get actual sequence on bottom strand. 
    Returns DNA spacers by default. Set return_rna=True to return RNA (T->U).
    """
    seq = sequence.replace(" ", "").replace("\n", "")
    pam = pam.upper()

    def dna_or_rna(s: str) -> str:
        return s.replace("T", "U") if return_rna else s

    hits: List[SpacerHit] = []

    # + strand
    def scan_plus():
        for i in range(len(seq) - len(pam) + 1):
            if not pam_matches_at(seq, i, pam):
                continue

            pam_start, pam_end = i, i + len(pam)
            if spacer_side == "5prime":
                spacer_start, spacer_end = pam_start - spacer_len, pam_start
            else:
                spacer_start, spacer_end = pam_end, pam_end + spacer_len

            if spacer_start < 0 or spacer_end > len(seq):
                continue

            spacer = seq[spacer_start:spacer_end].upper().replace("U", "T")
            pam_seq = seq[pam_start:pam_end].upper().replace("U", "T")

            hits.append(
                SpacerHit(
                    strand="+",
                    pam_start=pam_start,
                    pam_end=pam_end,
                    spacer_start=spacer_start,
                    spacer_end=spacer_end,
                    pam_dna=pam_seq,
                    spacer_top=dna_or_rna(spacer),
                )
            )

    # - strand
    def scan_minus():
        rc = reverse_complement(seq.replace("U", "T"))
        L = len(seq)

        for i in range(len(rc) - len(pam) + 1):
            if not pam_matches_at(rc, i, pam):
                continue

            pam_start = L - (i + len(pam))
            pam_end = L - i

            if spacer_side == "5prime":
                spacer_start, spacer_end = pam_end, pam_end + spacer_len
            else:
                spacer_start, spacer_end = pam_start - spacer_len, pam_start

            if spacer_start < 0 or spacer_end > L:
                continue

            spacer_plus = seq[spacer_start:spacer_end].upper().replace("U", "T")
            spacer_guide = reverse_complement(spacer_plus)
            pam_seq = seq[pam_start:pam_end].upper().replace("U", "T")

            hits.append(
                SpacerHit(
                    strand="-",
                    pam_start=pam_start,
                    pam_end=pam_end,
                    spacer_start=spacer_start,
                    spacer_end=spacer_end,
                    pam_dna=pam_seq,
                    spacer_top=spacer_plus,
                )
            )

    if search_strands in ("+", "both"):
        scan_plus()
    if search_strands in ("-", "both"):
        scan_minus()

    return hits


def find_pam_sites_with_optional_protospacer(
    segment: str,
    pam_motif: str,
    protospacer: Optional[str] = None,
    pam_side: str = "3prime",                 # "3prime"/"3'" or "5prime"/"5'"
    include_reverse_complement: bool = False, # scan both strands
) -> List[Dict]:
    """
    Find PAM sites in `segment` using an IUPAC-degenerate PAM motif.
    If `protospacer` is provided, require an adjacent exact protospacer match.

    pam_side interpretation (on the scanned strand):
      - "3prime": protospacer immediately upstream of PAM (protospacer + PAM)
      - "5prime": protospacer immediately downstream of PAM (PAM + protospacer)

    Returns a list of dict hits with coordinates in the + orientation of `segment`.
    """
    # normalize
    segment = segment.upper().replace("U", "T")
    pam_motif = pam_motif.upper().replace("U", "T")
    if protospacer is not None:
        protospacer = protospacer.upper().replace("U", "T")

    # normalize pam_side
    ps = pam_side.strip().lower()
    if ps in {"3", "3prime", "3'"}:
        ps = "3prime"
    elif ps in {"5", "5prime", "5'"}:
        ps = "5prime"
    else:
        raise ValueError("pam_side must be '3prime'/'3'' or '5prime'/'5'' (or '3'/'5').")

    pam_len = len(pam_motif)
    if pam_len == 0:
        raise ValueError("pam_motif cannot be empty.")

    proto_len = len(protospacer) if protospacer is not None else 0

    hits: List[Dict] = []

    def scan(seq: str, strand: str):
        # seq is the sequence in the orientation being scanned
        for pam_start in range(0, len(seq) - pam_len + 1):
            pam_window = seq[pam_start:pam_start + pam_len]
            if not iupac_match(pam_window, pam_motif):
                continue

            # If no protospacer requirement, record PAM hit
            if protospacer is None:
                # Map to + coords
                pos_plus = pam_start if strand == "+" else (len(segment) - pam_len - pam_start)
                hits.append({
                    "pam_start": pos_plus,
                    "pam_end": pos_plus + pam_len,   # end-exclusive in + coords
                    "strand": strand,
                    "pam_seq": pam_window,
                })
                continue

            # Determine where protospacer must lie relative to PAM
            if ps == "3prime":
                # protospacer + PAM => protospacer immediately upstream
                proto_start = pam_start - proto_len
                proto_end = pam_start
            else:
                # PAM + protospacer => protospacer immediately downstream
                proto_start = pam_start + pam_len
                proto_end = proto_start + proto_len

            if proto_start < 0 or proto_end > len(seq):
                continue

            proto_window = seq[proto_start:proto_end]
            if proto_window != protospacer:
                continue

            # Map both PAM and protospacer coordinates to + orientation of original segment
            if strand == "+":
                pam_start_plus = pam_start
                proto_start_plus = proto_start
            else:
                # For RC scanning, a window starting at i maps to + start = L - window_len - i
                L = len(segment)
                pam_start_plus = L - pam_len - pam_start
                proto_start_plus = L - proto_len - proto_start

            hits.append({
                "pam_start": pam_start_plus,
                "pam_end": pam_start_plus + pam_len,              # end-exclusive
                "protospacer_start": proto_start_plus,
                "protospacer_end": proto_start_plus + proto_len,  # end-exclusive
                "strand": strand,
                "pam_seq": pam_window,
                "protospacer_seq": proto_window,
                "pam_side": ps,
            })

    # scan + strand
    scan(segment, "+")

    # scan - strand (optional)
    if include_reverse_complement:
        scan(reverse_complement(segment), "-")

    # sort by PAM start then strand
    hits.sort(key=lambda d: (d["pam_start"], d["strand"]))
    return hits
