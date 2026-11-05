"""Generic sequence and FASTA operations.

Nothing here knows about guides, insertions or any particular assay: these are
the string and file primitives the rest of `core/` is built on.
"""

from __future__ import annotations

import re

from Bio import SeqIO
from Bio.Align import PairwiseAligner


COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def load_genome(filename):
    from Bio import SeqIO
    import re
    genome_sequence = str(next(SeqIO.parse(filename, "fasta")).seq)
    genome_sequence = re.sub(r'^>.*\n?','',genome_sequence,flags=re.MULTILINE)
    genome_sequence = genome_sequence.replace('\n','').replace('\r','')
    return genome_sequence


def load_fasta_as_dict(fasta_path: str) -> dict[str, str]:
    seqs = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seqs[rec.id] = str(rec.seq).upper()
    return seqs


def fasta_to_string(fasta_path):
    return str(next(SeqIO.parse(fasta_path, "fasta")).seq)


def at_richness(sequence: str, *, ignore_ambiguous: bool = True) -> float:
        """
        Compute AT richness (fraction of A/T bases) for a nucleic-acid sequence.
        
        Parameters
        ----------
        sequence : str
        DNA/RNA sequence. 'U' is treated as 'T'. Case-insensitive.
        ignore_ambiguous : bool, default True
        If True, compute A/T fraction over only unambiguous bases (A/C/G/T).
        If False, compute A/T fraction over all characters except whitespace/gaps.
        
        Returns
        -------
        float
        AT richness in [0, 1]. Returns 0.0 if no valid bases are found.
        """
        if sequence is None:
            raise ValueError("sequence must be a string, not None")
            
        seq = sequence.upper().replace("U", "T")
        
        # Remove common whitespace and gap characters
        
        seq = "".join(ch for ch in seq if ch not in {" ", "\n", "\r", "\t", "-"})
        
        if ignore_ambiguous:
            valid = [ch for ch in seq if ch in {"A", "C", "G", "T"}]
            denom = len(valid)
            at = sum(ch in {"A", "T"} for ch in valid)
        else:
            denom = len(seq)
            at = sum(ch in {"A", "T"} for ch in seq)
            
        return (at / denom) if denom else 0.0


def sliding_window(s, window):
    return [s[i:i+window] for i in range(0, len(s) - window + 1)]


def at_sliding_window( seq, window ):
    #for window, compute at-richness and store in a list
    chunks = sliding_window( seq, window )
    at = []
    for ii in range(0,len(chunks)):
        at.append( at_richness( chunks[ ii ] ))
    return at


def local_align(query, target):
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -4
    aligner.extend_gap_score = -0.5

    aln = aligner.align(query, target)[0]  # best
    return aln
