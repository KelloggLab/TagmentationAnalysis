from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Literal, Dict
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Bio.Align import PairwiseAligner
from collections import Counter



COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def which(df, pos):
    return df[df["ins0"] == pos]

def between(df,s,e):
    subset = df[df["ins0"].between(s, e)]
    return subset

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def load_genome(filename):
    from Bio import SeqIO
    import re
    genome_sequence = str(next(SeqIO.parse('cJP003_assembly.fasta', "fasta")).seq)
    genome_sequence = re.sub(r'^>.*\n?','',genome_sequence,flags=re.MULTILINE)
    genome_sequence = genome_sequence.replace('\n','').replace('\r','')
    return genome_sequence



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


def local_align(query, target):
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -4
    aligner.extend_gap_score = -0.5

    aln = aligner.align(query, target)[0]  # best
    return aln

def percent_within_distance(insertions, target_position, window_bp):
    """
    Compute the percentage of insertions within ±window_bp of a target position.

    Parameters
    ----------
    insertions : iterable of int
        Genomic coordinates of insertions (e.g., start positions).
    target_position : int
        Genomic coordinate to compare against.
    window_bp : int
        Distance window in base pairs (±window_bp).

    Returns
    -------
    percent : float
        Percentage of insertions within the window.
    """
    if len(insertions) == 0:
        return 0.0

    count_within = sum(
        abs(pos - target_position) <= window_bp
        for pos in insertions
    )

    return 100.0 * count_within / len(insertions)

def load_fasta_as_dict(fasta_path: str) -> dict[str, str]:
    seqs = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seqs[rec.id] = str(rec.seq).upper()
    return seqs

def extract_tsd_centered_motifs_5p(
    tsv_path: str,
    genome_fasta: str,
    tsd_len: int = 5,
    flank_left: int = 10,
    flank_right: int = 10,
    min_mapq: Optional[int] = None,
    orient_to_plus: bool = True,
) -> pd.DataFrame:
    """
    DONOR-SIDE = 5' junction reads.

    ins0 in TSV is the *reference 5' end* of the read:
      strand '+' => ins0 corresponds to LEFT edge of the TSD copy adjacent to that junction
      strand '-' => ins0 corresponds to RIGHT edge of the TSD copy adjacent to that junction

    We compute tsd_start0 per row accordingly, then extract:
      [flank_left] + [TSD (tsd_len)] + [flank_right]

    If unique_insertions=True, we dedupe by (ref, tsd_start0) (not by ins0).
    """
    df = pd.read_csv(tsv_path, sep="\t")

    required = {"ref", "ins0", "strand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TSV missing required columns: {sorted(missing)}")

    if min_mapq is not None:
        if "mapq" not in df.columns:
            raise ValueError("min_mapq was set but TSV has no 'mapq' column.")
        df = df[df["mapq"] >= min_mapq].copy()

    # --- Compute true TSD start (0-based) for each read, strand-aware ---
    ins0 = df["ins0"].astype(int)
    strand = df["strand"].astype(str)

    # '+' : ins0 is left edge => tsd_start = ins0
    # '-' : ins0 is right edge => tsd_start = ins0 - (tsd_len - 1)
    df["tsd_start0"] = ins0.where(strand == "+", ins0 - (tsd_len - 1))

    # Drop impossible negative coordinates
    df = df[df["tsd_start0"] >= 0].copy()

    genome = load_fasta_as_dict(genome_fasta)

    motifs = []
    keep = []

    for idx, row in df.iterrows():
        ref = str(row["ref"])
        if ref not in genome:
            continue

        tsd_start0 = int(row["tsd_start0"])
        tsd_end_excl = tsd_start0 + tsd_len

        start = tsd_start0 - flank_left
        end_excl = tsd_end_excl + flank_right

        seq = genome[ref]
        if start < 0 or end_excl > len(seq):
            continue

        motif = seq[start:end_excl]  # flank_left + tsd_len + flank_right

        if orient_to_plus and row["strand"] == "-":
            motif = revcomp(motif)

        motifs.append(motif)
        keep.append(idx)

    out = df.loc[keep].copy()
    out["motif_seq"] = motifs
    out["motif_len"] = flank_left + tsd_len + flank_right
    out["flank_left"] = flank_left
    out["tsd_len"] = tsd_len
    out["flank_right"] = flank_right
    return out

def top_n_insertions_frequencies(insertions, N):
    """
    Return the top N most frequent insertion positions as frequencies.

    Parameters
    ----------
    insertions : list of int
        Insertion genomic positions.
    N : int
        Number of top positions to return.

    Returns
    -------
    list of tuple
        [(position, frequency), ...] sorted by descending frequency.
        Frequencies sum to <= 1.0.
    """
    if not insertions or N <= 0:
        return []

    total = len(insertions)
    counts = Counter(insertions)

    # Sort by frequency (desc), then position (asc)
    top = sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0])
    )[:N]

    # Convert counts -> frequencies
    top_freqs = [(pos, count / total) for pos, count in top]

    return top_freqs


@dataclass(frozen=True)
class SpacerHit:
    strand: Literal["+","-"]
    pam_start: int
    pam_end: int
    spacer_start: int
    spacer_end: int
    pam_dna: str
    spacer_5to3: str   # DNA by default


IUPAC_DNA: Dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "U": "T",
    "R": "[AG]",
    "Y": "[CT]",
    "S": "[GC]",
    "W": "[AT]",
    "K": "[GT]",
    "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
    "N": "[ACGT]",
}


def _pam_matches_at(seq: str, i: int, pam: str) -> bool:
    if i < 0 or i + len(pam) > len(seq):
        return False
    for j, code in enumerate(pam.upper()):
        base = seq[i + j].upper().replace("U", "T")
        allowed = IUPAC_DNA[code]
        if allowed.startswith("["):
            if base not in allowed.strip("[]"):
                return False
        else:
            if base != allowed:
                return False
    return True


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
            if not _pam_matches_at(seq, i, pam):
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
                    spacer_5to3=dna_or_rna(spacer),
                )
            )

    # - strand
    def scan_minus():
        rc = reverse_complement(seq.replace("U", "T"))
        L = len(seq)

        for i in range(len(rc) - len(pam) + 1):
            if not _pam_matches_at(rc, i, pam):
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
                    spacer_5to3=dna_or_rna(spacer_guide),
                )
            )

    if search_strands in ("+", "both"):
        scan_plus()
    if search_strands in ("-", "both"):
        scan_minus()

    return hits

def plot_insertion_profile(tsv,gRNA,genome_sequence,fig_filename,bins=100):
    df = pd.read_csv(tsv, sep="\t")
    df = df[df["mapq"] >= 30]
    positions = df["ins0"]

    gmap=map_guide_to_genome(gRNA,genome_sequence)
    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 3))
    
    ax.hist(
        positions,
        bins=bins)
    
    ax.set_xlabel("Genomic position (bp, 0-based)")
    ax.set_ylabel("Insertion count")
    #ax.set_yscale("log")

    
    # Draw vertical lines marking guide region
    ax.axvspan(
        gmap[0]['start'],
        gmap[0]['end'],
        alpha=0.3,
        label=f"guideRNA ({gmap[0]['strand']})"
    )

    ax.annotate(
        "",                              # no text
        xy=(gmap[0]['start'], 0),             # arrow tip (on x-axis)
        xytext=(gmap[0]['start'], -0.05),      # arrow tail (slightly below axis)
        arrowprops=dict(
            arrowstyle="->",
            color="green",
            linewidth=4
        ),
        annotation_clip=False
    )
    window_bp = 100
    pct_ontarget = percent_within_distance(positions, gmap[0]['start'], window_bp)
    ndx = gmap[0]['start']
    ax.set_title(f"Insertion site distribution: {pct_ontarget:.1f}% at site: {ndx:.1f} with window: {window_bp:.1f}")
    
    fig.tight_layout()
    #fig.savefig("figures/"+Path(tsv).stem+"insertion_histogram.pdf")
    fig.savefig(fig_filename)
    plt.show()

def characterize_insertion_from_position(ins,PAM, guideRNA, genome_sequence):
    import TagmentationAnalysis.postprocessHelpers as helper
    import pandas as pd
    
    pos = ins['ins0']
    window = 120
    ts_window = 10
    strand = ins['strand']
    if strand == '+':
        segment_to_search = genome_sequence[pos:pos+window]
        all_spacers = helper.extract_spacers( segment_to_search, 'NGG', 20, '5prime', search_strands='-')
    else:
        segment_to_search = genome_sequence[pos-window:pos]
        all_spacers = helper.extract_spacers( segment_to_search, 'NGG', 20, '5prime', search_strands='+')
    

    alns = []
    pam_dna = []
    pam_start = []
    pam_end = []
    protospacer_dna = []
    proto_start = []
    proto_end = []
    int_dna = []
    intervene_start = []
    intervene_end = []
    ts_dna = []
    ts_start = []
    ts_end = []
    insertion_length = []
    aln_score = []
    grna_strand = []
    
    df = pd.DataFrame(columns=["PAM","protospacer","gRNA_strand","intervening","insertion_length","target_site",
                               "PAM_start","PAM_end","proto_start","proto_end","aln_score",
                               "intervene_start","intervene_end","ts_start","ts_end"])
    
    for ii in range(0,len(all_spacers)):
        aa=helper.local_align( guideRNA, all_spacers[ii].spacer_5to3 )
        alns.append( aa )
        aln_score.append( aa.score )
        pam_dna.append(all_spacers[ii].pam_dna)
        protospacer_dna.append(all_spacers[ii].spacer_5to3)
        
        if strand == '+':
            pam_start.append(all_spacers[ii].pam_start+pos)
            pam_end.append(all_spacers[ii].pam_end+pos)
            proto_start.append(all_spacers[ii].spacer_start+pos)
            proto_end.append(all_spacers[ii].spacer_end+pos)
            intstart = pos
            intend = pos+all_spacers[ii].pam_end
            grna_strand.append('-')
        else:
            pam_start.append(all_spacers[ii].pam_start+pos-window)
            pam_end.append(all_spacers[ii].pam_end+pos-window)
            proto_start.append(all_spacers[ii].spacer_start+pos-window)
            proto_end.append(all_spacers[ii].spacer_end+pos-window)
            intstart = pos - window + all_spacers[ii].pam_end
            intend = pos
            grna_strand.append('+')
        insertion_length.append(intend-intstart+1)
        intervene_start.append(intstart) 
        intervene_end.append(intend)
        int_dna.append(genome_sequence[intstart:intend])
        ts_start.append(pos-ts_window)
        ts_end.append(pos+ts_window)
        ts_dna.append(genome_sequence[pos-ts_window:pos+ts_window])

    df['PAM'] = pam_dna
    df['protospacer']=protospacer_dna
    df['intervening']=int_dna
    df['target_site']=ts_dna
    df['PAM_start']=pam_start
    df['PAM_end']=pam_end
    df['proto_start']=proto_start
    df['proto_end']=proto_end
    df['intervene_start']=intervene_start
    df['intervene_end']=intervene_end
    df['ts_start']=ts_start
    df['ts_end']=ts_end
    df['insertion_length']=insertion_length
    df['aln_score']=aln_score
    df['gRNA_strand']=grna_strand
    
    
    return df, alns


def characterize_insertion_from_spacer(tsv,PAM, guideRNA, genome_sequence):
    import TagmentationAnalysis.postprocessHelpers as helper
    import pandas as pd
    import statistics

    #from spacer find insertions that occur at the 'correct' distance: 
    distance = 71
    window = 10
    
    grna = helper.map_guide_to_genome(guideRNA,genome_sequence)
    if grna[0]['strand'] == '+':
        predicted_pos = grna[0]['end']+3+distance
    else:
        predicted_pos = grna[0]['start']-3-distance

    df = pd.DataFrame(columns=["PAM","protospacer","gRNA_strand","intervening","insertion_length","target_site",
                               "PAM_start","PAM_end","proto_start","proto_end","aln_score",
                               "intervene_start","intervene_end","ts_start","ts_end"])

    ontarget_insertions = helper.between(tsv,predicted_pos-window,predicted_pos+window)
    
    if len(ontarget_insertions['ins0']) == 0:
        return df
    best_ontarget = statistics.mode(ontarget_insertions['ins0'])
    

    df['proto_start']= [ grna[0]['start'] ]
    df['proto_end']= [ grna[0]['end'] ]
    df['protospacer']= [ grna[0]['sequence'] ]
    if grna[0]['strand'] == '+':
        df['PAM_start'] = [ grna[0]['end'] ]
        df['PAM_end']= [ grna[0]['end']+3 ]
        df['PAM']= [ genome_sequence[ grna[0]['end'] : grna[0]['end']+3 ] ]
        df['intervene_start']= [ grna[0]['end']+3 ]
        df['intervene_end']= [ best_ontarget ]
        df['intervening']= [ genome_sequence[ grna[0]['end']+3 :  best_ontarget  ] ]
        df['target_site']= [ genome_sequence[best_ontarget-window:best_ontarget+window] ]
        df['insertion_length']= [ len(genome_sequence[ grna[0]['end']+3 :  best_ontarget  ]) ]

    else:
        df['PAM_start'] = [ grna[0]['start']-3 ]
        df['PAM_end']= [ grna[0]['start'] ]
        df['PAM']= [ hp.reverse_complement( genome_sequence[grna[0]['start']-3:grna[0]['start']] ) ] #always list sequences 5'->3'
        df['intervene_start']= [ best_ontarget ]
        df['intervene_end']= [ grna[0]['start']-3 ]
        df['intervening']= [ hp.reverse_complement( genome_sequence[best_ontarget:grna[0]['start']-3] ) ]
        df['target_site']= [ hp.reverse_complement( genome_sequence[best_ontarget-window:best_ontarget+window] ) ]
        df['insertion_length']=[ len(hp.reverse_complement( genome_sequence[best_ontarget:grna[0]['start']-3] )) ]

    df['ts_start']= [ best_ontarget-window ]
    df['ts_end']= [ best_ontarget+window ]
    df['gRNA_strand']= [ grna[0]['strand'] ]
    return df

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
    import TagmentationAnalysis.postprocessHelpers as hp
    #for window, compute at-richness and store in a list
    chunks = sliding_window( seq, window )
    at = []
    for ii in range(0,len(chunks)):
        at.append( hp.at_richness( chunks[ ii ] ))
    return at