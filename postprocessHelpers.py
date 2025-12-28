
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Bio.Align import PairwiseAligner


COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]



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
        rc = reverse_complement_dna(seq.replace("U", "T"))
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
            spacer_guide = reverse_complement_dna(spacer_plus)
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
    pct_ontarget = percent_within_distance(positions, gmap[0]['start'], window_bp)
    ax.set_title(f"Insertion site distribution: {pct_ontarget:.1f}%")
    
    fig.tight_layout()
    #fig.savefig("figures/"+Path(tsv).stem+"insertion_histogram.pdf")
    fig.savefig(fig_filename)
    plt.show()

