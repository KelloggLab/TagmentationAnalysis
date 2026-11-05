"""Matching and scoring a sequence against a motif.

Two ways of asking the same question. `iupac_match` and the `degenerate_*`
family count mismatches against an IUPAC pattern, which is how PAMs are found.
The `*_pwm` functions build and score a position weight matrix, which gives a
graded answer instead of a count.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


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


def pam_matches_at(seq: str, i: int, pam: str) -> bool:
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


def iupac_match(seq_window: str, pattern: str) -> bool:
    """Return True if seq_window matches IUPAC pattern (same length)."""
    if len(seq_window) != len(pattern):
        return False
    seq_window = seq_window.upper().replace("U", "T")
    pattern = pattern.upper().replace("U", "T")
    for b, code in zip(seq_window, pattern):
        allowed = IUPAC_DNA.get(code)
        if allowed is None:
            raise ValueError(f"Unknown IUPAC code in PAM motif: {code}")
        if b not in allowed:
            return False
    return True


def degenerate_hamming(window: str, motif_iupac: str) -> int:
    """
    Returns number of mismatches under IUPAC degeneracy.
    0 = perfect match, higher = worse.
    """
    window = window.upper().replace("U", "T")
    motif_iupac = motif_iupac.upper()
    if len(window) != len(motif_iupac):
        raise ValueError("window and motif must have the same length")

    mismatches = 0
    for b, code in zip(window, motif_iupac):
        allowed = IUPAC_DNA.get(code)
        if allowed is None:
            raise ValueError(f"Unknown IUPAC code: {code}")
        if b not in allowed:
            mismatches += 1
    return mismatches


def degenerate_match_count(window: str, motif_iupac: str) -> int:
    """Returns number of matching positions (higher = better)."""
    return len(motif_iupac) - degenerate_hamming(window, motif_iupac)


def scan_degenerate_hamming(seq: str, motif_iupac: str):
    seq = seq.upper().replace("U", "T")
    m = len(motif_iupac)
    hits = []
    for i in range(len(seq) - m + 1):
        w = seq[i:i+m]
        d = degenerate_hamming(w, motif_iupac)
        hits.append((i, d, w))
    # sort by best (fewest mismatches)
    hits.sort(key=lambda t: t[1])
    return hits


def best_hit_fewest_mismatches_closest_center(seq: str, motif_iupac: str):
    """
    Returns the (start, mismatches, window) hit with:
      1) minimum mismatches
      2) closest window-center to sequence center
      3) (tie-break) smallest start index
    """
    seq = seq.upper().replace("U", "T")
    m = len(motif_iupac)

    seq_center = (len(seq) - 1) / 2.0  # center coordinate of the full sequence

    best = None
    best_key = None

    for start in range(len(seq) - m + 1):
        window = seq[start:start + m]
        mismatches = degenerate_hamming(window, motif_iupac)

        window_center = start + (m - 1) / 2.0
        dist_to_center = abs(window_center - seq_center)

        key = (mismatches, dist_to_center, start)  # lexicographic sort: smaller is better

        if best_key is None or key < best_key:
            best_key = key
            best = (start, mismatches, window)

    return best


def build_log_odds_pwm(counts_df, background=None, pseudocount=1.0, log_base=2):
    """
    Build a log-odds PWM from a counts matrix (positions x alphabet).

    Parameters
    ----------
    counts_df : pd.DataFrame
        Rows = positions (0..L-1), columns = letters (e.g., A,C,G,T), values = counts.
    background : dict or pd.Series or None
        Background probabilities for each letter. If None, uses uniform over columns.
        Example: {"A":0.25,"C":0.25,"G":0.25,"T":0.25}
    pseudocount : float, default 1.0
        Added to each count to avoid zeros (Laplace smoothing).
    log_base : int or float, default 2
        Base for the logarithm. log2 is standard for “bits”.

    Returns
    -------
    pwm_df : pd.DataFrame
        Log-odds PWM with same shape as counts_df:
        pwm_df.loc[i, base] = log_{log_base}( p_i(base) / bg(base) )
    prob_df : pd.DataFrame
        Smoothed position-specific probabilities used to compute the PWM.
    """
    alphabet = list(counts_df.columns)

    # Background
    if background is None:
        bg = pd.Series({a: 1.0 / len(alphabet) for a in alphabet}, dtype=float)
    else:
        bg = pd.Series(background, dtype=float).reindex(alphabet)
        if bg.isna().any():
            missing = bg[bg.isna()].index.tolist()
            raise ValueError(f"Background missing letters: {missing}")
        if (bg <= 0).any():
            raise ValueError("Background probabilities must be > 0")

    # Smooth counts -> probabilities
    smoothed = counts_df.astype(float) + float(pseudocount)
    prob_df = smoothed.div(smoothed.sum(axis=1), axis=0)

    # Log-odds
    denom = bg.values  # length K
    ratio = prob_df.values / denom  # (L x K)

    if log_base == 2:
        pwm = np.log2(ratio)
    else:
        pwm = np.log(ratio) / np.log(log_base)

    pwm_df = pd.DataFrame(pwm, index=counts_df.index, columns=alphabet)
    return pwm_df, prob_df


def score_sequence_pwm(seq, pwm_df, *, unknown_penalty=None):
    """
    Score a sequence with a log-odds PWM (sum across positions).

    Parameters
    ----------
    seq : str
        Sequence to score (length must equal #rows in pwm_df).
    pwm_df : pd.DataFrame
        Log-odds PWM (positions x alphabet).
    unknown_penalty : float or None, default None
        What to do if a character is not in pwm_df.columns:
        - None: raise ValueError
        - float: add this value for unknown characters

    Returns
    -------
    float
        Total log-odds score.
    """
    seq = seq.upper()
    L = pwm_df.shape[0]
    if len(seq) != L:
        raise ValueError(f"Sequence length {len(seq)} != PWM length {L}")

    col_index = {c: j for j, c in enumerate(pwm_df.columns)}
    vals = pwm_df.values

    total = 0.0
    for i, ch in enumerate(seq):
        j = col_index.get(ch)
        if j is None:
            if unknown_penalty is None:
                raise ValueError(f"Unknown base '{ch}' at position {i}")
            total += float(unknown_penalty)
        else:
            total += float(vals[i, j])

    return total


def pwm_max_possible_score(pwm_df):
    """
    Maximum possible score achievable by this PWM over any sequence of length L,
    i.e., sum of the per-position best letter log-odds.
    """
    return float(pwm_df.max(axis=1).sum())
