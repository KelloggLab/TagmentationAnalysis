"""Genome-wide fold-enrichment figures for the ChIP-seq pipeline.

One figure per replicate per contig, drawn from a MACS3 `bdgcmp -m FE`
bedGraph, optionally with consensus peaks shaded behind the track.

This used to be an inline `python3` heredoc in `chipseq_pipeline.sh`. It was
moved out for three reasons: the heredoc ignored the bedGraph's `chrom`
column and plotted every contig from zero on a single axis spanning the
summed length of all of them, it built its replicate list by `eval`-ing a
string concatenated by the shell, and nothing inside a heredoc can be tested.

Contig lengths come from the reference's `.fai`, so each axis spans the
contig it is showing and nothing else.

usage:
    python3 -I plot_enrichment.py --fai REF.fasta.fai \\
        [--peaks consensus_peaks.bed] \\
        --rep rep1_FE.bdg "Replicate 1" out/plots/rep1_genomewide_enrichment \\
        [--rep ...]
"""
from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import pandas as pd  # noqa: E402

#: Line colour per replicate, cycled. Kept from the original heredoc so
#: figures from before the move still match.
PALETTE = ["#2166ac", "#b2182b", "#1a9850"]

BDG_COLUMNS = ["chrom", "start", "end", "FE"]
PEAK_COLUMNS = ["chrom", "start", "end"]


def read_fai(path):
    """Return `{contig: length}` from a samtools `.fai`, in file order."""
    lengths = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.split("\t")
            lengths[fields[0]] = int(fields[1])
    if not lengths:
        raise ValueError(f"no contigs in {path}")
    return lengths


def read_bedgraph(path):
    """Read a 4-column bedGraph. Returns an empty frame for an empty file."""
    try:
        return pd.read_csv(path, sep="\t", header=None, names=BDG_COLUMNS,
                           dtype={"chrom": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=BDG_COLUMNS)


def read_peaks(path):
    """Read the first three columns of a BED. Empty frame for an empty file."""
    if path is None:
        return None
    try:
        return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                           names=PEAK_COLUMNS, dtype={"chrom": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=PEAK_COLUMNS)


def _position_formatter(contig_length):
    """Mb for anything chromosome-sized, kb for a plasmid.

    The original heredoc always labelled the axis in Mb, which reads as a row
    of zeroes on a contig a few hundred kb long.
    """
    if contig_length >= 1_000_000:
        return "Mb", ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}")
    return "kb", ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}")


def plot_contig(ax, track, contig_length, *, color, peaks=None):
    """Draw one contig's fold-enrichment track onto `ax`."""
    if peaks is not None:
        for _, p in peaks.iterrows():
            ax.axvspan(p["start"], p["end"], color="gray", alpha=0.15,
                       linewidth=0, zorder=1)

    ax.fill_between(track["start"], track["FE"], color=color, linewidth=0.3,
                    alpha=0.85, zorder=2)

    unit, formatter = _position_formatter(contig_length)
    ax.set_xlim(0, contig_length)
    ax.set_ylabel("Fold enrichment", fontsize=11)
    ax.set_xlabel(f"Genomic position ({unit})", fontsize=11)
    ax.xaxis.set_major_formatter(formatter)
    ax.spines[["top", "right"]].set_visible(False)


def plot_fold_enrichment(bdg_path, out_prefix, *, label, color, contig_lengths,
                         peaks=None):
    """Write a PNG and a PDF of `bdg_path`'s track, one pair per contig.

    With a single contig in `contig_lengths` the files are `out_prefix.png`
    and `.pdf`, as the heredoc wrote them. With more than one they are
    `out_prefix.<contig>.png` / `.pdf`, because a chromosome and a plasmid
    cannot share an axis.

    Returns the list of paths written. A contig with no rows in the bedGraph
    is skipped rather than drawn empty.
    """
    track = read_bedgraph(bdg_path)

    unknown = sorted(set(track["chrom"]) - set(contig_lengths))
    if unknown:
        print(f"    warning: {bdg_path} has contigs absent from the .fai, "
              f"not plotted: {', '.join(unknown)}", file=sys.stderr)

    multi = len(contig_lengths) > 1
    written = []

    for contig, length in contig_lengths.items():
        rows = track[track["chrom"] == contig]
        if rows.empty:
            print(f"    no coverage on {contig}, no plot written")
            continue

        contig_peaks = None
        if peaks is not None:
            contig_peaks = peaks[peaks["chrom"] == contig]

        fig, ax = plt.subplots(figsize=(14, 3))
        plot_contig(ax, rows, length, color=color, peaks=contig_peaks)
        ax.set_title(f"{label} — {contig}" if multi else label,
                     fontsize=12, loc="left", fontweight="bold")
        fig.tight_layout()

        prefix = f"{out_prefix}.{contig}" if multi else out_prefix
        for ext in ("png", "pdf"):
            path = f"{prefix}.{ext}"
            fig.savefig(path, dpi=300 if ext == "png" else None,
                        bbox_inches="tight")
            written.append(path)
        plt.close(fig)
        print(f"    Saved: {prefix}.png / .pdf")

    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fai", required=True,
                   help="samtools .fai for the reference, for contig lengths")
    p.add_argument("--peaks", default=None,
                   help="consensus peaks BED to shade behind the track")
    p.add_argument("--rep", required=True, action="append", nargs=3,
                   metavar=("BEDGRAPH", "LABEL", "OUT_PREFIX"),
                   help="one replicate; repeat for more")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    contig_lengths = read_fai(args.fai)
    peaks = read_peaks(args.peaks)

    for idx, (bdg_path, label, out_prefix) in enumerate(args.rep):
        plot_fold_enrichment(bdg_path, out_prefix, label=label,
                             color=PALETTE[idx % len(PALETTE)],
                             contig_lengths=contig_lengths, peaks=peaks)


if __name__ == "__main__":
    main()
