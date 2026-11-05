"""Figures.

Kept apart from the analysis so that computing a number and drawing it are
separate decisions. Each function returns its figure where it can; note that
`plot_insertion_profile` also calls `plt.show()`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .guides import map_guide_to_genome


def plot_insertion_profile(tsv_file,gRNA,genome_sequence,fig_filename,bins=100):
    df = pd.read_csv(tsv_file, sep="\t")
    df = df[df["mapq"] >= 30]
    positions = df["ins0"]

    gmap=map_guide_to_genome(gRNA,genome_sequence)
    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 3))

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.weight': 'bold',          # global default
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})


    font = {'family': 'Arial', 'size': 24}

    ax.hist(
        positions,
        bins=bins)
    
    ax.set_xlabel("Genomic position (bp, 0-based)", fontdict=font)
    ax.set_ylabel("Insertion count", fontdict=font)
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
            color="red",
            linewidth=4
        ),
        annotation_clip=False
    )
    window_bp = 100
    pct_ontarget = percent_within_distance(positions, gmap[0]['start'], window_bp)
    ndx = gmap[0]['start']
    #ax.set_title(f"Insertion site distribution: {pct_ontarget:.1f}% at site: {ndx:.1f} with window: {window_bp:.1f}")
    
    fig.tight_layout()
    fig.savefig(fig_filename)
    plt.show()


def plot_insertion_histogram(
    forward_positions, reverse_positions,
    region_start=0, region_end=None,
    bins=100, title="Insertion Sites",
    highlight_regions=None,
    filename=None
):
    if region_end is None:
        region_end = max(forward_positions + reverse_positions)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        forward_positions, bins=bins, range=(region_start, region_end),
        alpha=0.5, label='Top Strand', color='black'
    )
    ax.hist(
        reverse_positions, bins=bins, range=(region_start, region_end),
        alpha=0.5, label='Bottom Strand', color='grey'
    )

    if highlight_regions:
        ymin, ymax = ax.get_ylim()
        y = ymin - 0.01 * (ymax - ymin)
        for region in highlight_regions:
            if len(region) == 3:
                x_start, x_end, color = region
            else:
                x_start, x_end = region
                color = 'red'
            ax.hlines(y=y, xmin=x_start, xmax=x_end, colors=color, linewidth=5)

    ax.set_xlabel("Genomic Position")
    ax.set_ylabel("Insertion Count")
    ax.set_title(title)
    ax.set_xlim(region_start, region_end)
    ax.legend()
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300)

    return fig, ax


def plot_integration_profile(data, title="Integration Distance From PAM", xlabel="Distance From PAM (bp)", ylabel="Count", 
                   xlabel_fontsize=12, ylabel_fontsize=12, title_fontsize=14, 
                   hist_color='grey', edge_color='black', linewidth=5, dpi=300, xlim=(20,100), filename=None):
    
    plt.clf()
    plt.hist(data, edgecolor=edge_color, color=hist_color, linewidth=linewidth)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    
    if xlim:
        plt.xlim(xlim)
    
    plt.gcf().set_dpi(dpi)
    
    plt.tick_params(axis='both', labelsize=10)

    if filename:
        plt.savefig(filename, dpi=dpi)
        plt.close()
    else:
        plt.show()
    return plt
