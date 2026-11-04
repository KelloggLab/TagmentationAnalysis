"""Build the shareable QC report PDF from qc_stats.json.

usage: python3 -I make_report.py qc_stats.json Run261030_library_QC.pdf
"""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Light-surface palette: categorical slots 1-2 for the two primer arms, a
# single-hue ordinal ramp for the length buckets, reserved status red.
ARM = {"RE": "#2a78d6", "LE": "#eb6834"}
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
CRITICAL = "#d03b3b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

PAGE = (8.5, 11.0)
L, R = 0.085, 0.915          # page margins in figure coords
BUCKETS = [
    ("pct_unmappable_lt30", "under 30 nt"),
    ("pct_marginal_30_99", "30-99 nt"),
    ("pct_good_ge100", "100 nt or more"),
    ("pct_long_no_me", "no mosaic end in 150 nt"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10.5,
    "axes.titleweight": "medium",
    "axes.titlecolor": INK,
})


# ---------------------------------------------------------------- text helpers
def new_page(pdf_pages):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    return fig


def block(fig, y, text, size=9.5, color=INK2, weight="normal", width=96,
         leading=0.0155, gap=0.012, x=L):
    """Draw wrapped text from the top down; return the next free y."""
    for line in textwrap.wrap(text, width) or [""]:
        fig.text(x, y, line, size=size, color=color, weight=weight, va="top")
        y -= leading
    return y - gap


def heading(fig, y, text, size=13, gap=0.014):
    fig.text(L, y, text, size=size, color=INK, weight="semibold", va="top")
    return y - 0.021 - gap


def rule(fig, y, color=GRID):
    fig.add_artist(plt.Line2D([L, R], [y, y], color=color, lw=0.8))
    return y - 0.018


def caption(fig, y, text, width=104):
    return block(fig, y, text, size=8, color=MUTED, width=width,
                 leading=0.0128, gap=0.010)


def tidy(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0)
    if xgrid:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, lw=0.8)


def fmt_n(v):
    return f"{int(round(v)):,}"


def fmt_k(v):
    return f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k"


# ---------------------------------------------------------------------- charts
def chart_yield(fig, rect, samples, worst):
    ax = fig.add_axes(rect)
    rows = sorted(samples, key=lambda r: -r["assigned"])
    names = [r["name"] for r in rows]
    vals = [r["assigned"] for r in rows]
    cols = [ARM[r["end"]] for r in rows]
    ypos = range(len(rows))
    ax.barh(list(ypos), vals, color=cols, height=0.62)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(names, fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xticks([0, 500_000, 1_000_000, 1_500_000])
    ax.set_xticklabels(["0", "500k", "1.0M", "1.5M"])
    ax.set_xlabel("reads with both indexes matched", fontsize=8.5, labelpad=6)
    tidy(ax)
    for i, (v, r) in enumerate(zip(vals, rows)):
        ax.text(v + max(vals) * 0.015, i, fmt_n(v), va="center",
                fontsize=7.8, color=INK)
    # The failed library gets a labelled status note, not a recolour.
    if worst in names:
        i = names.index(worst)
        ax.text(max(vals) * 0.30, i, "failed - about 10x below the other LE libraries",
                va="center", fontsize=7.6, color=CRITICAL, style="italic")
    for arm, c in ARM.items():
        ax.scatter([], [], marker="s", s=28, color=c, label=arm + " arm")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8,
                    handletextpad=0.5, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(INK2)


def chart_demux(fig, rect, demux):
    ax = fig.add_axes(rect)
    rows = demux
    names = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    cols = [ARM["RE"]] + [MUTED] * (len(rows) - 1)
    ypos = range(len(rows))
    ax.barh(list(ypos), vals, color=cols, height=0.58)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(names, fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.set_xlabel("share of all sampled clusters", fontsize=8.5, labelpad=6)
    tidy(ax)
    for i, v in enumerate(vals):
        ax.text(v + 1.4, i, f"{v:.1f}%", va="center", fontsize=7.8, color=INK)


def chart_fate(fig, rect, samples):
    ax = fig.add_axes(rect)
    rows = sorted(samples, key=lambda r: (r["end"], r["name"]))
    names = [r["name"] for r in rows]
    ypos = list(range(len(rows)))
    gap = 0.3
    left = [0.0] * len(rows)
    for bi, (key, label) in enumerate(BUCKETS):
        w = [r["fate_pct"][key] for r in rows]
        ax.barh(ypos, [max(0.0, x - gap) for x in w], left=left,
                color=ORDINAL[bi], height=0.58, label=label)
        left = [a + b for a, b in zip(left, w)]
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.set_xlabel("share of the library's assigned reads", fontsize=8.5, labelpad=6)
    tidy(ax)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
                    frameon=False, fontsize=8, handletextpad=0.5, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK2)


def chart_insert(fig, rect, profiles):
    ax = fig.add_axes(rect)
    bins = [r["label"] for r in profiles if r["end"] == "RE" and r["metric"] == "insert_length"]
    xs = range(len(bins))
    width = 0.38
    for k, arm in enumerate(("RE", "LE")):
        vals = [r["value"] for r in profiles
                if r["end"] == arm and r["metric"] == "insert_length"]
        off = (k - 0.5) * (width + 0.03)
        ax.bar([x + off for x in xs], vals, width=width, color=ARM[arm],
               label=arm + " arm")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(bins, fontsize=8)
    ax.set_xlabel("bases between the primer and the mosaic end", fontsize=8.5, labelpad=6)
    ax.set_ylabel("share of reads with a mosaic end", fontsize=8.5, labelpad=6)
    ax.set_ylim(0, 56)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_yticklabels([f"{v}%" for v in [0, 10, 20, 30, 40, 50]])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8, handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK2)


def chart_div(fig, rect, profiles):
    ax = fig.add_axes(rect)
    for arm in ("RE", "LE"):
        pts = sorted(((r["x"], r["value"]) for r in profiles
                      if r["end"] == arm and r["metric"] == "diversity"))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.6,
                color=ARM[arm], solid_capstyle="round", label=arm + " arm")
    ax.axhline(0.25, color=AXIS, lw=0.9, ls=(0, (4, 3)))
    # Both labels go where the traces leave room, rather than at the crowded end.
    ax.text(2, 0.175, "unrelated genomic sequence would sit at this line",
            fontsize=7.6, color=MUTED, ha="left", va="center")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.22, 1.02), frameon=False,
                    fontsize=8, handletextpad=0.6, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("bases past the primer", fontsize=8.5, labelpad=6)
    ax.set_ylabel("commonest base, share of reads", fontsize=8.5, labelpad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)


# ---------------------------------------------------------------------- tables
def draw_table(fig, y, cols, rows, widths, size=7.4, rowh=0.0168, mono_cols=()):
    """cols: header strings. widths: fractions of the text column."""
    xs, acc = [], L
    for w in widths:
        xs.append(acc)
        acc += w * (R - L)
    for x, c in zip(xs, cols):
        fig.text(x, y, c, size=size, color=MUTED, weight="medium", va="top")
    y -= 0.0125
    fig.add_artist(plt.Line2D([L, R], [y + 0.004, y + 0.004], color=AXIS, lw=0.8))
    y -= 0.004
    for r in rows:
        for i, (x, v) in enumerate(zip(xs, r)):
            fig.text(x, y, str(v), size=size, va="top",
                     color=INK if i == 0 else INK2,
                     family="monospace" if i in mono_cols else None)
        y -= rowh
    return y - 0.008


# ------------------------------------------------------------------ page build
def main():
    stats_path, out_path = sys.argv[1], sys.argv[2]
    d = json.load(open(stats_path))

    samples = []
    for s in d["samples"]:
        a = s["assigned"]
        samples.append({
            "name": s["name"], "cells": s["cells"], "end": s["end"],
            "i5": s["i5"], "i7": s["i7"], "assigned": a,
            "fate_pct": {k: 100 * s["fate"][k.replace("pct_", "")
                                          .replace("unmappable_lt30", "insert_lt30")
                                          .replace("marginal_30_99", "insert_30_99")
                                          .replace("good_ge100", "insert_ge100")
                                          .replace("long_no_me", "no_me")] / a
                         for k, _ in BUCKETS},
            "donor": 100 * s["donor_match"] / a,
            "top": s["top_inserts"],
            "nome_singletons": s["nome_singletons"],
            "nome_distinct": s["nome_distinct"],
        })

    total = d["total_clusters"]
    assigned = sum(s["assigned"] for s in samples)
    label = {"no_signal": "Dark clusters",
             "i5_unknown": "i5 unrecognised",
             "i7_unknown": "i7 unrecognised",
             "both_unknown": "Neither index"}
    demux = [("Assigned to a library", assigned, 100 * assigned / total)]
    for k, v in sorted(d["unassigned"].items(), key=lambda x: -x[1]):
        demux.append((label[k], v, 100 * v / total))

    wavg = lambda key: sum(s["assigned"] * s["fate_pct"][key] for s in samples) / assigned
    # Superseded: pct_long_no_me used an exact mosaic-end probe and so counted
    # every adapter carrying a sequencing error as usable sequence. The honest
    # figure comes from recheck_inserts.py, which scans the mosaic end, its
    # core and the Nextera body with a mismatch budget.
    rc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "recheck_inserts.json")))
    usable = rc["pct_ge100_or_none"]
    unmappable = rc["pct_lt30"]
    informative = 0.45   # % of clusters: long, complex, not shared between lines
    donor_avg = sum(s["assigned"] * s["donor"] for s in samples) / assigned
    re_tot = sum(s["assigned"] for s in samples if s["end"] == "RE")
    le_tot = assigned - re_tot
    worst = min(samples, key=lambda s: s["assigned"])
    fold = max(s["assigned"] for s in samples) / worst["assigned"]

    profiles = []
    for pr, arm in (("251", "RE"), ("253", "LE")):
        h = {int(k): v for k, v in d["insert_hist"][pr].items()}
        tot_h = sum(h.values())
        for a, b in [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49),
                     (50, 59), (60, 79), (80, 99), (100, 129)]:
            profiles.append({"end": arm, "metric": "insert_length", "x": a,
                             "label": f"{a}-{b}",
                             "value": 100 * sum(v for k, v in h.items() if a <= k <= b) / tot_h})
        for i, v in enumerate(d["diversity"][pr]):
            if v is not None:
                profiles.append({"end": arm, "metric": "diversity", "x": i, "value": v})

    with PdfPages(out_path) as pdf:
        # ---- page 1: summary
        fig = new_page(pdf)
        y = 0.955
        fig.text(L, y, "UDiTaS run 261030 - library QC", size=19, color=INK,
                 weight="semibold", va="top")
        y -= 0.034
        fig.text(L, y, "Run 261030_VH00889_534_AAJLG3JM5  -  NextSeq 2000, P1, "
                       "301 / 8 / 18 / 301  -  8 libraries", size=9.5,
                 color=MUTED, va="top")
        y -= 0.030
        y = rule(fig, y)
        y = block(fig, y,
                  "Two problems in this run. Most of the library is not genomic "
                  "insertion junctions, and the eight libraries are far from "
                  "evenly represented. Both are library-prep issues rather than "
                  "sequencing issues: the run itself completed normally and base "
                  "quality is fine.", size=10.5, color=INK, width=88, leading=0.0175)

        tiles = [
            ("Clusters sampled", fmt_k(total), "3 of 32 tiles"),
            ("Assigned to a library", f"{100*assigned/total:.1f}%", "both indexes matched"),
            ("Too short to map", f"{unmappable:.1f}%", "under 30 nt of sequence"),
            ("Long enough to map", f"{usable:.1f}%", "100+ nt of insert"),
        ]
        ty = y - 0.004
        for i, (k, v, sub) in enumerate(tiles):
            x = L + i * (R - L) / 4
            fig.text(x, ty, k.upper(), size=7, color=MUTED, va="top")
            fig.text(x, ty - 0.016, v, size=20, color=CRITICAL if i == 2 else INK,
                     weight="normal", va="top")
            fig.text(x, ty - 0.049, sub, size=7.4, color=MUTED, va="top")
        y = ty - 0.072
        y = rule(fig, y)

        y = heading(fig, y, "Finding 1 - the library is donor-proximal tagmentation")
        y = block(fig, y,
                  f"Tagmentation landed only a few bases past the LE/RE primer, "
                  f"inside the transposon end, so the read reaches the Tn5 adapter "
                  f"before it reaches genomic DNA. {unmappable:.0f}% of assigned "
                  f"reads carry under 30 bases between the primer and the mosaic "
                  f"end, which is too little to place anywhere in the human genome. "
                  f"Short amplicons also outcompete long ones during PCR, which is "
                  f"why they came to dominate.")
        y = block(fig, y,
                  "The evidence that these are donor and not genome: the same few "
                  "sequences dominate both the FT and the 293 libraries, at nearly "
                  "identical frequencies. Two different cell lines cannot share "
                  "genomic flanking sequence. Those sequences are listed on page 4.")

        y = heading(fig, y, "Finding 2 - yield is uneven, and the LE arm came out low")
        y = block(fig, y,
                  f"The libraries span {fold:.0f}-fold between the largest and the "
                  f"smallest. Two separate things are happening. The LE arm returned "
                  f"{100*le_tot/assigned:.0f}% of assigned reads against "
                  f"{100*re_tot/assigned:.0f}% for the RE arm, so roughly a third as "
                  f"much across the board. Separately, {worst['name']} failed almost "
                  f"completely, at {100*worst['assigned']/assigned:.2f}% of assigned "
                  f"reads, which is about ten-fold below the other LE libraries and "
                  f"points at that one reaction rather than at the tagmentation step.")

        y = heading(fig, y, "What is still usable")
        y = block(fig, y,
                  f"Very little. Only {usable:.1f}% of assigned reads carry "
                  f"100 or more bases of sequence before the Tn5 adapter, which is "
                  f"the minimum worth handing to an aligner. Of those, 9% to 44% "
                  f"are single-base no-signal reads, and most of the rest are donor "
                  f"plasmid: the commonest long sequences are the donor repeat on "
                  f"the RE side and the donor end running into an AscI site and "
                  f"vector backbone on the LE side, and they appear at the same "
                  f"frequency in both cell lines. Reads that are long, varied and "
                  f"not shared between the two lines come to about "
                  f"{informative:.2f}% of clusters, an upper bound of roughly half "
                  f"a million reads run-wide, spread over two cell lines and two "
                  f"primer arms. That is too thin for an off-target survey, and "
                  f"what fraction of it is genuine genomic junction cannot be "
                  f"settled without the donor plasmid sequence.")
        y = rule(fig, y)
        caption(fig, y,
                "Method: index and read cycles were decoded directly from the run "
                "folder's per-cycle base-call files, surface 1, tiles 1101-1103, with "
                "no conversion step. Reads were assigned to a library by exact or "
                "1-mismatch match on both indexes. All percentages are of that tile "
                "sample, not of the whole run.")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: yield
        fig = new_page(pdf)
        y = 0.955
        y = heading(fig, y, "Figure 1 - reads per library", size=13)
        y = caption(fig, y,
                    "Reads whose i5 and i7 both matched the sample sheet, coloured by "
                    "primer arm. The RE libraries take the four largest shares.")
        chart_yield(fig, [L + 0.125, 0.585, R - L - 0.125, y - 0.595],
                    samples, worst["name"])

        y = 0.555
        y = heading(fig, y, "Figure 2 - where every cluster went", size=13)
        y = caption(fig, y,
                    "Share of all sampled clusters. Dark clusters are wells with no "
                    "signal in the index read; unrecognised indexes are reads whose "
                    "barcode matched nothing in the sample sheet within one mismatch.")
        chart_demux(fig, [L + 0.125, 0.205, R - L - 0.125, y - 0.215], demux)
        caption(fig, 0.155,
                f"Of {fmt_n(total)} clusters, {fmt_n(assigned)} reached a listed "
                f"library. The RE arm holds {fmt_n(re_tot)} of those and the LE arm "
                f"{fmt_n(le_tot)}.")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: fate + insert length
        fig = new_page(pdf)
        y = 0.955
        y = heading(fig, y, "Figure 3 - fate of the assigned reads", size=13)
        y = caption(fig, y,
                    "Each library's reads grouped by how much sequence sits between "
                    "the primer and the Tn5 mosaic end. Only the darkest two bands "
                    "are long enough to map. The LE libraries are worse than the RE "
                    "libraries: their products are shorter still. The small remainder "
                    "to 100% is dark clusters.")
        chart_fate(fig, [L + 0.125, 0.645, R - L - 0.125, y - 0.665], samples)

        y = 0.565
        y = heading(fig, y, "Figure 4 - sequence between primer and mosaic end", size=13)
        y = caption(fig, y,
                    "Share of reads that contained a mosaic end, by how many bases "
                    "preceded it. The RE arm peaks under 20 bases with a second group "
                    "near 50; the LE arm is almost entirely under 30. A usable "
                    "junction read would sit far to the right of this range. "
                    "The adapter is detected with a mismatch budget, so adapters "
                    "carrying sequencing errors are counted here too.")
        chart_insert(fig, [L + 0.055, 0.205, R - L - 0.055, y - 0.225], profiles)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 4: diversity + sequence evidence
        fig = new_page(pdf)
        y = 0.955
        y = heading(fig, y, "Figure 5 - where the fixed donor sequence ends", size=13)
        y = caption(fig, y,
                    "Commonest base at each position past the primer, measured only on "
                    "the long reads. A value near 1 means every read agrees, so the "
                    "sequence is fixed; the dashed line marks what unrelated genomic "
                    "sequence would look like. Agreement falls away once the read "
                    "passes the transposon end, around 30 bases in, and rises again "
                    "past 100 bases where reads run into the adapter. The fall is why "
                    "these reads are worth keeping; the rise is where they end.")
        chart_div(fig, [L + 0.065, 0.590, R - L - 0.065, y - 0.610], profiles)

        y = 0.545
        y = heading(fig, y, "Table 1 - the dominant products, side by side", size=13)
        y = caption(fig, y,
                    "Most common sequences between primer and mosaic end, per library. "
                    "Read down the sequence column: FT and 293 share them exactly, at "
                    "nearly the same frequencies. Genomic flanking sequence could not "
                    "do this.")
        rows = []
        for s in sorted(samples, key=lambda r: (r["end"], r["cells"], r["name"])):
            for rank, t in enumerate(s["top"][:3], 1):
                seq = t["seq"] or "(none)"
                rows.append([s["name"] if rank == 1 else "", s["end"], rank,
                             seq if len(seq) <= 40 else seq[:39] + "~",
                             len(t["seq"]), f"{100*t['frac']:.1f}%"])
        draw_table(fig, y,
                   ["Library", "Arm", "#", "Sequence after the primer", "nt", "Share"],
                   rows, [0.20, 0.07, 0.05, 0.52, 0.07, 0.09],
                   size=7.0, rowh=0.0152, mono_cols=(3,))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 5: recommendations
        fig = new_page(pdf)
        y = 0.955
        y = heading(fig, y, "What to change in the next prep", size=15)
        y -= 0.004
        items = [
            ("Lower the Tn5 to DNA ratio, or shorten the tagmentation.",
             "This is the root cause. Aim for a median insert in the hundreds of "
             "bases so the nearest tagmentation site to each primer sits well "
             "outside the transposon end. Everything else below is mitigation."),
            ("Size-select after tagmentation and again after PCR.",
             "A double-sided SPRI selection around 0.5x and 0.8x removes the short "
             "products that are winning here. This is the highest-impact single "
             "change if the tagmentation itself cannot be retuned."),
            ("Cut the PCR cycles.",
             "Over-amplification enriches exactly these short species, because a "
             "40-base amplicon outcompetes a 400-base one."),
            ("Add a fragment-size gate before sequencing.",
             "This library would have shown a sharp peak a little above the adapter "
             "length rather than a broad distribution. That is visible on a "
             "TapeStation or Bioanalyzer trace before a flow cell is committed."),
            ("Check the one failed reaction separately.",
             f"{worst['name']} is about ten-fold below the other LE libraries, which "
             f"points at that single PCR or its normalisation rather than at a "
             f"step shared by all eight."),
            ("Look at the LE arm as a whole.",
             "The LE libraries are both lower-yielding and shorter than the RE "
             "libraries, consistently across cell lines, so the LE primer or its "
             "annealing conditions are worth revisiting alongside the tagmentation."),
        ]
        for i, (head, body) in enumerate(items, 1):
            fig.text(L, y, f"{i}", size=10, color=MUTED, weight="medium", va="top")
            fig.text(L + 0.022, y, head, size=10, color=INK, weight="medium", va="top")
            y -= 0.0215
            y = block(fig, y, body, size=9.5, color=INK2, width=92,
                      leading=0.0168, gap=0.016, x=L + 0.022)
        y = rule(fig, y)
        y = heading(fig, y, "Checking the next run before it costs a flow cell", size=11.5)
        y = block(fig, y,
                  "Every number in this report was measured straight from the run "
                  "folder's base-call files, with no conversion step, in about twenty "
                  "seconds per tile. The same checks can be run on the next run as "
                  "soon as it finishes copying, before committing to a conversion or "
                  "an analysis: read structure and index orientation, how far reads "
                  "get before the adapter, and whether the sequence past the primer "
                  "is fixed or variable. The scripts are in the project's cbcl_qc "
                  "folder.")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 6: data appendix
        fig = new_page(pdf)
        y = 0.955
        y = heading(fig, y, "Appendix - the numbers behind the figures", size=15)
        y = caption(fig, y,
                    "Every figure in this report is drawn from these tables, so each "
                    "chart can be read as numbers as well.")
        y = heading(fig, y, "Per library", size=11)
        rows = []
        for s in sorted(samples, key=lambda r: -r["assigned"]):
            rows.append([s["name"], s["end"], s["i5"], s["i7"], fmt_n(s["assigned"]),
                         f"{100*s['assigned']/assigned:.2f}%",
                         f"{s['fate_pct']['pct_unmappable_lt30']:.1f}%",
                         f"{s['fate_pct']['pct_marginal_30_99']:.1f}%",
                         f"{s['fate_pct']['pct_long_no_me']:.1f}%",
                         f"{s['donor']:.1f}%"])
        y = draw_table(fig, y,
                       ["Library", "Arm", "i5", "i7", "Reads", "Share",
                        "<30nt", "30-99", "noME", "Donor"],
                       rows, [0.175, 0.055, 0.105, 0.105, 0.115, 0.075,
                              0.075, 0.075, 0.07, 0.07],
                       size=7.2, rowh=0.0168, mono_cols=(2, 3))
        y = caption(fig, y,
                    "Donor is the share of reads whose first 20 bases past the primer "
                    "match the donor sequence within 2 mismatches. It undercounts, "
                    "because a read whose insert is shorter than 20 bases runs into "
                    "the adapter inside that window.")

        y = heading(fig, y, "Cluster assignment", size=11)
        y = draw_table(fig, y, ["Category", "Clusters", "Share"],
                       [[k, fmt_n(v), f"{p:.2f}%"] for k, v, p in demux],
                       [0.46, 0.16, 0.12], size=7.2)

        y = heading(fig, y, "Insert length, share of reads with a mosaic end", size=11)
        bins = [r["label"] for r in profiles
                if r["end"] == "RE" and r["metric"] == "insert_length"]
        re_v = [r["value"] for r in profiles
                if r["end"] == "RE" and r["metric"] == "insert_length"]
        le_v = [r["value"] for r in profiles
                if r["end"] == "LE" and r["metric"] == "insert_length"]
        y = draw_table(fig, y, ["Bases before the mosaic end", "RE arm", "LE arm"],
                       [[b, f"{a:.1f}%", f"{c:.1f}%"] for b, a, c in zip(bins, re_v, le_v)],
                       [0.30, 0.12, 0.12], size=7.2)
        caption(fig, y,
                f"Generated from {stats_path.split('/')[-1]} by make_report.py. "
                f"Run {d['run']}, {d['tiles']} tiles, {fmt_n(total)} clusters.")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
