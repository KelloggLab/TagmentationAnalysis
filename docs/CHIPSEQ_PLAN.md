# Can `apps/chipseq/chipseq_pipeline.sh` be rewritten in terms of `core/`?

> **Carried out on 2026-11-05.** Items 1-3 below are implemented as
> `apps/chipseq/plot_enrichment.py` and `tests/test_chipseq_plots.py`; items
> 4 and 5 were decided against, as proposed. Two deviations from this plan,
> both noted in place: the replicate arguments are `--rep BEDGRAPH LABEL
> PREFIX` (three `argparse` values) rather than the colon-joined form
> sketched below, since a path may contain a colon; and the module writes one
> file pair per contig rather than one figure of panels, because a 120 kb
> plasmid panel beside a 4.6 Mb chromosome is a sliver. The axis unit also
> switches to kb below 1 Mb, which the plan did not mention.

Mostly no — and the honest answer matters more than a tidy one.

## What the pipeline actually is

478 lines of bash that call six external programs in order:

| Step | Work | Done by |
|---|---|---|
| 0 | adapter/quality trimming | `trim_galore --paired` |
| 1 | reference index | `bwa index`, `samtools faidx` |
| 2 | alignment | `bwa mem` |
| 3 | dedup + MAPQ/proper-pair filter | `samtools fixmate -m / sort / markdup / view` |
| 4 | peak calling | `macs3 callpeak -f BAMPE` |
| 5 | fold-enrichment track | `macs3 bdgcmp -m FE` |
| 6 | consensus peaks | `bedtools multiinter / intersect / merge` |
| 7 | genome-wide figure | inline `python3` heredoc, lines 431–469 |

Steps 0–6 contain essentially no logic of our own. They are argument
construction and process sequencing for tools that exist only as
command-line programs. There is nothing in `core/` that can replace them,
and porting them to Python would add a `subprocess` layer around the same
commands while deleting nothing. **Recommendation: leave steps 0–6 in bash.**

`core/` also cannot help in the way it first appears to. It is CAST-specific
almost everywhere: `guides.py` maps spacers and PAMs, `insertions.py` works on
one-insertion-per-read tables, `predict_insertion.py` encodes the 71 bp
donor geometry, `plots.py` draws insertion profiles, `bbmap_workflow.py`
extracts Tn5 insertion sites. ChIP-seq has no spacer, no insertion and no
transposon end. Claiming reuse here would mean bending functions to fit.

## What genuinely can move — step 7, and only step 7

The one piece of real Python in the pipeline is the plotting heredoc, and it
should become a module. Three reasons, in order of importance:

### 1. It has a real bug: the figure ignores the `chrom` column

Both the bedGraph and the consensus BED are read with a `chrom` column
(lines 442, 449) that is never used. Every row is drawn against `start`
alone on a single axis spanning `0..GENOME_SIZE`, where `GENOME_SIZE` is the
**sum of all contigs** (`awk '{sum+=$2}'` over the `.fai`, line ~216).

With a single-contig genome this is correct. With a chromosome plus a
plasmid — routine for the strains this assay uses — positions from both
contigs are plotted from zero on top of each other, and the x-axis is as
long as the two combined. The peak shading has the same flaw. The figure
would look plausible and be wrong.

### 2. It runs `eval()` on an environment variable

```python
reps = eval("[" + os.environ["PLOT_REPS_PY"] + "]")
```

The shell builds a Python tuple literal by string concatenation (line 428)
and the Python side `eval`s it. Any output path containing a quote or
backslash breaks it or injects code. Passing the paths as `argv` removes the
construct entirely.

### 3. It cannot be tested where it is

A heredoc inside a 478-line script can only be exercised by running the whole
pipeline, which needs bwa, macs3, bedtools and real FASTQ. `apps/chipseq/` is
the only app with **no tests at all** (`tests/README.md` lists it as such).
As a module with a signature, it takes a bedGraph and a BED and writes a PNG,
and that is testable in a second with three lines of synthetic input.

## Proposed changes

1. **New `apps/chipseq/plot_enrichment.py`** holding the step-7 logic as a
   function plus a `main()` driven by `argv`:

   ```python
   plot_fold_enrichment(bdg_path, out_prefix, *, label, color,
                        contig_lengths, peaks_bed=None)
   ```

   It plots **per contig**, one panel per contig (or one file per contig),
   each axis spanning that contig's own length, read from the `.fai`. This
   fixes the bug above rather than carrying it across.

   It lives in `apps/chipseq/`, not `core/plots.py`: a fold-enrichment track
   over a bacterial genome is this assay's figure, and `core/plots.py` is
   insertion-profile plotting. Putting it in `core/` would be filing it where
   nothing else will ever call it. If a second app later needs a genome-wide
   track, that is the moment to promote it.

2. **Step 7 in the script becomes one call**, with the `.fai` path and the
   replicate paths as arguments:

   ```bash
   python3 -I "$HERE/plot_enrichment.py" \
       --fai "${REFERENCE}.fai" \
       --peaks "$OUTDIR/consensus/consensus_peaks.bed" \
       --rep "$OUTDIR/macs3_rep1/rep1_FE.bdg:Replicate 1:$OUTDIR/plots/rep1_genomewide_enrichment" ...
   ```

   `PLOT_REPS_PY`, the `eval`, the `GENOME_SIZE` export and the four other
   environment variables all go away.

3. **`tests/test_chipseq_plots.py`** — the first test this app has had.
   Writes a tiny `.fai` (two contigs), a bedGraph and a peaks BED; asserts a
   non-empty PNG and PDF per contig, that an out-of-range coordinate is not
   silently plotted on the wrong panel, and that a path containing a quote
   is handled. Needs only pandas and matplotlib, so it will not skip.

4. **Leave `GENOME_SIZE` as it is in the script.** MACS3's `-g` wants exactly
   the summed effective size, so the `awk` over the `.fai` is right there.
   Only the *plot* needed per-contig lengths. Adding a `core.sequence`
   genome-size helper would be make-work; `sequence.load_genome` reads the
   whole FASTA into memory to get what one line of `awk` already has.

5. **Not proposed:** `insertions.which` / `between` over peak tables.
   `bedtools intersect` already does that job correctly, including strand and
   sort order, and is already a dependency.

## What this does not do

- It does not reduce the pipeline's dependency on bwa/macs3/bedtools, and
  should not try to.
- It does not give `apps/chipseq/` end-to-end coverage. A smoke test on
  synthetic FASTQ through the full script is still missing, and is the next
  thing worth doing after this. It would skip without the tools installed,
  the way `test_insertion_pipeline.py` does.
- It changes the published figure for any multi-contig reference, by
  correcting it. Any existing figure from a multi-contig run should be
  regenerated.
