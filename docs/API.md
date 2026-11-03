# API reference

Functions in the `tagmentation` package. Home: [KelloggLab/TagmentationAnalysis](https://github.com/KelloggLab/TagmentationAnalysis)

```python
from tagmentation import postprocessHelpers as hp
from tagmentation import TsvMapHelperFunctions as tsvhp
from tagmentation import NovoCAST_helper as nc
from tagmentation import NGSanalysis_module as ngs
```

## Conventions

- **Coordinates are 0-based** and ranges are end-exclusive (Python slicing), unless noted.
- **Sequences** are plain strings in the DNA alphabet. Most functions upper-case their input and treat `U` as `T`.
- **Positions are reported on the + (top) strand** of the input sequence, even for − strand hits.
- **"Insertion TSV"** means the file written by `pipeline/map_junction_insertions.py` (or the filtered `.nobackbone.tsv`), with columns `read_id, ref, ins0, ins1, strand, mapq, cigar, aln_start0, aln_end0_excl`. Load it with `pd.read_csv(path, sep="\t")`.

---

## postprocessHelpers

Sequence utilities: genome loading, guide and PAM finding, IUPAC motif matching, A/T content, local alignment, position weight matrices.

### `SpacerHit`

Frozen dataclass returned by [`extract_spacers`](#extract_spacerssequence-pam-spacer_len20-spacer_side5prime-search_strandsboth-return_rnafalse).

| Field | Meaning |
|---|---|
| `strand` | `"+"` or `"-"`, the strand the PAM was found on |
| `pam_start`, `pam_end` | PAM position on the + strand (end-exclusive) |
| `spacer_start`, `spacer_end` | Spacer position on the + strand (end-exclusive) |
| `pam_dna` | PAM sequence, + strand |
| `spacer_top` | Spacer sequence as it reads on the **+ strand** |

For a − strand hit, `spacer_top` is *not* the guide sequence; apply `reverse_complement` to get the guide.

### `IUPAC_DNA`

Dict mapping IUPAC codes (`A C G T U R Y S W K M B D H V N`) to the bases they allow. Used by the degenerate-matching functions.

### `reverse_complement(seq)`

Reverse complement of `seq`. Preserves case; characters outside `ACGTacgt` pass through unchanged.

### `load_genome(filename)`

Reads the **first** record of a FASTA file and returns its sequence as one string. Use this for a single-contig reference such as an *E. coli* genome.

### `load_fasta_as_dict(fasta_path)`

Reads **all** records of a FASTA file into `{record_id: sequence}`, upper-cased. Use when the reference has more than one contig; this is what `extract_tsd_centered_motifs_5p` uses to match the TSV's `ref` column.

### `map_guide_to_genome(guide, genome)`

Finds every exact occurrence of a guide and its reverse complement.

Returns a **list of dicts**, forward-strand hits first:

| Key | Meaning |
|---|---|
| `start`, `end` | Position in the genome (end-exclusive) |
| `strand` | `"+"` or `"-"` |
| `sequence` | Matched genomic sequence, + strand |

Returns `[]` if the guide is absent. Several functions here use `hits[0]`, so check the list is non-empty, and that the guide is unique, before relying on them.

> **Note:** for a `-` hit, `start`/`end` are the span on the + strand and `sequence` is the + strand sequence there — not the guide itself. Apply `reverse_complement` to `sequence` to recover the guide.

### `extract_sequence_windows(positions, genome, window_bp)`

For each position, extracts `window_bp` on both sides, giving sequences of `2 * window_bp + 1` bp. Positions outside the genome are skipped; windows at the ends are clipped (so they come back shorter).

Returns a list of dicts with `position`, `start`, `end`, `sequence`.

### `at_richness(sequence, *, ignore_ambiguous=True)`

Fraction of A/T bases, from 0 to 1. `U` counts as `T`; whitespace and `-` are removed. With `ignore_ambiguous=True` (the default) the denominator counts only unambiguous `ACGT`; with `False`, every remaining character counts. Returns `0.0` if no valid bases are found, and raises `ValueError` if `sequence` is `None`.

### `sliding_window(s, window)`

List of every overlapping substring of length `window`, step 1. Returns `[]` if `window` is longer than `s`.

### `at_sliding_window(seq, window)`

A/T richness of each sliding window: `[at_richness(w) for w in sliding_window(seq, window)]`. Useful for finding A/T-rich stretches around insertion sites.

### `local_align(query, target)`

Best local (Smith-Waterman) alignment, via Biopython's `PairwiseAligner`. Scores: match +2, mismatch −1, gap open −4, gap extend −0.5. Returns a Biopython alignment object; `.score` is the number used for ranking candidate off-target spacers.

### `extract_spacers(sequence, pam, spacer_len=20, spacer_side="5prime", search_strands="both", return_rna=False)`

Finds every PAM match (IUPAC codes allowed, e.g. `"NGG"`) and returns the adjacent spacer of `spacer_len` bp.

| Parameter | Meaning |
|---|---|
| `spacer_side` | `"5prime"`: spacer immediately 5′ of the PAM. `"3prime"`: immediately 3′ of it. |
| `search_strands` | `"+"`, `"-"` or `"both"` |
| `return_rna` | `True` returns the spacer with `T` replaced by `U` |

Returns a list of [`SpacerHit`](#spacerhit). Hits whose spacer would fall off either end are skipped.

> **Note:** `return_rna` only affects `+` strand hits. On `-` strand hits, `spacer_top` is always returned as + strand DNA.

### `find_pam_sites_with_optional_protospacer(segment, pam_motif, protospacer=None, pam_side="3prime", include_reverse_complement=False)`

Finds PAM sites using an IUPAC-degenerate motif, optionally requiring an exact protospacer match next to each PAM.

| Parameter | Meaning |
|---|---|
| `pam_side` | `"3prime"` (`"3"`, `"3'"`): protospacer immediately upstream, i.e. protospacer+PAM. `"5prime"` (`"5"`, `"5'"`): PAM+protospacer. |
| `include_reverse_complement` | Also scan the − strand |

Returns a list of dicts sorted by `pam_start`, then strand, with coordinates on the + strand of `segment`. Each hit has `pam_start`, `pam_end`, `strand`, `pam_seq`; when `protospacer` is given, also `protospacer_start`, `protospacer_end`, `protospacer_seq` and `pam_side`. Raises `ValueError` for an empty motif, an unknown IUPAC code, or an invalid `pam_side`.

### `degenerate_hamming(window, motif_iupac)`

Number of mismatches between equal-length `window` and an IUPAC motif. `0` is a perfect match. Raises `ValueError` on a length mismatch or unknown code.

### `degenerate_match_count(window, motif_iupac)`

The complement of the above: number of matching positions, where higher is better.

### `scan_degenerate_hamming(seq, motif_iupac)`

Scores every window in `seq` against the motif. Returns a list of `(start, mismatches, window)` sorted best-first.

### `best_hit_fewest_mismatches_closest_center(seq, motif_iupac)`

Single best `(start, mismatches, window)`, choosing by fewest mismatches, then closeness to the centre of `seq`, then lowest start index. Useful when you expect one PAM near the middle of an extracted window.

### `build_log_odds_pwm(counts_df, background=None, pseudocount=1.0, log_base=2)`

Builds a log-odds position weight matrix from a counts table (rows = positions, columns = letters such as A, C, G, T).

| Parameter | Meaning |
|---|---|
| `background` | Dict or Series of per-letter background probabilities; `None` means uniform |
| `pseudocount` | Added to every count (Laplace smoothing) to avoid zeros |
| `log_base` | `2` for bits (the default) |

Returns `(pwm_df, prob_df)`: the log-odds matrix and the smoothed probabilities behind it. Raises `ValueError` if the background is missing letters or has non-positive values.

### `score_sequence_pwm(seq, pwm_df, *, unknown_penalty=None)`

Sums PWM scores across positions. `len(seq)` must equal the number of rows in `pwm_df`. A character outside the PWM's columns raises `ValueError`, unless `unknown_penalty` is a number, which is added instead.

### `pwm_max_possible_score(pwm_df)`

Highest score any sequence could reach: the sum of the best letter at each position. Useful for normalizing scores.

---

## NovoCAST_helper

Prediction for novel (engineered) CAST systems: where a given spacer should direct insertion, based on the spacing measured in our profiling data rather than on sequencing reads.

### `INSERTION_DISTANCE`, `TARGET_SITE_WINDOW`

Module constants: the insertion point lies `INSERTION_DISTANCE` (71) bp from the end of the PAM, and `TARGET_SITE_WINDOW` (10) bp of context is taken on each side of it.

### `predict_insertion_from_spacer(guideRNA, genome_sequence)`

Predicts where a CAST would insert for a given guide, using a fixed spacing of **71 bp** downstream of the PAM, with a ±10 bp target-site window. Assumes a 3 bp PAM.

Returns a one-row DataFrame with `PAM`, `protospacer`, `gRNA_strand`, `intervening`, `insertion_length`, `target_site`, and the coordinate columns `PAM_start/end`, `proto_start/end`, `intervene_start/end`, `ts_start/end`. The `aln_score` column exists but stays `NaN`, so the frame lines up with the one from `characterize_insertion_from_position`.

This is a prediction from the guide alone and uses no sequencing data. For the same values measured from real insertions, use [`characterize_insertion_from_spacer`](#characterize_insertion_from_spacertsv-pam-guiderna-genome_sequence). Raises `IndexError` if the guide isn't in the genome.

> **Note:** this function used to live in `postprocessHelpers`. It is still re-exported there, so `hp.predict_insertion_from_spacer(...)` keeps working in older notebooks, but new code should import it from `tagmentation.NovoCAST_helper`.

---

## TsvMapHelperFunctions

Analysis of insertion TSVs: subsetting, on-target percentages, target-site duplication motifs, profile plots, and relating insertions to guides and PAMs.

### `which(df, pos)`

Rows of an insertion DataFrame whose `ins0` equals `pos`.

### `between(df, s, e)`

Rows whose `ins0` falls in `[s, e]`, inclusive of both ends.

### `percent_within_distance(insertions, target_position, window_bp)`

Percentage (0–100) of insertion positions within ±`window_bp` of `target_position`. Returns `0.0` for an empty input. `insertions` can be a list or a pandas Series of coordinates. This is the usual on-target measure.

### `top_n_insertions_frequencies(insertions, N)`

The `N` most common insertion positions as `(position, frequency)`, sorted by descending frequency then ascending position. Frequencies are fractions of all insertions passed in. Returns `[]` if the input is empty or `N <= 0`.

### `extract_tsd_centered_motifs_5p(tsv_path, genome_fasta, tsd_len=5, flank_left=10, flank_right=10, min_mapq=None, orient_to_plus=True)`

Extracts the sequence around each insertion, centred on the target-site duplication (TSD). **For 5′ junction reads only** (`--donor-side 5p`), where `ins0` is the read's reference 5′ end: on `+` reads that is the left edge of the TSD, and on `-` reads the right edge. The TSD start is computed per read accordingly.

| Parameter | Meaning |
|---|---|
| `tsv_path` | Path to an insertion TSV (read from disk, not a DataFrame) |
| `genome_fasta` | Reference FASTA; records are matched against the TSV's `ref` column |
| `tsd_len` | TSD length in bp |
| `flank_left`, `flank_right` | bp of context on each side |
| `min_mapq` | Drop rows below this MAPQ |
| `orient_to_plus` | Reverse-complement `-` strand motifs so all motifs read the same way |

Returns the kept rows as a DataFrame with added columns `tsd_start0`, `motif_seq`, `motif_len`, `flank_left`, `tsd_len`, `flank_right`. Rows are dropped when coordinates fall off the contig or the `ref` is absent from the FASTA. Raises `ValueError` if `ref`, `ins0` or `strand` is missing, or if `min_mapq` is set without a `mapq` column.

The `motif_seq` column is what you feed to `build_log_odds_pwm` after counting bases per position.

### `plot_insertion_profile(tsv_file, gRNA, genome_sequence, fig_filename, bins=100)`

Histogram of insertion positions across the genome, with the guide region shaded and marked by an arrow. Reads the TSV from disk and **keeps only MAPQ ≥ 30**. Saves to `fig_filename` and also shows the figure. Returns `None`. Needs Arial; matplotlib falls back to a default font with a warning if it's unavailable.

### `characterize_insertion_from_position(tsv, PAM, guideRNA, genome_sequence, spacer_len=20)`

Works backwards from one observed insertion to the candidate spacers that could explain it — the usual way to characterize a possible off-target event. `tsv` is a **single row** (one insertion), already loaded; it searches a 120 bp window on the appropriate side and scores every spacer against `guideRNA` by local alignment.

Returns `(df, alns)`: one row per candidate spacer, with the same columns as `predict_insertion_from_spacer` plus `aln_score`, and the list of alignment objects. Sort by `aln_score` to find the best candidate; the function deliberately does no filtering.

> **Note:** the `PAM` argument is currently ignored — the search is hard-coded to `NGG`.

### `characterize_insertion_from_spacer(tsv, PAM, guideRNA, genome_sequence)`

The measured counterpart of `predict_insertion_from_spacer`: it locates the guide, looks for real insertions within ±10 bp of the position predicted 71 bp from the PAM, and describes the most common one (`statistics.mode`). `tsv` is a full insertion DataFrame.

Returns a one-row DataFrame with the same columns, or an **empty DataFrame** when no insertion is found in the window. Works only for guides that map uniquely to the genome.

> **Note:** the `PAM` argument is currently ignored; a 3 bp PAM is assumed.

---

## NGSanalysis_module (legacy)

Functions behind the legacy bbmap workflow (`legacy/CAST_TagmentationNotebook.ipynb` and `legacy/tagmentation_app.py`). New analyses should use `pipeline/map_junction_insertions.py` instead.

### `process_reads(fastq, primer, first_ten_in_donor, refgenome, bbmap_bin, samtools_bin, path_to_output_dir)`

The whole legacy pipeline: trim reads at `primer`, drop reads starting with `first_ten_in_donor` (non-enzymatic events), write `trimmed.fastq`, index and map with bbmap (`minid=0.9 ambig=toss local=t slow=t tipsearch=200`, 20 GB heap), then call insertion sites on both strands.

Returns `[forward_insertions, reverse_insertions]`, two lists of positions. `bbmap_bin` and `samtools_bin` are the **folders** holding `bbmap.sh` and `samtools`; pass `""` to use whatever is on your PATH. Creates the output directory if needed.

> **Note:** `ambig=toss` discards multi-mapping reads. Both strands are called with `tn5_offset=5`.

### `fasta_to_string(fasta_path)`

Sequence of the first FASTA record, as a string.

### `trim_reads_by_primer(fastq_filepath, primer)`

Finds `primer` in each read and keeps everything after it, preserving quality scores. Returns a list of `(read_id, seq, qualities)` for reads containing the primer. Reads without it are dropped. Returns `None` (with a message) if no read matched, which is the signal that the primer sequence is wrong.

### `write_trimmed_fastq(trimmed_reads, output_path)`

Writes those tuples to a FASTQ file, converting qualities back to Phred+33. Creates the parent directory if needed.

### `aligned_ref_length(cigar)`

Number of reference bases a CIGAR string consumes (counting `M`, `D`, `N`, `=`, `X`). This is what makes the reverse-strand end position correct for clipped or gapped alignments.

### `extract_insertion_sites(samtools_bin, bam_file, isforward, tn5_offset=5)`

Reads a BAM with `samtools view` and returns insertion positions for one strand.

| Parameter | Meaning |
|---|---|
| `samtools_bin` | Folder containing `samtools`; `""` uses PATH |
| `isforward` | `True` for forward-strand reads, `False` for reverse |
| `tn5_offset` | bp shifted into the read from the junction edge |

Forward reads give `start0 + tn5_offset`; reverse reads give `start0 + aligned_ref_length - 1 - tn5_offset`. Reads on the other strand are skipped, so calling it twice partitions the BAM.

> **Note:** the 5 bp offset suits Tn5 cut sites. For CAST junction reads the read's genomic 5′ end is already the insertion point, so pass `tn5_offset=0` if you don't want the shift.

### `sgRNA_finder(sgRNA, genome, PAM_len, natural_system)`

Locates an sgRNA and the PAM beside it. Set `natural_system=False` for novel CASTs, where the PAM sits on the other side.

Returns `(sgRNA_start_site, PAM_start_site, PAM_end_site, is_rc)`, where `is_rc` is `True` if the guide was found as a reverse complement. Returns `None` (with a message) if the guide isn't in the genome, which makes the caller fail when unpacking.

> **Note:** these are not a sorted range. When the PAM lies upstream of the guide, `PAM_end_site` is numerically *smaller* than `PAM_start_site`. For a guide at position 10 with `PAM_len=3`, a natural system on the + strand returns `(10, 10, 7, False)`, while a novel system returns `(10, 20, 23, False)`. `PAM_start_site` is the edge that `integration_finder` measures distances from in natural systems, and `PAM_end_site` the one it uses for novel systems.

### `integration_finder(insertion_idx_list, ccdb_gene, ccdb_start_site, PAM_start_site, PAM_end_site, natural_system, is_rc)`

Keeps insertions inside the target gene and converts each to a signed distance from the PAM, taking strand and system type into account. Returns a list of distances, used for the integration-profile histogram and as the on-target count.

### `plot_insertion_histogram(forward_positions, reverse_positions, region_start=0, region_end=None, bins=100, title="Insertion Sites", highlight_regions=None, filename=None)`

Overlaid histograms of top- and bottom-strand insertions. `region_end` defaults to the largest position. `highlight_regions` is a list of `(start, end)` or `(start, end, color)` drawn as horizontal bars under the axis, for marking the guide and PAM. Saves to `filename` when given (300 dpi).

Returns `(fig, ax)`. In Streamlit, pass the figure: `st.pyplot(result[0])`.

### `plot_integration_profile(data, title=..., xlabel=..., ylabel=..., xlim=(20, 100), filename=None, ...)`

Histogram of insertion distances from the PAM, i.e. the output of `integration_finder`. Also accepts font sizes, colors, `linewidth` and `dpi`. Saves to `filename` when given, otherwise shows the plot. Returns the `matplotlib.pyplot` module, not a figure.

### `save_to_temp(uploadedFile, tmpdir)`

Writes an uploaded file object (from Streamlit) to a temporary file in `tmpdir` and returns its path, so command-line tools can read it. Creates `tmpdir` if needed. The suffix is always `.fasta`, whatever the upload was.
