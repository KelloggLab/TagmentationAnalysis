#!/usr/bin/env bash
#
# chipseq_pipeline.sh
#
# End-to-end ChIP-seq pipeline for 1-3 biological replicates of true
# paired-end FASTQ (R1 + R2 per sample) against a bacterial reference genome.
# R1 and R2 are aligned TOGETHER as read pairs (bwa mem paired mode), and
# MACS3 is run in BAMPE mode, which uses each properly-paired fragment's own
# aligned span as the fragment -- no fragment-size guessing (--extsize) is
# needed, since the true insert size comes directly from the alignment.
#
# Steps:
#   0. Adapter/quality trim each R1/R2 pair together (Trim Galore, --paired)
#      -- ON by default
#   1. Build BWA index (if not already present)
#   2. Align each replicate's R1+R2 as a pair (bwa mem, paired-end)
#   3. Fixmate, sort, MARK duplicates (not removed -- see note at the
#      markdup call), filter to properly-paired + MAPQ
#   4. Call peaks per replicate with MACS3 (-f BAMPE, --keep-dup auto,
#      bedgraph output enabled)
#   5. Build fold-enrichment tracks (macs3 bdgcmp)
#   6. Build a reproducible "consensus" peak set (peaks present in ALL
#      replicates), via bedtools intersect (2 reps) / multiinter (3 reps)
#   7. Generate a genome-wide fold-enrichment plot for EACH replicate
#
# Trimming is always run by default (safe even on already-clean data --
# Trim Galore only removes adapter/low-quality bases it actually detects).
# Disable with --no-trim if you've already trimmed your reads elsewhere.
#
# ------------------------------------------------------------------------
# USAGE
#   ./chipseq_pipeline.sh \
#       --r1-chip rep1_chip_R1.fastq rep2_chip_R1.fastq [rep3_chip_R1.fastq] \
#       --r2-chip rep1_chip_R2.fastq rep2_chip_R2.fastq [rep3_chip_R2.fastq] \
#       --r1-inp  rep1_inp_R1.fastq  rep2_inp_R1.fastq  [rep3_inp_R1.fastq] \
#       --r2-inp  rep1_inp_R2.fastq  rep2_inp_R2.fastq  [rep3_inp_R2.fastq] \
#       -r reference.fasta \
#       [-o output_dir] [-t threads] [-q mapq] [-p pvalue] [--no-trim]
#
#   Files within each --r1-chip/--r2-chip/--r1-inp/--r2-inp list must be in
#   the SAME replicate order -- the Nth file in every list is replicate N,
#   and --r1-chip[N] / --r2-chip[N] are the two mates of the same fragment.
#   1, 2, or 3 replicates are supported; all four lists must be the same
#   length.
#
# REQUIRED TOOLS ON PATH: bwa, samtools, macs3, bedtools, python3, trim_galore
#   (python3 needs pandas + matplotlib)
# ------------------------------------------------------------------------

set -euo pipefail

# Directory holding this script, so sibling helpers resolve however it
# was invoked.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------- defaults ----------------------------------
THREADS=8
MAPQ=30
PVAL=0.01
OUTDIR="chipseq_results"
TRIM=1               # trimming is ON by default; disable with --no-trim
INCLUDE_PEAKS=0       # gray consensus-peak shading on plots is OFF by default; enable with --include-peaks
REMOVE_DUP=0          # duplicates are MARKED (not removed) by default; remove with --remove-dup
REFERENCE=""

R1_CHIP=()
R2_CHIP=()
R1_INP=()
R2_INP=()

usage() {
    cat <<EOF
ChIP-seq pipeline (paired-end): trim -> paired alignment -> dedup/filter ->
MACS3 BAMPE peak calling -> bedtools consensus peaks -> genome-wide
fold-enrichment plots (per replicate).

Usage:
  $0 --r1-chip R1_CHIP_REP1 [R1_CHIP_REP2] [R1_CHIP_REP3] \\
     --r2-chip R2_CHIP_REP1 [R2_CHIP_REP2] [R2_CHIP_REP3] \\
     --r1-inp  R1_INPUT_REP1 [R1_INPUT_REP2] [R1_INPUT_REP3] \\
     --r2-inp  R2_INPUT_REP1 [R2_INPUT_REP2] [R2_INPUT_REP3] \\
     -r reference.fasta [options]

Required (1-3 files each, same replicate order, same count in all four):
  --r1-chip FILE...   ChIP R1 FASTQ, one per replicate
  --r2-chip FILE...   ChIP R2 FASTQ (mate of --r1-chip), one per replicate
  --r1-inp  FILE...   Input R1 FASTQ, one per replicate
  --r2-inp  FILE...   Input R2 FASTQ (mate of --r1-inp), one per replicate
  -r, --reference FILE   Reference genome FASTA

Options:
  -o, --outdir DIR    Output directory              (default: chipseq_results)
  -t, --threads INT   Threads                       (default: 8)
  -q, --mapq INT      MAPQ filter threshold         (default: 30)
  -p, --pvalue FLOAT  MACS3 p-value cutoff          (default: 0.01)
  --no-trim            Skip adapter/quality trimming (default: trimming is
                                                       ON; safe to leave on
                                                       even for already-
                                                       clean data)
  --include-peaks      Shade consensus peaks as gray bands on the genome-wide
                                                       fold-enrichment plots
                                                       (default: OFF -- peaks
                                                       are still called and
                                                       written to consensus/,
                                                       just not drawn)
  --remove-dup          Remove duplicate reads (samtools markdup -r) instead
                                                       of only marking them.
                                                       (default: OFF -- reads
                                                       are marked and left in;
                                                       MACS3's --keep-dup auto
                                                       handles duplicates at
                                                       the peak-calling step)
  -h, --help           Show this help message

Requires on PATH: bwa, samtools, macs3, bedtools, python3, trim_galore
  (python3 needs pandas + matplotlib)

Example (2 replicates):
  $0 --r1-chip rep1_chip_R1.fastq rep2_chip_R1.fastq \\
     --r2-chip rep1_chip_R2.fastq rep2_chip_R2.fastq \\
     --r1-inp  rep1_inp_R1.fastq  rep2_inp_R1.fastq \\
     --r2-inp  rep1_inp_R2.fastq  rep2_inp_R2.fastq \\
     -r reference.fasta -o results -t 8

Example (3 replicates):
  $0 --r1-chip rep1_chip_R1.fastq rep2_chip_R1.fastq rep3_chip_R1.fastq \\
     --r2-chip rep1_chip_R2.fastq rep2_chip_R2.fastq rep3_chip_R2.fastq \\
     --r1-inp  rep1_inp_R1.fastq  rep2_inp_R1.fastq  rep3_inp_R1.fastq \\
     --r2-inp  rep1_inp_R2.fastq  rep2_inp_R2.fastq  rep3_inp_R2.fastq \\
     -r reference.fasta -o results -t 8
EOF
    exit 1
}

[[ $# -eq 0 ]] && usage

# ---------------------------- parse args ---------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --r1-chip)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do R1_CHIP+=("$1"); shift; done
            ;;
        --r2-chip)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do R2_CHIP+=("$1"); shift; done
            ;;
        --r1-inp)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do R1_INP+=("$1"); shift; done
            ;;
        --r2-inp)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do R2_INP+=("$1"); shift; done
            ;;
        -r|--reference)
            REFERENCE="${2:?Missing argument for $1}"; shift 2 ;;
        -o|--outdir)
            OUTDIR="${2:?Missing argument for $1}"; shift 2 ;;
        -t|--threads)
            THREADS="${2:?Missing argument for $1}"; shift 2 ;;
        -q|--mapq)
            MAPQ="${2:?Missing argument for $1}"; shift 2 ;;
        -p|--pvalue)
            PVAL="${2:?Missing argument for $1}"; shift 2 ;;
        --no-trim)
            TRIM=0; shift ;;
        --trim)
            TRIM=1; shift ;;
        --include-peaks)
            INCLUDE_PEAKS=1; shift ;;
        --remove-dup)
            REMOVE_DUP=1; shift ;;
        -h|--help)
            usage ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage
            ;;
    esac
done

# ---------------------------- validate replicate counts -------------------
N=${#R1_CHIP[@]}

if [[ $N -eq 0 ]]; then
    echo "ERROR: no replicates given -- provide at least 1 file to each of --r1-chip/--r2-chip/--r1-inp/--r2-inp" >&2
    usage
fi
if [[ $N -gt 3 ]]; then
    echo "ERROR: at most 3 replicates are supported (got $N)" >&2
    exit 1
fi
if [[ ${#R2_CHIP[@]} -ne $N || ${#R1_INP[@]} -ne $N || ${#R2_INP[@]} -ne $N ]]; then
    echo "ERROR: --r1-chip, --r2-chip, --r1-inp, --r2-inp must all supply the same number of files" >&2
    echo "  --r1-chip: $N   --r2-chip: ${#R2_CHIP[@]}   --r1-inp: ${#R1_INP[@]}   --r2-inp: ${#R2_INP[@]}" >&2
    exit 1
fi

: "${REFERENCE:?Missing -r/--reference reference.fasta}"

for f in "$REFERENCE" "${R1_CHIP[@]}" "${R2_CHIP[@]}" "${R1_INP[@]}" "${R2_INP[@]}"; do
    [[ -f "$f" ]] || { echo "ERROR: file not found: $f"; exit 1; }
done

echo "Detected $N replicate(s). Paired-end mode (R1+R2 aligned together)."
if [[ "$REMOVE_DUP" -eq 1 ]]; then
    echo "Duplicate reads will be REMOVED (--remove-dup)."
else
    echo "Duplicate reads will be marked, not removed."
fi

mkdir -p "$OUTDIR"/{trimmed,bam,consensus,plots,logs}
for ((i=1; i<=N; i++)); do
    mkdir -p "$OUTDIR/macs3_rep${i}"
done
LOG="$OUTDIR/logs/pipeline.log"
echo "=== ChIP-seq pipeline (paired-end, $N replicate(s)) started: $(date) ===" | tee "$LOG"

# ---------------------------- 1. reference index ---------------------------
if [[ ! -f "${REFERENCE}.bwt" ]]; then
    echo "[1/7] Building BWA index for $REFERENCE..." | tee -a "$LOG"
    bwa index "$REFERENCE" 2>>"$LOG"
else
    echo "[1/7] BWA index already exists, skipping." | tee -a "$LOG"
fi

if [[ ! -f "${REFERENCE}.fai" ]]; then
    samtools faidx "$REFERENCE"
fi
GENOME_SIZE=$(awk '{sum+=$2} END{print sum}' "${REFERENCE}.fai")
echo "    Effective genome size: $GENOME_SIZE bp" | tee -a "$LOG"

# ---------------------------- helper functions -----------------------------
# trim_pair sets the globals TRIMMED_R1 / TRIMMED_R2 as a side effect (bash
# 3.2, shipped on macOS, has neither `mapfile` nor namerefs to return two
# values cleanly).
trim_pair () {
    local r1=$1
    local r2=$2
    local sample=$3

    if [[ "$TRIM" -eq 0 ]]; then
        echo "    Trimming disabled (--no-trim); using $r1 / $r2 as-is." >> "$LOG"
        TRIMMED_R1="$r1"
        TRIMMED_R2="$r2"
        return
    fi

    echo "    Trimming $r1 + $r2 -> $sample ..." >> "$LOG"
    if ! trim_galore --paired --gzip "$r1" "$r2" -o "$OUTDIR/trimmed" --basename "$sample" >>"$LOG" 2>&1; then
        echo "ERROR: trim_galore failed on $r1 / $r2 -- see $LOG for details" >> "$LOG"
        echo "ERROR: trim_galore failed on $r1 / $r2 -- see $LOG for details" >&2
        echo "--- last 20 lines of log ---" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi

    # Trim Galore's paired-mode output is named <basename>_val_1.fq.gz /
    # <basename>_val_2.fq.gz, but fall back to matching on the original
    # input filenames' stems in case the installed version names things
    # differently.
    local r1_stem r2_stem
    r1_stem=$(basename "$r1"); r1_stem="${r1_stem%.gz}"; r1_stem="${r1_stem%.fastq}"; r1_stem="${r1_stem%.fq}"
    r2_stem=$(basename "$r2"); r2_stem="${r2_stem%.gz}"; r2_stem="${r2_stem%.fastq}"; r2_stem="${r2_stem%.fq}"

    TRIMMED_R1=$(find "$OUTDIR/trimmed" -maxdepth 1 \( -name "${sample}*val_1*.f*q*" -o -name "${r1_stem}*val*.f*q*" \) | head -1)
    TRIMMED_R2=$(find "$OUTDIR/trimmed" -maxdepth 1 \( -name "${sample}*val_2*.f*q*" -o -name "${r2_stem}*val*.f*q*" \) | head -1)

    if [[ -z "$TRIMMED_R1" || -z "$TRIMMED_R2" ]]; then
        echo "ERROR: trim_galore ran but trimmed R1/R2 for '$sample' were not found in $OUTDIR/trimmed" >> "$LOG"
        echo "ERROR: trim_galore ran but trimmed R1/R2 for '$sample' were not found in $OUTDIR/trimmed" >&2
        echo "Contents of $OUTDIR/trimmed:" >&2
        ls -la "$OUTDIR/trimmed" | tee -a "$LOG" >&2
        exit 1
    fi
}

count_fastq_reads () {
    local f=$1
    local n
    if [[ "$f" == *.gz ]]; then
        n=$(zcat < "$f" | wc -l)
    else
        n=$(wc -l < "$f")
    fi
    echo $(( n / 4 ))
}

report_read_counts () {
    local r1=$1
    local r2=$2
    local sample=$3

    local n_r1 n_r2 n_paired
    n_r1=$(count_fastq_reads "$r1")
    n_r2=$(count_fastq_reads "$r2")
    # "properly paired" in flagstat counts individual read records (both
    # mates), so divide by 2 to get the number of R1-R2 fragments.
    n_paired=$(samtools flagstat "$OUTDIR/bam/${sample}.filtered.bam" | awk '/properly paired/ {print $1}')
    n_paired=$(( n_paired / 2 ))

    echo "    $sample read counts: R1 reads=$n_r1  R2 reads=$n_r2  R1-R2 paired reads=$n_paired" | tee -a "$LOG"
}

align_dedup_filter () {
    local r1=$1
    local r2=$2
    local sample=$3
    echo "Processing $sample..." | tee -a "$LOG"

    # bwa mem in paired mode (two FASTQs) aligns R1/R2 as mates; fixmate
    # (run directly on bwa's output, which already has mates adjacent)
    # fills in mate-coordinate/mate-score tags that markdup needs.
    bwa mem -t "$THREADS" "$REFERENCE" "$r1" "$r2" 2>>"$LOG" \
        | samtools fixmate -m -u - - 2>>"$LOG" \
        | samtools sort -@ "$THREADS" -o "$OUTDIR/bam/${sample}.sorted.bam" - 2>>"$LOG"
    samtools index "$OUTDIR/bam/${sample}.sorted.bam"

    # Duplicates are MARKED (flag 0x400) by default and not removed; pass
    # --remove-dup to actually strip them out (samtools markdup -r) instead.
    if [[ "$REMOVE_DUP" -eq 1 ]]; then
        samtools markdup -r "$OUTDIR/bam/${sample}.sorted.bam" "$OUTDIR/bam/${sample}.dedup.bam" 2>>"$LOG"
    else
        samtools markdup "$OUTDIR/bam/${sample}.sorted.bam" "$OUTDIR/bam/${sample}.dedup.bam" 2>>"$LOG"
    fi
    samtools index "$OUTDIR/bam/${sample}.dedup.bam"

    # -f 2 (properly paired) keeps only concordant fragments, whose insert
    # size MACS3 will read directly from the BAM in BAMPE mode below.
    # -F 0x900 drops secondary/supplementary alignments.
    samtools view -b -q "$MAPQ" -f 2 -F 0x900 "$OUTDIR/bam/${sample}.dedup.bam" > "$OUTDIR/bam/${sample}.filtered.bam"
    samtools index "$OUTDIR/bam/${sample}.filtered.bam"

    echo "    $sample flagstat:" | tee -a "$LOG"
    samtools flagstat "$OUTDIR/bam/${sample}.filtered.bam" | tee -a "$LOG"
}

# ---------------------------- 0. trimming -----------------------------------
if [[ "$TRIM" -eq 1 ]]; then
    echo "[0/7] Trimming adapters/low-quality bases (Trim Galore, paired mode)..." | tee -a "$LOG"
else
    echo "[0/7] Skipping trimming (--no-trim set)." | tee -a "$LOG"
fi

TRIMMED_CHIP_R1=(); TRIMMED_CHIP_R2=()
TRIMMED_INP_R1=();  TRIMMED_INP_R2=()
for ((i=1; i<=N; i++)); do
    trim_pair "${R1_CHIP[i-1]}" "${R2_CHIP[i-1]}" "chip_rep${i}"
    TRIMMED_CHIP_R1[i]="$TRIMMED_R1"; TRIMMED_CHIP_R2[i]="$TRIMMED_R2"

    trim_pair "${R1_INP[i-1]}" "${R2_INP[i-1]}" "input_rep${i}"
    TRIMMED_INP_R1[i]="$TRIMMED_R1"; TRIMMED_INP_R2[i]="$TRIMMED_R2"
done

# ---------------------------- 2-3. align, dedup, filter --------------------
for ((i=1; i<=N; i++)); do
    echo "[2-3/7]" | tee -a "$LOG"
    align_dedup_filter "${TRIMMED_CHIP_R1[i]}" "${TRIMMED_CHIP_R2[i]}" "chip_rep${i}"
    report_read_counts "${TRIMMED_CHIP_R1[i]}" "${TRIMMED_CHIP_R2[i]}" "chip_rep${i}"

    align_dedup_filter "${TRIMMED_INP_R1[i]}"  "${TRIMMED_INP_R2[i]}"  "input_rep${i}"
    report_read_counts "${TRIMMED_INP_R1[i]}"  "${TRIMMED_INP_R2[i]}"  "input_rep${i}"
done

# ---------------------------- 4. MACS3 peak calling -------------------------
# -f BAMPE tells MACS3 to build each fragment's pileup directly from its
# properly-paired R1/R2 span, instead of extending a single-end tag by a
# guessed --extsize.
echo "[4/7] Calling peaks with MACS3 (paired-end / BAMPE mode)..." | tee -a "$LOG"
for ((i=1; i<=N; i++)); do
    macs3 callpeak \
        -t "$OUTDIR/bam/chip_rep${i}.filtered.bam" \
        -c "$OUTDIR/bam/input_rep${i}.filtered.bam" \
        -f BAMPE -g "$GENOME_SIZE" -n "rep${i}" \
        --outdir "$OUTDIR/macs3_rep${i}" \
        -p "$PVAL" --keep-dup auto -B 2>>"$LOG"
    echo "    rep${i} peaks: $(wc -l < "$OUTDIR/macs3_rep${i}/rep${i}_peaks.narrowPeak")" | tee -a "$LOG"
done

# ---------------------------- 5. fold-enrichment tracks ---------------------
echo "[5/7] Building fold-enrichment tracks..." | tee -a "$LOG"
for ((i=1; i<=N; i++)); do
    macs3 bdgcmp \
        -t "$OUTDIR/macs3_rep${i}/rep${i}_treat_pileup.bdg" \
        -c "$OUTDIR/macs3_rep${i}/rep${i}_control_lambda.bdg" \
        -o "$OUTDIR/macs3_rep${i}/rep${i}_FE.bdg" -m FE 2>>"$LOG"
done

# ---------------------------- 6. consensus peaks (bedtools) -----------------
# A peak is "reproducible"/consensus if it is present in ALL N replicates.
# N=1: consensus is just that replicate's peak set.
# N=2: equivalent to a two-way bedtools intersect.
# N=3: bedtools multiinter, keeping regions covered by all 3 files.
echo "[6/7] Building consensus peak set across $N replicate(s)..." | tee -a "$LOG"
if [[ $N -eq 1 ]]; then
    cut -f1-3 "$OUTDIR/macs3_rep1/rep1_peaks.narrowPeak" \
        | sort -k1,1 -k2,2n \
        | bedtools merge -i - \
        > "$OUTDIR/consensus/consensus_peaks.bed"
else
    SORTED_BEDS=()
    for ((i=1; i<=N; i++)); do
        cut -f1-3 "$OUTDIR/macs3_rep${i}/rep${i}_peaks.narrowPeak" \
            | sort -k1,1 -k2,2n \
            > "$OUTDIR/consensus/rep${i}_sorted.bed"
        SORTED_BEDS+=("$OUTDIR/consensus/rep${i}_sorted.bed")
    done
    bedtools multiinter -i "${SORTED_BEDS[@]}" \
        | awk -v n="$N" 'BEGIN{OFS="\t"} $4==n {print $1,$2,$3}' \
        | sort -k1,1 -k2,2n \
        | bedtools merge -i - \
        > "$OUTDIR/consensus/consensus_peaks.bed"
fi

N_CONSENSUS=$(wc -l < "$OUTDIR/consensus/consensus_peaks.bed")
echo "    consensus (merged, present in all $N replicate(s)) peaks: $N_CONSENSUS" | tee -a "$LOG"

for ((i=1; i<=N; i++)); do
    bedtools intersect -u \
        -a "$OUTDIR/macs3_rep${i}/rep${i}_peaks.narrowPeak" \
        -b "$OUTDIR/consensus/consensus_peaks.bed" \
        > "$OUTDIR/consensus/rep${i}_reproducible.narrowPeak"
    echo "    rep${i} peaks supported by all replicates: $(wc -l < "$OUTDIR/consensus/rep${i}_reproducible.narrowPeak")" | tee -a "$LOG"
done

# ---------------------------- 7. genome-wide plots (per replicate) ----------
echo "[7/7] Generating genome-wide enrichment plots..." | tee -a "$LOG"

PLOT_ARGS=()
for ((i=1; i<=N; i++)); do
    PLOT_ARGS+=(--rep "$OUTDIR/macs3_rep${i}/rep${i}_FE.bdg" "Replicate ${i}" \
                "$OUTDIR/plots/rep${i}_genomewide_enrichment")
done
if [[ "$INCLUDE_PEAKS" == "1" ]]; then
    PLOT_ARGS+=(--peaks "$OUTDIR/consensus/consensus_peaks.bed")
fi

# Contig lengths come from the .fai, not the summed GENOME_SIZE above: each
# contig gets its own axis, so a chromosome and a plasmid are not drawn on
# top of each other.
python3 -I "$HERE/plot_enrichment.py" --fai "${REFERENCE}.fai" "${PLOT_ARGS[@]}" \
    2>&1 | tee -a "$LOG"

echo "=== Pipeline complete: $(date) ===" | tee -a "$LOG"
echo ""
echo "Key outputs:"
echo "  Consensus peaks:  $OUTDIR/consensus/consensus_peaks.bed"
echo "  Enrichment plots: $OUTDIR/plots/   (one PNG + PDF per replicate per contig)"
echo "  Full log:          $OUTDIR/logs/pipeline.log"
