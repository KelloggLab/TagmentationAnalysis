#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [-f] <profile.txt> [bin_size]" >&2
    echo "  -f          : plot per-bin frequency (fraction of ALL reads, both strands) instead of raw counts" >&2
    echo "  profile.txt : whitespace-delimited file; genomic position in col 3, strand (+/-) in col 5" >&2
    echo "  bin_size    : bp per bin (default 1000)" >&2
    exit 1
}

FREQ=0
while getopts ":f" opt; do
    case "$opt" in
        f) FREQ=1 ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

[ $# -ge 1 ] || usage

INFILE="$1"
BIN="${2:-1000}"

[ -f "$INFILE" ] || { echo "Error: file '$INFILE' not found" >&2; exit 1; }

# Combined total (both strands), used as the single denominator when -f is given
TOTAL=$(awk '$5=="+" || $5=="-"' "$INFILE" | wc -l)
[ "$TOTAL" -gt 0 ] || TOTAL=1

YLABEL="Count (+ strand up / - strand down)"
[ "$FREQ" -eq 1 ] && YLABEL="Frequency of all reads (+ strand up / - strand down)"

gnuplot -persist \
  -e "infile='${INFILE}'" \
  -e "bin=${BIN}" \
  -e "freq=${FREQ}" \
  -e "total=${TOTAL}" \
  -e "ylabel='${YLABEL}'" \
  - <<'GPEOF'
set terminal qt size 1400,500

set xlabel "Genomic position (bp)"
set ylabel ylabel
set style fill solid
set boxwidth bin*0.9
set format x "%.0f"
set xzeroaxis lt -1

plot \
  infile using (strcol(5) eq "+" ? floor($3/bin)*bin : 1/0):(freq ? 1.0/total : 1)  smooth frequency with boxes lc rgb "blue" title "+ strand", \
  infile using (strcol(5) eq "-" ? floor($3/bin)*bin : 1/0):(freq ? -1.0/total : -1) smooth frequency with boxes lc rgb "red"  title "- strand"
GPEOF
