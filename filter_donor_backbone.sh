#!/usr/bin/env bash
# Remove insertion calls whose read derives from the donor plasmid backbone
# (as opposed to unique host genomic DNA), regardless of where in the genome
# that read happened to align.
#
# Rationale: after LE/RE trimming, a read's remaining sequence is supposed to
# be unique flanking genomic DNA. If that remaining sequence instead matches
# the donor plasmid's *backbone* (ori, oriT, resistance marker, MCS/primer
# sites -- everything outside the Cargo that is meant to integrate), the read
# is not evidence of a genuine, informative integration event: it's either
# leftover free donor plasmid in the prep, or a pre-existing backbone
# integration already baked into the reference strain (e.g. a confirmed
# clone's landing site). Either way it should not be reported as new signal.
#
# This intentionally does NOT screen against the full donor plasmid -- only
# the backbone. A genuine new integration legitimately carries Cargo sequence
# right up to the LE/RE boundary, and that must still be called.
#
# Usage:
#   filter_donor_backbone.sh -g <donor.gb> -f <trimmed.fastq> \
#       -t <insertions.5p.tsv> -o <output.tsv> [-m <min_match_bp>] [-w <workdir>]
#
# Required:
#   -g   Donor plasmid GenBank file. Must contain a misc_feature annotated
#        "Cargo" with location 1..N -- everything after N is treated as
#        backbone and used as the screening reference.
#   -f   Trimmed reads fastq -- the transposon-end-trimmed reads that were
#        aligned to the genome to produce the file given to -t.
#   -t   insertions.5p.tsv to filter. Must have a header row containing a
#        "read_id" column.
#   -o   Path to write the filtered tsv.
#
# Optional:
#   -m   Minimum aligned (M) bp against the backbone to count as a hit.
#        Default: 30.
#   -w   Directory for intermediate files (backbone fasta/index, blacklist).
#        Default: a cache directory next to the donor GenBank file, keyed on
#        its filename, so repeated calls against the same donor (e.g. once
#        per sample) reuse the same bwa index instead of rebuilding it.
#
# Environment:
#   BWA, SAMTOOLS   override the executables used (default: whatever is on
#                    PATH; errors out with a clear message if not found).
#
# Example (one sample, one read type; call once per fastq/tsv you have):
#   filter_donor_backbone.sh \
#       -g ~/Downloads/pmcpdonor_kan_nr_noamp_orit.gb \
#       -f mapped_reads/29_S29_L001_merged.trimmed.fastq \
#       -t mapped_reads/29_S29_L001_merged.insertions.5p.tsv \
#       -o mapped_reads/29_S29_L001_merged.insertions.5p.nobackbone.tsv
#
# To process every read type for a sample, or every sample in a project,
# loop over your own file list and call this script once per fastq/tsv pair
# -- it makes no assumption about naming or directory layout.
set -euo pipefail

MIN_MATCH=30
WORKDIR=""
DONOR_GB=""
TRIMMED=""
TSV=""
OUT=""

usage() {
    cat >&2 <<'EOF'
Usage: filter_donor_backbone.sh -g <donor.gb> -f <trimmed.fastq> -t <insertions.5p.tsv> -o <output.tsv> [-m <min_match_bp>] [-w <workdir>]
EOF
    exit 1
}

while getopts ":g:f:t:o:m:w:" opt; do
    case "$opt" in
        g) DONOR_GB="$OPTARG" ;;
        f) TRIMMED="$OPTARG" ;;
        t) TSV="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        m) MIN_MATCH="$OPTARG" ;;
        w) WORKDIR="$OPTARG" ;;
        *) usage ;;
    esac
done

[ -n "$DONOR_GB" ] && [ -n "$TRIMMED" ] && [ -n "$TSV" ] && [ -n "$OUT" ] || usage
[ -f "$DONOR_GB" ] || { echo "Error: donor GenBank file not found: $DONOR_GB" >&2; exit 1; }
[ -f "$TRIMMED" ]  || { echo "Error: trimmed fastq not found: $TRIMMED" >&2; exit 1; }
[ -f "$TSV" ]      || { echo "Error: insertions tsv not found: $TSV" >&2; exit 1; }

find_tool() {
    local name="$1" override="$2"
    if [ -n "$override" ]; then
        echo "$override"
    elif command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
    else
        echo "Error: '$name' not found on PATH. Set the ${name^^} environment variable to its full path." >&2
        exit 1
    fi
}
BWA_BIN="$(find_tool bwa "${BWA:-}")"
SAMTOOLS_BIN="$(find_tool samtools "${SAMTOOLS:-}")"

if [ -z "$WORKDIR" ]; then
    donor_dir="$(cd "$(dirname "$DONOR_GB")" && pwd)"
    donor_base="$(basename "$DONOR_GB")"
    WORKDIR="${donor_dir}/.donor_backbone_cache/${donor_base}"
fi
mkdir -p "$WORKDIR"

BACKBONE_FASTA="${WORKDIR}/donor_backbone.fasta"

# --- 1. Extract the donor backbone (everything outside the annotated Cargo),
#        reusing the cached copy if it's already up to date. ---
if [ ! -f "$BACKBONE_FASTA" ] || [ "$DONOR_GB" -nt "$BACKBONE_FASTA" ]; then
    python3 - "$DONOR_GB" "$BACKBONE_FASTA" <<'PYEOF'
import re, sys

gb_path, out_path = sys.argv[1], sys.argv[2]

def load_gb_origin(fn):
    seq = []
    started = False
    with open(fn) as f:
        for line in f:
            if line.startswith('ORIGIN'):
                started = True
                continue
            if line.startswith('//'):
                break
            if started:
                seq.append(''.join(line.split()[1:]))
    return ''.join(seq).upper()

def find_cargo_end(fn):
    with open(fn) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.match(r'\s*misc_feature\s+1\.\.(\d+)', line)
        if m and i + 1 < len(lines) and 'Cargo' in lines[i + 1]:
            return int(m.group(1))
    return None

donor = load_gb_origin(gb_path)
cargo_end = find_cargo_end(gb_path)
if cargo_end is None:
    sys.exit("Could not find a 'Cargo' misc_feature spanning 1..N in the GenBank file; "
             "edit this script to set cargo_end manually.")

backbone = donor[cargo_end:]  # donor[0:cargo_end] is Cargo (1-based inclusive -> 0-based exclusive)
with open(out_path, 'w') as out:
    out.write('>donor_backbone\n')
    for i in range(0, len(backbone), 70):
        out.write(backbone[i:i+70] + '\n')

print(f"donor length={len(donor)} cargo=1..{cargo_end} backbone={cargo_end+1}..{len(donor)} "
      f"({len(backbone)} bp) -> {out_path}", file=sys.stderr)
PYEOF
    "$BWA_BIN" index "$BACKBONE_FASTA" 2>/dev/null
fi

# --- 2. Align the trimmed reads to the backbone and collect hits ---
BLACKLIST="${WORKDIR}/$(basename "$TRIMMED").backbone_read_ids.txt"
"$BWA_BIN" mem -t 4 "$BACKBONE_FASTA" "$TRIMMED" 2>/dev/null \
    | "$SAMTOOLS_BIN" view -F 4 - \
    | awk -v minlen="$MIN_MATCH" '{
          cigar=$6; matched=0;
          while (match(cigar, /[0-9]+M/)) {
              matched += substr(cigar,RSTART,RLENGTH-1) + 0;
              cigar = substr(cigar, RSTART+RLENGTH);
          }
          if (matched >= minlen) print $1;
      }' | sort -u > "$BLACKLIST"

# --- 3. Filter the insertions tsv by read_id (looked up by header name) ---
python3 - "$TSV" "$BLACKLIST" "$OUT" <<'PYEOF'
import sys
tsv_path, blacklist_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(blacklist_path) as f:
    blacklist = set(line.strip() for line in f if line.strip())

with open(tsv_path) as fin, open(out_path, 'w') as fout:
    header = fin.readline()
    fout.write(header)
    cols = header.rstrip('\n').split('\t')
    try:
        id_idx = cols.index('read_id')
    except ValueError:
        sys.exit(f"Error: no 'read_id' column in header of {tsv_path}")

    kept = dropped = 0
    for line in fin:
        read_id = line.rstrip('\n').split('\t')[id_idx]
        if read_id in blacklist:
            dropped += 1
            continue
        kept += 1
        fout.write(line)

print(f"{tsv_path}: kept {kept}, dropped {dropped} (donor-backbone reads) -> {out_path}")
PYEOF
