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
#   ./filter_donor_backbone.sh <donor.gb> <sample_prefix> <mapped_reads_dir>
#
# Example:
#   ./filter_donor_backbone.sh ~/Downloads/pmcpdonor_kan_nr_noamp_orit.gb \
#       29_S29_L001 mapped_reads
#
# For each of {R1_001,R2_001,merged} under <mapped_reads_dir> matching
# <sample_prefix>, produces <sample_prefix>_<x>.insertions.5p.nobackbone.tsv
set -euo pipefail

BWA=${BWA:-/Users/ekello73/miniconda3/envs/analyze_tagmentation/bin/bwa}
SAMTOOLS=${SAMTOOLS:-/Users/ekello73/src/samtools-1.22.1/samtools}
MIN_MATCH=${MIN_MATCH:-30}   # minimum total aligned (M) bp to count as a backbone hit

usage() { echo "Usage: $0 <donor.gb> <sample_prefix> <mapped_reads_dir>" >&2; exit 1; }
[ $# -eq 3 ] || usage

DONOR_GB="$1"
PREFIX="$2"
DIR="$3"
[ -f "$DONOR_GB" ] || { echo "Error: $DONOR_GB not found" >&2; exit 1; }

WORKDIR="${DIR}/donor_ref"
mkdir -p "$WORKDIR"

# --- 1. Extract the donor backbone (everything outside the annotated Cargo) ---
python3 - "$DONOR_GB" "$WORKDIR/donor_backbone.fasta" <<'PYEOF'
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

$BWA index "$WORKDIR/donor_backbone.fasta" 2>/dev/null

# --- 2. For each read type, align trimmed reads to the backbone and filter ---
for TYPE in R1_001 R2_001 merged; do
    TRIMMED="${DIR}/${PREFIX}_${TYPE}.trimmed.fastq"
    TSV="${DIR}/${PREFIX}_${TYPE}.insertions.5p.tsv"
    [ -f "$TRIMMED" ] && [ -f "$TSV" ] || { echo "skip ${TYPE}: missing trimmed fastq or tsv"; continue; }

    BLACKLIST="${WORKDIR}/${PREFIX}_${TYPE}.backbone_read_ids.txt"
    $BWA mem -t 4 "$WORKDIR/donor_backbone.fasta" "$TRIMMED" 2>/dev/null \
        | $SAMTOOLS view -F 4 - \
        | awk -v minlen="$MIN_MATCH" '{
              cigar=$6; matched=0;
              while (match(cigar, /[0-9]+M/)) {
                  matched += substr(cigar,RSTART,RLENGTH-1) + 0;
                  cigar = substr(cigar, RSTART+RLENGTH);
              }
              if (matched >= minlen) print $1;
          }' | sort -u > "$BLACKLIST"

    OUT="${DIR}/${PREFIX}_${TYPE}.insertions.5p.nobackbone.tsv"
    python3 - "$TSV" "$BLACKLIST" "$OUT" <<'PYEOF'
import sys
tsv_path, blacklist_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(blacklist_path) as f:
    blacklist = set(line.strip() for line in f if line.strip())

kept = dropped = 0
with open(tsv_path) as fin, open(out_path, 'w') as fout:
    header = fin.readline()
    fout.write(header)
    for line in fin:
        read_id = line.split('\t', 1)[0]
        if read_id in blacklist:
            dropped += 1
            continue
        kept += 1
        fout.write(line)
print(f"{tsv_path}: kept {kept}, dropped {dropped} (donor-backbone reads)")
PYEOF
done
