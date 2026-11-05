"""The bbmap-based insertion-calling workflow.

`process_reads` is the whole of it: trim reads at a primer, drop reads that
start with donor sequence, map with bbmap, then call insertion sites from the
BAM on both strands. It predates
`apps/ecoli_profiling/map_junction_insertions.py`, which is what new analyses
should use; this is here because notebooks still call it.

Needs `bbmap.sh` and `samtools`, whose containing folders are passed in.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from Bio import SeqIO


def trim_reads_by_primer(fastq_filepath, primer):
    total_reads = []
    
    trimmed_reads = []

    primer = primer.upper()

    primer_len = len(primer)
    for rec in SeqIO.parse(fastq_filepath, "fastq"):
        seq = str(rec.seq)
        total_reads.append(seq)
        primer_idx = seq.find(primer)
        if primer_idx != -1:
            trimmed_seq = seq[primer_idx + primer_len:]
            trimmed_qual = rec.letter_annotations["phred_quality"][primer_idx + primer_len:]
            trimmed_reads.append((rec.id, trimmed_seq, trimmed_qual))
    
    if len(trimmed_reads) > 0:
        print(f"Total reads before trimming: {len(total_reads)}")
        return trimmed_reads

    else:
        print("Could not trim reads -- double check your primer sequence and offset.")


def write_trimmed_fastq(trimmed_reads, output_path):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_path, "w") as f:
        for read_id, seq, qual in trimmed_reads:
            qual_str = ''.join([chr(q + 33) for q in qual])
            f.write(f"@{read_id}\n{seq}\n+\n{qual_str}\n")


CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')


def aligned_ref_length(cigar):
    """Number of reference bases consumed by alignment"""
    length = 0
    for n, op in CIGAR_RE.findall(cigar):
        if op in "MDN=X":
            length += int(n)
    return length


def extract_insertion_sites(samtools_bin,bam_file, isforward, tn5_offset=5):
    result = subprocess.run(
        [str(Path(samtools_bin) / "samtools"), "view", str(bam_file)],
        capture_output=True,
        text=True,
        check=True
    )

    positions = []

    for line in result.stdout.splitlines():
        if not line:
            continue

        fields = line.split("\t")
        #print(line)

        flag   = int(fields[1])
        pos1   = int(fields[3])      # 1-based leftmost position
        cigar  = fields[5]

        start0 = pos1 - 1             # convert to 0-based
        ref_len = aligned_ref_length(cigar)

        is_reverse = flag & 16

        if not is_reverse and isforward:
            # forward strand: insertion at left edge
            ins = start0 + tn5_offset
            positions.append(ins)
        elif is_reverse and not isforward:
            # reverse strand: insertion at right edge
            ins = start0 + ref_len - 1 - tn5_offset
            positions.append(ins)

        

    return positions


def process_reads(fastq, primer, first_ten_in_donor, refgenome, bbmap_bin, samtools_bin, path_to_output_dir):
    path_to_output_dir = Path(path_to_output_dir)
    path_to_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trim reads by primer
    trimmed_reads = trim_reads_by_primer(fastq, primer)

    # 2. Remove donor reads
    donor_removed_reads = [read for read in trimmed_reads if not read[1].startswith(first_ten_in_donor)]
    print(f"Total reads after primer trimming: {len(trimmed_reads)}")
    print(f"Total reads after donor removal: {len(donor_removed_reads)}")

    # 3. Write output FASTQ
    trimmed_fastq = path_to_output_dir / "trimmed.fastq"
    write_trimmed_fastq(donor_removed_reads, trimmed_fastq)

    # 4. Index + map
    bbmap_sh = str(Path(bbmap_bin) / "bbmap.sh")
    subprocess.run([bbmap_sh, f"ref={refgenome}", "-Xmx20g", "ztd=1"], check=True)

    mapped_bam = trimmed_fastq.with_suffix(".bam")
    subprocess.run([
        bbmap_sh,
        f"in={str(trimmed_fastq)}",
        f"outm={str(mapped_bam)}",
        "-Xmx20g", "ztd=1",
        "minid=0.9", "ambig=toss", "local=t", "slow=t", "tipsearch=200",
    ], check=True)

    # 5. Optional: quick sanity check that BAM has alignments
    samtools = str(Path(samtools_bin) / "samtools")
    n_lines = subprocess.check_output([samtools, "view", str(mapped_bam)]).splitlines()
    print(f"Mapped alignments: {len(n_lines)}")

    # 6. Extract insertion sites (prefer doing strand logic inside)
    # If your extract_insertion_sites still takes strand="f"/"r", keep it:
    forward_insertions = extract_insertion_sites(samtools_bin,mapped_bam,isforward=True,tn5_offset=5)
    reverse_insertions = extract_insertion_sites(samtools_bin,mapped_bam,isforward=False,tn5_offset=5)

    return [forward_insertions, reverse_insertions]
