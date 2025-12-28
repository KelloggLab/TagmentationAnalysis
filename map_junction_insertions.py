#!/usr/bin/env python3
"""
Map donor–genome junction reads to a reference genome and call insertion coordinates per read.

Pipeline:
  1) Trim donor/transposon sequence using cutadapt
  2) Map to reference using bwa mem
  3) Sort/index BAM
  4) Call insertion site per read from BAM using CIGAR-aware coordinates

Insertion coordinate definition:
  --donor-side 5p : insertion = reference 5' end of alignment
  --donor-side 3p : insertion = reference 3' end of alignment

Output TSV columns:
  read_id, ref, ins0, ins1, strand, mapq, cigar, aln_start0, aln_end0_excl

Notes:
  - ins0 is 0-based coordinate (Python/BED style)
  - ins1 is 1-based coordinate (human/SAM-style position)
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pysam


def run(cmd, *, shell=False):
    cmd_str = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
    print(f"[cmd] {cmd_str}")
    subprocess.run(cmd, check=True, shell=shell)


def ensure_bwa_index(ref_fa: Path):
    # BWA index outputs: .amb .ann .bwt .pac .sa
    needed = [ref_fa.with_suffix(ref_fa.suffix + ext) for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]]
    if all(p.exists() for p in needed):
        return
    run(["bwa", "index", str(ref_fa)])


def cutadapt_trim(
    fastq_in: Path,
    fastq_out: Path,
    donor_seq: str,
    donor_side: str,
    min_len: int,
    error_rate: float,
    cores: int,
):
    """
    Trims donor/transposon sequence from reads.
    If donor_side == '5p', donor expected near the 5' end; we remove it and keep downstream sequence.
    If donor_side == '3p', donor expected near the 3' end; we remove it and keep upstream sequence.

    We allow partial matching with error rate; adjust if your junction is strict.
    """
    # cutadapt:
    # -g = 5' adapter, -a = 3' adapter
    # --discard-untrimmed optionally could be used if you ONLY want junction reads.
    # Here we keep trimmed reads that contain donor sequence; discard those that don't match.
    if donor_side == "5p":
        adapter_arg = ["-g", donor_seq]
    else:
        adapter_arg = ["-a", donor_seq]

    cmd = [
        "cutadapt",
        *adapter_arg,
        "--discard-untrimmed",
        "-e", str(error_rate),
        "-m", str(min_len),
        "-j", str(cores),
        "-o", str(fastq_out),
        str(fastq_in),
    ]
    run(cmd)


def bwa_map(ref_fa: Path, fastq: Path, bam_out: Path, cores: int):
    """
    Map with bwa mem, then sort to BAM.
    """
    sam_tmp = bam_out.with_suffix(".sam")
    run(["bwa", "mem", "-t", str(cores), str(ref_fa), str(fastq)], shell=False)
    # The above prints to stdout; do it properly with redirection:
    run(f"bwa mem -t {cores} {ref_fa} {fastq} > {sam_tmp}", shell=True)

    # Convert + sort
    run(["samtools", "sort", "-@", str(cores), "-o", str(bam_out), str(sam_tmp)])
    run(["samtools", "index", str(bam_out)])
    try:
        sam_tmp.unlink()
    except FileNotFoundError:
        pass


def ref_aln_span(read: pysam.AlignedSegment) -> Tuple[int, int]:
    """
    Returns (start0, end0_excl) on reference for aligned portion.
    pysam gives:
      reference_start : 0-based inclusive
      reference_end   : 0-based exclusive
    """
    return read.reference_start, read.reference_end


def insertion_coord(read: pysam.AlignedSegment, donor_side: str) -> Optional[int]:
    """
    Return insertion coordinate as 0-based integer, or None if cannot be called.

    donor_side == '5p' => call reference 5' end of read
    donor_side == '3p' => call reference 3' end of read

    5' end on reference:
      - forward read: reference_start
      - reverse read: reference_end - 1

    3' end on reference:
      - forward read: reference_end - 1
      - reverse read: reference_start
    """
    if read.is_unmapped:
        return None
    if read.reference_start is None or read.reference_end is None:
        return None
    if read.cigarstring is None or read.cigarstring == "*":
        return None

    start0, end0_excl = ref_aln_span(read)
    if end0_excl <= start0:
        return None

    forward = not read.is_reverse

    if donor_side == "5p":
        # insertion adjacent to read's 5' genomic base
        return start0 if forward else (end0_excl - 1)
    else:
        # donor at 3', insertion adjacent to read's 3' genomic base
        return (end0_excl - 1) if forward else start0


def call_insertions(
    bam_path: Path,
    out_tsv: Path,
    donor_side: str,
    min_mapq: int,
    primary_only: bool,
):
    """
    Iterate BAM and write per-read insertion coordinates.
    """
    bam = pysam.AlignmentFile(str(bam_path), "rb")

    with open(out_tsv, "w") as f:
        f.write("\t".join([
            "read_id", "ref", "ins0", "ins1", "strand", "mapq", "cigar",
            "aln_start0", "aln_end0_excl"
        ]) + "\n")

        n_total = 0
        n_kept = 0

        for read in bam.fetch(until_eof=True):
            n_total += 1

            if read.is_unmapped:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if primary_only and (read.is_secondary or read.is_supplementary):
                continue

            ins0 = insertion_coord(read, donor_side=donor_side)
            if ins0 is None:
                continue

            ref = bam.get_reference_name(read.reference_id)
            strand = "-" if read.is_reverse else "+"
            start0, end0_excl = ref_aln_span(read)

            f.write("\t".join(map(str, [
                read.query_name, ref, ins0, ins0 + 1, strand, read.mapping_quality,
                read.cigarstring, start0, end0_excl
            ])) + "\n")
            n_kept += 1

    bam.close()
    print(f"[info] BAM reads scanned: {n_total}")
    print(f"[info] Insertions written: {n_kept}")
    print(f"[info] Output TSV: {out_tsv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq", required=True, type=Path, help="Merged FASTQ (single-end).")
    ap.add_argument("--ref", required=True, type=Path, help="Reference genome FASTA (E. coli).")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    ap.add_argument("--donor-seq", required=True, help="Donor/transposon sequence to trim (junction-proximal).")
    ap.add_argument("--donor-side", choices=["5p", "3p"], default="5p",
                    help="Where donor sequence is in the read before trimming: 5p or 3p.")

    ap.add_argument("--min-len", type=int, default=20, help="Minimum length of trimmed genomic read to keep.")
    ap.add_argument("--error-rate", type=float, default=0.1, help="cutadapt error rate (e.g. 0.1).")
    ap.add_argument("--cores", type=int, default=8, help="Threads for cutadapt/bwa/samtools.")

    ap.add_argument("--min-mapq", type=int, default=20, help="Minimum MAPQ to report insertion.")
    ap.add_argument("--primary-only", action="store_true", help="Drop secondary/supplementary alignments.")

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fastq = args.fastq.resolve()
    ref = args.ref.resolve()

    trimmed_fastq = args.outdir / f"{fastq.stem}.trimmed.fastq"
    bam = args.outdir / f"{fastq.stem}.mapped.sorted.bam"
    out_tsv = args.outdir / f"{fastq.stem}.insertions.{args.donor_side}.tsv"

    ensure_bwa_index(ref)

    cutadapt_trim(
        fastq_in=fastq,
        fastq_out=trimmed_fastq,
        donor_seq=args.donor_seq,
        donor_side=args.donor_side,
        min_len=args.min_len,
        error_rate=args.error_rate,
        cores=args.cores,
    )

    bwa_map(ref_fa=ref, fastq=trimmed_fastq, bam_out=bam, cores=args.cores)

    call_insertions(
        bam_path=bam,
        out_tsv=out_tsv,
        donor_side=args.donor_side,
        min_mapq=args.min_mapq,
        primary_only=args.primary_only,
    )


if __name__ == "__main__":
    main()

