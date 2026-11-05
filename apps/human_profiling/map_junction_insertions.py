#!/usr/bin/env python3
"""Map UdiTaS junction reads to the human genome and call insertion sites.

The counterpart of `apps/ecoli_profiling/map_junction_insertions.py`, but it is
a separate script rather than a flag on that one, because four things about
this assay differ and each of them changes the pipeline.

**1. The reads arrive already trimmed.** `extract_salvage.py` cuts the primer
and the transposon end off inside the CBCL reader, so what reaches this script
is pure genomic flank. There is no cutadapt step here. Handing one of those
FASTQs to the E. coli script instead would discard every read: it requires
`--donor-seq` and runs cutadapt with `--discard-untrimmed`, looking for a donor
sequence that is no longer in the read.

Because the flank starts at the junction and reads outward into the genome, the
insertion coordinate is always the read's 5' genomic base. There is no
`--donor-side` option; the trimming has already fixed it.

**2. The UMI is the point.** `extract_salvage.py` puts it in the read name as
`t<tile>_<cluster>_<umi>`. Reads sharing an insertion site *and* a UMI are PCR
duplicates of one molecule, so the number worth reporting per site is distinct
UMIs, not reads. The per-site output carries both.

**3. The reference is 3.1 Gb, not 4.6 Mb.** Three consequences:

- The BWA index is never built here. On GRCh38 that is about an hour of CPU
  and some 5 GB of output, which is not something a per-sample mapping script
  should start by surprise. A missing index is an error that names the files.
- `bwa mem` is piped straight into `samtools sort`. The E. coli script writes
  an intermediate `.sam`, which at human scale is a very large temporary file
  for no benefit.
- The genome is never read into Python. Nothing here needs reference sequence,
  so nothing here loads it. `core.sequence.load_genome` in particular must not
  be pointed at GRCh38: it reads only the first record, and it builds two
  copies of the string it does read.

**4. A 40 nt flank in a repetitive genome multimaps.** `min_flank` is 40, which
is ample in E. coli and marginal in human. So the defaults are stricter than
the E. coli script's: MAPQ 30 rather than 20, secondary and supplementary
alignments dropped rather than kept, and alternate/unplaced contigs excluded,
since a read on an alt haplotype is usually the same locus as one on the
primary chromosome. Each filter's toll is reported, because with reads this
short those counts are the result's main caveat.

usage:
    python3 map_junction_insertions.py \\
        --fastq salvage/293T_RE.fastq.gz \\
        --ref /path/to/GRCh38.primary_assembly.fa \\
        --outdir mapped/293T_RE
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pysam

#: BWA writes these alongside the FASTA. All five must exist.
BWA_INDEX_SUFFIXES = (".amb", ".ann", ".bwt", ".pac", ".sa")

#: Chromosomes kept by default, with and without the `chr` prefix, so GRCh38
#: from either Ensembl or UCSC works without a flag.
PRIMARY_CHROMS = frozenset(
    [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]
    + [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM", "chrMT"]
)

#: How far behind the current alignment start a buffered site must fall before
#: it can be flushed. An insertion coordinate for a reverse-strand read is at
#: the far end of its alignment, so it is not monotonic in the BAM's sort
#: order; this window is the slack that makes it safe. Reads are at most 301
#: nt in this assay.
FLUSH_WINDOW = 1000

PER_READ_HEADER = ["read_id", "umi", "ref", "ins0", "ins1", "strand", "mapq",
                   "cigar", "aln_start0", "aln_end0_excl"]
PER_SITE_HEADER = ["ref", "ins0", "ins1", "strand", "reads", "umis"]


def run(cmd, *, shell=False):
    cmd_str = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
    print(f"[cmd] {cmd_str}")
    subprocess.run(cmd, check=True, shell=shell)


def require_bwa_index(ref_fa: Path):
    """Fail unless the BWA index is already built.

    Deliberately does not build it. See the module docstring: indexing GRCh38
    is an hour-scale job, and a mapping script should not launch one because a
    path was mistyped.
    """
    missing = [str(ref_fa) + ext for ext in BWA_INDEX_SUFFIXES
               if not Path(str(ref_fa) + ext).exists()]
    if missing:
        raise SystemExit(
            f"error: no BWA index for {ref_fa}\n"
            f"missing: {', '.join(missing)}\n\n"
            f"Build it once, separately -- on the human genome this takes about\n"
            f"an hour and writes roughly 5 GB:\n\n"
            f"    bwa index {ref_fa}\n")


def umi_of(read_name: str) -> str:
    """UMI out of an `extract_salvage.py` read name (`t<tile>_<k>_<umi>`).

    Returns "" for a name in any other shape, so a FASTQ from somewhere else
    still maps -- it just cannot be UMI-collapsed, and `main` says so.
    """
    parts = read_name.split("_")
    return parts[-1] if len(parts) >= 3 else ""


def map_reads(ref_fa: Path, fastq: Path, bam_out: Path, cores: int):
    """`bwa mem` piped into `samtools sort`, then index.

    Piped rather than staged through a `.sam` on disk: at human scale that
    intermediate is tens of GB for a file that is read once.
    """
    cmd = (f"bwa mem -t {cores} {ref_fa} {fastq} "
           f"| samtools sort -@ {cores} -o {bam_out} -")
    run(cmd, shell=True)
    run(["samtools", "index", str(bam_out)])


def insertion_coord(read: pysam.AlignedSegment):
    """0-based insertion coordinate, or None if it cannot be called.

    The flank reads outward from the junction, so the insertion is at the
    read's 5' genomic base: `reference_start` on the forward strand,
    `reference_end - 1` on the reverse.
    """
    if read.is_unmapped:
        return None
    if read.reference_start is None or read.reference_end is None:
        return None
    if read.cigarstring is None or read.cigarstring == "*":
        return None
    if read.reference_end <= read.reference_start:
        return None
    return read.reference_start if not read.is_reverse else read.reference_end - 1


def keep_contig(name: str, primary_only: bool) -> bool:
    return (not primary_only) or (name in PRIMARY_CHROMS)


def call_insertions(bam_path: Path, per_read_tsv: Path, per_site_tsv: Path, *,
                    min_mapq: int, primary_chroms_only: bool,
                    drop_secondary: bool):
    """Write the per-read and per-site tables, and return the tallies.

    Sites are accumulated in a buffer that is flushed as the BAM's coordinate
    order moves past them, so peak memory is set by how many reads pile onto
    one locus rather than by how many loci the genome has. That matters here
    and does not in E. coli.
    """
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    stat = Counter()
    # (ref, ins0, strand) -> {umi: read count}; "" counts as its own UMI.
    buffer: dict = defaultdict(lambda: defaultdict(int))
    site_rows = 0

    per_site = open(per_site_tsv, "w")
    per_site.write("\t".join(PER_SITE_HEADER) + "\n")

    def flush(up_to=None, ref_name=None):
        nonlocal site_rows
        done = [k for k in buffer
                if up_to is None or k[0] != ref_name or k[1] < up_to]
        for key in sorted(done, key=lambda k: (k[0], k[1], k[2])):
            ref, ins0, strand = key
            umis = buffer.pop(key)
            reads = sum(umis.values())
            per_site.write("\t".join(map(str, [
                ref, ins0, ins0 + 1, strand, reads, len(umis)])) + "\n")
            site_rows += 1

    with open(per_read_tsv, "w") as f:
        f.write("\t".join(PER_READ_HEADER) + "\n")

        current_ref = None
        for read in bam.fetch(until_eof=True):
            stat["scanned"] += 1

            if read.is_unmapped:
                stat["unmapped"] += 1
                continue
            if drop_secondary and (read.is_secondary or read.is_supplementary):
                stat["secondary_or_supplementary"] += 1
                continue
            if read.mapping_quality < min_mapq:
                stat["below_min_mapq"] += 1
                continue

            ref = bam.get_reference_name(read.reference_id)
            if not keep_contig(ref, primary_chroms_only):
                stat["non_primary_contig"] += 1
                continue

            ins0 = insertion_coord(read)
            if ins0 is None:
                stat["no_coordinate"] += 1
                continue

            if ref != current_ref:
                flush()
                current_ref = ref
            elif read.reference_start > FLUSH_WINDOW:
                flush(up_to=read.reference_start - FLUSH_WINDOW, ref_name=ref)

            strand = "-" if read.is_reverse else "+"
            umi = umi_of(read.query_name)
            if not umi:
                stat["no_umi_in_name"] += 1

            f.write("\t".join(map(str, [
                read.query_name, umi, ref, ins0, ins0 + 1, strand,
                read.mapping_quality, read.cigarstring,
                read.reference_start, read.reference_end])) + "\n")
            buffer[(ref, ins0, strand)][umi] += 1
            stat["kept"] += 1

    flush()
    per_site.close()
    bam.close()
    stat["sites"] = site_rows
    return stat


def report(stat: Counter, per_read_tsv: Path, per_site_tsv: Path):
    scanned, kept = stat["scanned"], stat["kept"]

    def pct(n):
        return f"{100 * n / scanned:5.1f}%" if scanned else "    --"

    print(f"\n[info] alignments scanned: {scanned}")
    for key, label in [
        ("unmapped", "unmapped"),
        ("secondary_or_supplementary", "secondary/supplementary"),
        ("below_min_mapq", "below --min-mapq"),
        ("non_primary_contig", "on a non-primary contig"),
        ("no_coordinate", "no callable coordinate"),
    ]:
        if stat[key]:
            print(f"[info]   dropped, {label:<24} {stat[key]:>10}  {pct(stat[key])}")
    print(f"[info] reads kept: {kept}  {pct(kept)}")
    print(f"[info] insertion sites: {stat['sites']}")

    if stat["no_umi_in_name"]:
        print(f"[warn] {stat['no_umi_in_name']} of the kept reads have no UMI in "
              f"the read name, so they are not UMI-collapsed. Reads written by "
              f"extract_salvage.py are named t<tile>_<cluster>_<umi>.")
    if scanned and kept / scanned < 0.5:
        print("[warn] under half the alignments survived. With a 40 nt flank in a "
              "repetitive genome that can be normal, but check the breakdown "
              "above before reading anything into the site counts.")

    print(f"[info] per-read TSV: {per_read_tsv}")
    print(f"[info] per-site TSV: {per_site_tsv}")


def fastq_stem(path: Path) -> str:
    """`293T_RE.fastq.gz` -> `293T_RE`, rather than `293T_RE.fastq`."""
    name = path.name
    for ext in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Map pre-trimmed UdiTaS junction reads to the human genome "
                    "and call UMI-collapsed insertion sites.")
    ap.add_argument("--fastq", required=True, type=Path,
                    help="FASTQ from extract_salvage.py; .gz is fine. Already "
                         "trimmed to genomic flank -- no trimming happens here.")
    ap.add_argument("--ref", required=True, type=Path,
                    help="Human reference FASTA. Its BWA index must already "
                         "exist; this script will not build one.")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--min-mapq", type=int, default=30,
                    help="Minimum MAPQ (default 30, stricter than the E. coli "
                         "script's 20, because a 40 nt flank multimaps in a "
                         "repetitive genome).")
    ap.add_argument("--keep-secondary", action="store_true",
                    help="Keep secondary/supplementary alignments. Off by "
                         "default: one molecule would be counted repeatedly.")
    ap.add_argument("--all-contigs", action="store_true",
                    help="Keep alternate, unplaced and decoy contigs. Off by "
                         "default: an alt-contig hit is usually the same locus "
                         "as its primary chromosome.")
    ap.add_argument("--cores", type=int, default=min(8, os.cpu_count() or 1))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    fastq = args.fastq.resolve()
    ref = args.ref.resolve()
    if not fastq.exists():
        raise SystemExit(f"error: no such FASTQ: {fastq}")
    if not ref.exists():
        raise SystemExit(f"error: no such reference: {ref}")
    require_bwa_index(ref)

    stem = fastq_stem(fastq)
    bam = args.outdir / f"{stem}.sorted.bam"
    per_read_tsv = args.outdir / f"{stem}.insertions.tsv"
    per_site_tsv = args.outdir / f"{stem}.sites.tsv"

    map_reads(ref_fa=ref, fastq=fastq, bam_out=bam, cores=args.cores)
    stat = call_insertions(
        bam, per_read_tsv, per_site_tsv,
        min_mapq=args.min_mapq,
        primary_chroms_only=not args.all_contigs,
        drop_secondary=not args.keep_secondary,
    )
    report(stat, per_read_tsv, per_site_tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
