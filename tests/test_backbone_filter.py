"""Tests for `pipeline/filter_donor_backbone.sh`.

The filter's job is to drop insertion calls whose read is really donor-plasmid
backbone, while keeping calls whose read carries Cargo sequence -- because a
genuine new integration legitimately reads from the Cargo right up to the
transposon end. That distinction is the whole point of the script, and it is
what these tests pin down.

The fixture makes this testable by splicing one stretch of donor backbone and
one stretch of Cargo into the toy genome, so reads from those regions both map
to the genome (appearing in the insertion table) and match the donor. See
make_fixtures.py.

Needs cutadapt, bwa, samtools and pysam; the classes skip without them.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_fixtures  # noqa: E402

FILTER = REPO / "pipeline" / "filter_donor_backbone.sh"
REQUIRED_TOOLS = ["cutadapt", "bwa", "samtools"]


def missing_requirements():
    missing = [t for t in REQUIRED_TOOLS if shutil.which(t) is None]
    try:
        import pysam  # noqa: F401
    except ImportError:
        missing.append("pysam")
    return missing


_MISSING = missing_requirements()


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def sites(rows):
    """Reads grouped by (position, strand), which is how the fixture plants
    them -- one site is deliberately planted on both strands, so position
    alone would conflate two entries."""
    counts = Counter()
    for r in rows:
        _, pos, strand, _ = r["read_id"].split("_")
        counts[(int(pos), strand)] += 1
    return counts


@unittest.skipIf(_MISSING, f"missing: {', '.join(_MISSING)}")
class BackboneFilterTest(unittest.TestCase):
    """The fixture and the genome mapping are built once; each test then runs
    the filter itself with whatever options it is about."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls.truth = make_fixtures.build(cls.work)

        outdir = cls.work / "out"
        proc = subprocess.run(
            [sys.executable, str(REPO / "pipeline" / "map_junction_insertions.py"),
             "--fastq", str(cls.work / cls.truth["reads_fastq"]),
             "--ref", str(cls.work / cls.truth["genome_fasta"]),
             "--outdir", str(outdir),
             "--donor-seq", cls.truth["donor_end"],
             "--donor-side", "5p", "--primary-only", "--cores", "2"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f"mapping step failed:\n{proc.stderr[-4000:]}")

        cls.trimmed = outdir / "toy_reads.trimmed.fastq"
        cls.tsv = outdir / "toy_reads.insertions.5p.tsv"
        cls.donor_gb = cls.work / cls.truth["donor_genbank"]
        cls.unfiltered = read_tsv(cls.tsv)

        # Which sites are what, taken from the fixture rather than hardcoded.
        cls.by_origin = {}
        for e in cls.truth["expected_insertions"]:
            cls.by_origin.setdefault(e["origin"], []).append(e)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_filter(self, *extra, donor=None, tsv=None, expect_ok=True):
        """Run the script into a fresh workdir and return (proc, out_path)."""
        workdir = Path(tempfile.mkdtemp(dir=self.work))
        out = workdir / "filtered.tsv"
        proc = subprocess.run(
            [str(FILTER),
             "-g", str(donor or self.donor_gb),
             "-f", str(self.trimmed),
             "-t", str(tsv or self.tsv),
             "-o", str(out),
             "-w", str(workdir), *extra],
            capture_output=True, text=True)
        if expect_ok and proc.returncode != 0:
            self.fail(f"filter failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
        return proc, out

    # -- the backbone reference it builds -------------------------------

    def test_backbone_fasta_is_everything_after_the_cargo(self):
        proc, _ = self.run_filter()
        workdir = Path(proc.args[proc.args.index("-w") + 1])
        fasta = workdir / "donor_backbone.fasta"
        self.assertTrue(fasta.exists())
        lines = fasta.read_text().splitlines()
        self.assertEqual(lines[0], ">donor_backbone")
        seq = "".join(lines[1:])
        self.assertEqual(seq, self.truth["donor"]["backbone"])
        self.assertEqual(len(seq), self.truth["donor"]["backbone_len"])

    def test_it_reports_the_cargo_boundary(self):
        proc, _ = self.run_filter()
        d = self.truth["donor"]
        self.assertIn(f"cargo=1..{d['cargo_end']}", proc.stderr)
        self.assertIn(f"({d['backbone_len']} bp)", proc.stderr)

    # -- what it keeps and what it drops --------------------------------

    def test_backbone_reads_are_dropped(self):
        _, out = self.run_filter()
        surviving = sites(read_tsv(out))
        for e in self.by_origin["backbone"]:
            self.assertEqual(surviving[(e["pos"], e["strand"])], 0,
                             f"backbone site {e['pos']} should be filtered out")

    def test_cargo_reads_are_kept(self):
        """Deliberate design decision, stated in the script's header: only the
        backbone is screened, because a real integration carries Cargo right up
        to the transposon end."""
        _, out = self.run_filter()
        surviving = sites(read_tsv(out))
        for e in self.by_origin["cargo"]:
            self.assertEqual(surviving[(e["pos"], e["strand"])], e["n_reads"],
                             f"cargo site {e['pos']} should survive")

    def test_ordinary_genomic_reads_are_untouched(self):
        _, out = self.run_filter()
        surviving = sites(read_tsv(out))
        for e in self.by_origin["genome"]:
            self.assertEqual(surviving[(e["pos"], e["strand"])], e["n_reads"], e["pos"])

    def test_drop_count_matches_the_fixture(self):
        _, out = self.run_filter()
        rows = read_tsv(out)
        self.assertEqual(
            len(rows),
            self.truth["expected_insertion_rows"] - self.truth["expected_backbone_filtered"])

    def test_it_reports_kept_and_dropped(self):
        proc, _ = self.run_filter()
        expected_dropped = self.truth["expected_backbone_filtered"]
        kept = self.truth["expected_insertion_rows"] - expected_dropped
        self.assertIn(f"kept {kept}, dropped {expected_dropped}", proc.stdout)

    # -- the -m threshold -----------------------------------------------

    def test_partial_backbone_read_is_dropped_at_the_default_threshold(self):
        """These reads carry 40 bp of backbone then ordinary genome; bwa aligns
        them as 40M30S, so 40 aligned bases clears the default -m 30."""
        _, out = self.run_filter()
        surviving = sites(read_tsv(out))
        for e in self.by_origin["backbone_partial"]:
            self.assertEqual(surviving[(e["pos"], e["strand"])], 0, e["pos"])

    def test_partial_backbone_read_survives_a_higher_threshold(self):
        _, out = self.run_filter("-m", "50")
        surviving = sites(read_tsv(out))
        for e in self.by_origin["backbone_partial"]:
            self.assertEqual(surviving[(e["pos"], e["strand"])], e["n_reads"],
                             f"{e['pos']} has only {self.truth['partial_backbone_bp']} bp "
                             "of backbone, so -m 50 should spare it")

    def test_full_backbone_read_is_dropped_even_at_a_higher_threshold(self):
        _, out = self.run_filter("-m", "50")
        surviving = sites(read_tsv(out))
        for e in self.by_origin["backbone"]:
            self.assertEqual(surviving[(e["pos"], e["strand"])], 0, e["pos"])

    def test_a_threshold_above_the_read_length_keeps_everything(self):
        _, out = self.run_filter("-m", "1000")
        self.assertEqual(len(read_tsv(out)), len(self.unfiltered))

    # -- table integrity -------------------------------------------------

    def test_header_and_columns_are_preserved(self):
        _, out = self.run_filter()
        with open(out) as fh:
            header = fh.readline().rstrip("\n").split("\t")
        self.assertEqual(header,
                         ["read_id", "ref", "ins0", "ins1", "strand", "mapq",
                          "cigar", "aln_start0", "aln_end0_excl"])

    def test_surviving_rows_are_byte_identical_to_their_input_rows(self):
        """The filter should remove whole lines and change nothing else."""
        _, out = self.run_filter()
        with open(self.tsv) as fh:
            original = {line.split("\t")[0]: line for line in fh.readlines()[1:]}
        with open(out) as fh:
            for line in fh.readlines()[1:]:
                read_id = line.split("\t")[0]
                self.assertEqual(line, original[read_id], read_id)

    def test_output_is_a_subset_of_the_input(self):
        _, out = self.run_filter()
        kept = {r["read_id"] for r in read_tsv(out)}
        self.assertTrue(kept < {r["read_id"] for r in self.unfiltered})

    # -- caching ---------------------------------------------------------

    def test_the_backbone_index_is_reused_on_a_second_run(self):
        workdir = Path(tempfile.mkdtemp(dir=self.work))
        args = [str(FILTER), "-g", str(self.donor_gb), "-f", str(self.trimmed),
                "-t", str(self.tsv), "-w", str(workdir)]
        first = subprocess.run(args + ["-o", str(workdir / "a.tsv")],
                               capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        fasta = workdir / "donor_backbone.fasta"
        stamp = fasta.stat().st_mtime_ns

        second = subprocess.run(args + ["-o", str(workdir / "b.tsv")],
                                capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        self.assertEqual(fasta.stat().st_mtime_ns, stamp,
                         "backbone fasta was rebuilt instead of reused")
        # Rebuild or not, the answer must be the same.
        self.assertEqual((workdir / "a.tsv").read_bytes(),
                         (workdir / "b.tsv").read_bytes())

    def test_a_newer_genbank_invalidates_the_cache(self):
        workdir = Path(tempfile.mkdtemp(dir=self.work))
        donor = workdir / "donor.gb"
        shutil.copy(self.donor_gb, donor)
        args = [str(FILTER), "-g", str(donor), "-f", str(self.trimmed),
                "-t", str(self.tsv), "-w", str(workdir)]
        subprocess.run(args + ["-o", str(workdir / "a.tsv")],
                       capture_output=True, text=True, check=True)
        fasta = workdir / "donor_backbone.fasta"
        stamp = fasta.stat().st_mtime_ns

        # Touch the GenBank into the future so it is newer than the cache.
        future = (fasta.stat().st_mtime_ns + 2_000_000_000)
        os.utime(donor, ns=(future, future))
        subprocess.run(args + ["-o", str(workdir / "b.tsv")],
                       capture_output=True, text=True, check=True)
        self.assertNotEqual(fasta.stat().st_mtime_ns, stamp,
                            "cache should have been rebuilt for a newer GenBank")

    # -- failure modes ---------------------------------------------------

    def test_missing_cargo_feature_is_a_clear_error(self):
        broken = self.work / "no_cargo.gb"
        broken.write_text(self.donor_gb.read_text().replace('/label="Cargo"',
                                                            '/label="Elsewhere"'))
        proc, out = self.run_filter(donor=broken, expect_ok=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Cargo", proc.stderr)
        self.assertFalse(out.exists(), "no output should be written on failure")

    def test_missing_read_id_column_is_a_clear_error(self):
        bad = self.work / "no_read_id.tsv"
        with open(self.tsv) as fh:
            lines = fh.readlines()
        lines[0] = lines[0].replace("read_id", "name")
        bad.write_text("".join(lines))
        proc, _ = self.run_filter(tsv=bad, expect_ok=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("read_id", proc.stderr)

    def test_missing_input_file_is_rejected(self):
        proc, _ = self.run_filter(donor=self.work / "nope.gb", expect_ok=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not found", proc.stderr)

    def test_missing_required_option_prints_usage(self):
        proc = subprocess.run([str(FILTER), "-g", str(self.donor_gb)],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Usage:", proc.stderr)


@unittest.skipIf(shutil.which("bash") is None, "no bash")
class ToolResolutionTest(unittest.TestCase):
    """`find_tool` used bash 4's ${name^^}, which is a syntax error on the
    bash 3.2 that macOS ships -- the error message itself used to fail. This
    keeps that fixed."""

    def test_missing_bwa_names_the_override_variable_in_upper_case(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ["donor.gb", "reads.fastq", "ins.tsv"]:
                (d / name).write_text("placeholder\n")
            proc = subprocess.run(
                [str(FILTER), "-g", str(d / "donor.gb"), "-f", str(d / "reads.fastq"),
                 "-t", str(d / "ins.tsv"), "-o", str(d / "out.tsv")],
                capture_output=True, text=True,
                env={"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not found on PATH", proc.stderr)
        self.assertIn("BWA", proc.stderr)
        self.assertNotIn("bad substitution", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
