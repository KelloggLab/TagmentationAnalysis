"""End-to-end test: does the pipeline recover insertions we planted ourselves?

This is the regression anchor the architecture plan asks for. It runs the real
`apps/ecoli_profiling/map_junction_insertions.py` -- real cutadapt, real bwa, real samtools
-- over the synthetic dataset from make_fixtures.py, and checks the output TSV
against the known answer. Any refactoring that changes a called coordinate will
fail here.

Needs cutadapt, bwa and samtools on PATH, and pysam importable; the whole class
skips if they are missing. In the lab environment:

    conda activate seqsuite
    python -m unittest discover -s tests -v
"""

import csv
import json
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

REQUIRED_TOOLS = ["cutadapt", "bwa", "samtools"]
DONOR_SIDE = "5p"


def missing_requirements():
    missing = [t for t in REQUIRED_TOOLS if shutil.which(t) is None]
    try:
        import pysam  # noqa: F401
    except ImportError:
        missing.append("pysam")
    return missing


_MISSING = missing_requirements()


@unittest.skipIf(_MISSING, f"missing: {', '.join(_MISSING)}")
class InsertionPipelineTest(unittest.TestCase):
    """One pipeline run, many assertions about its output."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        work = Path(cls._tmp.name)
        cls.truth = make_fixtures.build(work)

        outdir = work / "out"
        proc = subprocess.run(
            [sys.executable, str(REPO / "apps" / "ecoli_profiling" / "map_junction_insertions.py"),
             "--fastq", str(work / cls.truth["reads_fastq"]),
             "--ref", str(work / cls.truth["genome_fasta"]),
             "--outdir", str(outdir),
             "--donor-seq", cls.truth["donor_end"],
             "--donor-side", DONOR_SIDE,
             "--primary-only",
             "--cores", "2"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"pipeline failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")

        tsv = outdir / f"toy_reads.insertions.{DONOR_SIDE}.tsv"
        assert tsv.exists(), f"no output TSV at {tsv}"
        with tsv.open() as fh:
            cls.rows = list(csv.DictReader(fh, delimiter="\t"))
        cls.outdir = outdir

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- the schema other code depends on -------------------------------

    def test_column_order_is_unchanged(self):
        """core.insertions and filter_donor_backbone.sh index by name and
        by column number, so the header is part of the contract."""
        self.assertEqual(
            list(self.rows[0].keys()),
            ["read_id", "ref", "ins0", "ins1", "strand", "mapq", "cigar",
             "aln_start0", "aln_end0_excl"],
        )

    def test_expected_output_files_exist(self):
        for name in ["toy_reads.trimmed.fastq",
                     "toy_reads.mapped.sorted.bam",
                     "toy_reads.mapped.sorted.bam.bai"]:
            self.assertTrue((self.outdir / name).exists(), f"missing {name}")

    def test_intermediate_sam_is_cleaned_up(self):
        self.assertFalse((self.outdir / "toy_reads.mapped.sorted.sam").exists())

    # -- the insertion calls themselves ---------------------------------

    def test_every_planted_read_is_called_at_its_planted_site(self):
        """The strongest assertion here: each read name carries the position and
        strand it was built from, so this checks every planted read individually."""
        checked = 0
        for row in self.rows:
            if not row["read_id"].startswith("ins_"):
                continue
            _, pos, strand, _ = row["read_id"].split("_")
            self.assertEqual(int(row["ins0"]), int(pos),
                             f"{row['read_id']} called at {row['ins0']}")
            self.assertEqual(row["strand"], strand, f"{row['read_id']} strand")
            checked += 1
        self.assertEqual(checked, self.truth["expected_insertion_rows"])

    def test_read_counts_per_site_are_exact(self):
        got = Counter((int(r["ins0"]), r["strand"])
                      for r in self.rows if r["read_id"].startswith("ins_"))
        want = Counter()
        for e in self.truth["expected_insertions"]:
            want[(e["pos"], e["strand"])] = e["n_reads"]
        self.assertEqual(dict(got), dict(want))

    def test_both_strands_report_the_same_coordinate(self):
        """A plus-strand read gives reference_start and a minus-strand read
        gives reference_end - 1. Site 12071 was planted on both, so if the two
        branches of insertion_coord() ever disagree, this catches it."""
        at_site = {r["strand"] for r in self.rows if int(r["ins0"]) == 12071}
        self.assertEqual(at_site, {"+", "-"})

    def test_ins1_is_ins0_plus_one(self):
        for row in self.rows:
            self.assertEqual(int(row["ins1"]), int(row["ins0"]) + 1)

    def test_coordinates_are_consistent_with_the_alignment_span(self):
        for row in self.rows:
            start, end = int(row["aln_start0"]), int(row["aln_end0_excl"])
            self.assertLess(start, end)
            expected = start if row["strand"] == "+" else end - 1
            self.assertEqual(int(row["ins0"]), expected, row["read_id"])

    def test_reference_name_is_reported(self):
        self.assertEqual({r["ref"] for r in self.rows},
                         {self.truth["genome_name"]})

    def test_mapq_filter_is_respected(self):
        self.assertTrue(all(int(r["mapq"]) >= 20 for r in self.rows))

    # -- the reads that should be thrown away ---------------------------

    def test_every_decoy_is_dropped(self):
        """Including the donor-free ones.

        This used to leak. `cutadapt -g SEQ` is unanchored and defaults to
        -O 3, so a read merely *beginning* with three bases resembling the end
        of the donor was "trimmed", survived --discard-untrimmed, and was
        reported as an insertion despite never containing a junction. The
        pipeline now passes -O len(donor_seq) by default, so the whole
        transposon end has to be present. If that default is ever relaxed,
        this test is what notices.
        """
        leaked = sorted(r["read_id"] for r in self.rows
                        if not r["read_id"].startswith("ins_"))
        self.assertEqual(leaked, [])

    def test_total_row_count(self):
        self.assertEqual(len(self.rows), self.truth["expected_insertion_rows"])

    def test_lowering_min_overlap_reintroduces_the_leak(self):
        """The other half of the above: it is the -O default doing the work,
        not luck. At -O 3 (cutadapt's own default) the partial match is enough
        again, and a donor-free read comes back as an insertion call."""
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "out"
            proc = subprocess.run(
                [sys.executable, str(REPO / "apps" / "ecoli_profiling" / "map_junction_insertions.py"),
                 "--fastq", str(Path(self._tmp.name) / self.truth["reads_fastq"]),
                 "--ref", str(Path(self._tmp.name) / self.truth["genome_fasta"]),
                 "--outdir", str(outdir),
                 "--donor-seq", self.truth["donor_end"],
                 "--donor-side", DONOR_SIDE, "--primary-only",
                 "--min-overlap", "3", "--cores", "2"],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            with (outdir / f"toy_reads.insertions.{DONOR_SIDE}.tsv").open() as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
        leaked = sorted(r["read_id"] for r in rows
                        if not r["read_id"].startswith("ins_"))
        # Which decoy leaks depends on the random genome, so assert the
        # behaviour rather than the name: donor-free reads get through.
        self.assertTrue(leaked, "expected at least one leak at -O 3")
        for name in leaked:
            self.assertTrue(name.startswith("decoy_no_donor"), name)


class FixtureGeneratorTest(unittest.TestCase):
    """The fixtures are only useful if they are reproducible."""

    def test_two_runs_give_identical_files(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            make_fixtures.build(a)
            make_fixtures.build(b)
            for name in ["toy_genome.fasta", "toy_reads.fastq", "truth.json"]:
                self.assertEqual((Path(a) / name).read_bytes(),
                                 (Path(b) / name).read_bytes(),
                                 f"{name} is not deterministic")

    def test_donor_end_does_not_occur_in_the_toy_genome(self):
        with tempfile.TemporaryDirectory() as d:
            truth = make_fixtures.build(d)
            genome = "".join(
                line.strip() for line in (Path(d) / truth["genome_fasta"]).read_text().splitlines()
                if not line.startswith(">"))
            self.assertNotIn(truth["donor_end"], genome)
            self.assertEqual(len(genome), truth["genome_len"])

    def test_truth_file_is_self_consistent(self):
        with tempfile.TemporaryDirectory() as d:
            make_fixtures.build(d)
            truth = json.loads((Path(d) / "truth.json").read_text())
            planted = sum(e["n_reads"] for e in truth["expected_insertions"])
            decoys = sum(e["n_reads"] for e in truth["decoys"])
            self.assertEqual(truth["expected_insertion_rows"], planted)
            self.assertEqual(truth["total_reads"], planted + decoys)


if __name__ == "__main__":
    unittest.main()
