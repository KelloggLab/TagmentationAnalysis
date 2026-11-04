"""End-to-end test: does the pipeline recover insertions we planted ourselves?

This is the regression anchor the architecture plan asks for. It runs the real
`pipeline/map_junction_insertions.py` -- real cutadapt, real bwa, real samtools
-- over the synthetic dataset from make_fixtures.py, and checks the output TSV
against the known answer. Any refactoring that changes a called coordinate will
fail here.

Needs cutadapt, bwa and samtools on PATH, and pysam importable; the whole class
skips if they are missing. In the lab environment:

    conda activate analyze_tagmentation
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
            [sys.executable, str(REPO / "pipeline" / "map_junction_insertions.py"),
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
        """TsvMapHelperFunctions and filter_donor_backbone.sh index by name and
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
        strand it was built from, so this checks all 96 reads individually."""
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

    def test_unmappable_and_short_decoys_are_absent(self):
        names = {r["read_id"] for r in self.rows}
        for kind in ["short_flank", "unmappable"]:
            leaked = sorted(n for n in names if n.startswith(f"decoy_{kind}"))
            self.assertEqual(leaked, [], f"{kind} decoys should be dropped")

    def test_one_no_donor_decoy_leaks_through(self):
        """Known characteristic, not a planted expectation.

        `cutadapt -g SEQ` is unanchored, so it will also trim a short partial
        match at the 5' end. One of the eight donor-free decoys happens to
        begin with 3 bases that look like the end of the donor sequence; it is
        trimmed, therefore not discarded by --discard-untrimmed, and goes on to
        map cleanly as ordinary genomic sequence.

        It is recorded here so the behaviour is visible rather than silent. If
        the adapter is ever anchored (`-g ^SEQ`) or --discard-untrimmed is
        replaced by a minimum-overlap rule, this test should fail and be
        updated -- that would be an improvement.
        """
        leaked = sorted(r["read_id"] for r in self.rows
                        if r["read_id"].startswith("decoy_no_donor"))
        self.assertEqual(leaked, ["decoy_no_donor_7"])
        row, = [r for r in self.rows if r["read_id"] == "decoy_no_donor_7"]
        self.assertEqual(row["cigar"], "67M")

    def test_total_row_count(self):
        self.assertEqual(len(self.rows),
                         self.truth["expected_insertion_rows"] + 1)


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
