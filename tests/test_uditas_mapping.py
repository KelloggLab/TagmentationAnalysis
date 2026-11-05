"""Tests for `apps/human_profiling/map_junction_insertions.py`.

The UdiTaS counterpart of the E. coli mapper. It needs its own tests because
the things that make it different from the E. coli script are exactly the
things that can be silently wrong: which base of the alignment is the
insertion, whether PCR duplicates are collapsed by UMI, and which alignments
are thrown away before counting.

No bwa and no human genome here. `call_insertions` reads a BAM, so these build
BAMs with pysam directly -- a few reads at known coordinates on a header that
names a primary chromosome, an alt contig and a short contig. That covers the
coordinate arithmetic, the UMI collapsing, every filter and the flush window,
which is all of the logic this script actually owns.
"""

import contextlib
import gzip
import io
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HP = REPO / "apps" / "human_profiling"
sys.path.insert(0, str(HP))
sys.path.insert(1, str(REPO))

try:
    import pysam
    import map_junction_insertions as mji
    IMPORT_ERROR = None
except Exception as exc:                      # pragma: no cover
    IMPORT_ERROR = str(exc)

HEADER = {
    "HD": {"VN": "1.6", "SO": "coordinate"},
    "SQ": [
        {"SN": "chr1", "LN": 248_956_422},
        {"SN": "chr2", "LN": 242_193_529},
        {"SN": "chr1_KI270706v1_alt", "LN": 175_055},
    ],
}
REF_ID = {"chr1": 0, "chr2": 1, "chr1_KI270706v1_alt": 2}


def seg(header, *, name, ref="chr1", pos=1000, length=40, reverse=False,
        mapq=60, unmapped=False, secondary=False, supplementary=False):
    """One alignment. `length` is both the query length and the aligned span,
    so `reference_end` is `pos + length`."""
    a = pysam.AlignedSegment(header)
    a.query_name = name
    a.query_sequence = "A" * length
    a.query_qualities = pysam.qualitystring_to_array("I" * length)
    a.flag = 0
    a.is_unmapped = unmapped
    if not unmapped:
        a.reference_id = REF_ID[ref]
        a.reference_start = pos
        a.cigartuples = [(0, length)]
        a.mapping_quality = mapq
        a.is_reverse = reverse
        a.is_secondary = secondary
        a.is_supplementary = supplementary
    return a


@unittest.skipIf(IMPORT_ERROR, f"cannot import: {IMPORT_ERROR}")
class MapperTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.header = pysam.AlignmentHeader.from_dict(HEADER)

    def write_bam(self, reads):
        path = self.tmp / "in.bam"
        with pysam.AlignmentFile(str(path), "wb", header=self.header) as out:
            for r in reads:
                out.write(r)
        return path

    def call(self, reads, **kw):
        """Run call_insertions over `reads`; return (stat, per_read, sites)."""
        bam = self.write_bam(reads)
        per_read = self.tmp / "reads.tsv"
        per_site = self.tmp / "sites.tsv"
        opts = dict(min_mapq=30, primary_chroms_only=True, drop_secondary=True)
        opts.update(kw)
        stat = mji.call_insertions(bam, per_read, per_site, **opts)
        return stat, self.rows(per_read), self.rows(per_site)

    @staticmethod
    def rows(tsv):
        lines = tsv.read_text().splitlines()
        head = lines[0].split("\t")
        return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


class CoordinateTest(MapperTest):
    """The insertion is the read's 5' genomic base, because the flank reads
    outward from the junction."""

    def test_forward_read_uses_alignment_start(self):
        _, reads, _ = self.call([seg(self.header, name="t0_1_AAAAAAAAAA",
                                     pos=1000, length=40)])
        self.assertEqual(reads[0]["ins0"], "1000")
        self.assertEqual(reads[0]["strand"], "+")

    def test_reverse_read_uses_alignment_end_minus_one(self):
        _, reads, _ = self.call([seg(self.header, name="t0_1_AAAAAAAAAA",
                                     pos=1000, length=40, reverse=True)])
        self.assertEqual(reads[0]["ins0"], "1039")
        self.assertEqual(reads[0]["strand"], "-")

    def test_ins1_is_ins0_plus_one(self):
        _, reads, _ = self.call([seg(self.header, name="t0_1_AAAAAAAAAA")])
        self.assertEqual(int(reads[0]["ins1"]), int(reads[0]["ins0"]) + 1)

    def test_the_two_strands_at_one_coordinate_are_separate_sites(self):
        """A read whose alignment ends where another's begins is not the same
        insertion; strand is part of the site identity."""
        _, _, sites = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", pos=1000, length=40),
            seg(self.header, name="t0_2_CCCCCCCCCC", pos=961, length=40,
                reverse=True),
        ])
        self.assertEqual(len(sites), 2)
        self.assertEqual({s["strand"] for s in sites}, {"+", "-"})
        self.assertEqual({s["ins0"] for s in sites}, {"1000"})

    def test_unmapped_read_has_no_coordinate(self):
        stat, reads, _ = self.call([seg(self.header, name="t0_1_AAAAAAAAAA",
                                        unmapped=True)])
        self.assertEqual(stat["unmapped"], 1)
        self.assertEqual(reads, [])


class UmiCollapseTest(MapperTest):
    def test_duplicate_umis_collapse_but_reads_still_count(self):
        reads = [
            seg(self.header, name="t0_1_AAAAAAAAAA", pos=1000),
            seg(self.header, name="t0_2_AAAAAAAAAA", pos=1000),
            seg(self.header, name="t0_3_CCCCCCCCCC", pos=1000),
        ]
        _, per_read, sites = self.call(reads)
        self.assertEqual(len(per_read), 3)
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["reads"], "3")
        self.assertEqual(sites[0]["umis"], "2")

    def test_umi_is_pulled_from_the_read_name(self):
        _, per_read, _ = self.call([seg(self.header, name="t3_17_ACGTACGTAC")])
        self.assertEqual(per_read[0]["umi"], "ACGTACGTAC")

    def test_same_umi_at_different_sites_stays_two_sites(self):
        """One UMI can legitimately appear at two loci; it only merges reads
        within a site."""
        _, _, sites = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", pos=1000),
            seg(self.header, name="t0_2_AAAAAAAAAA", pos=50_000),
        ])
        self.assertEqual(len(sites), 2)
        self.assertEqual([s["umis"] for s in sites], ["1", "1"])

    def test_a_name_with_no_umi_is_reported_not_dropped(self):
        stat, per_read, sites = self.call([
            seg(self.header, name="SRR123.1", pos=1000)])
        self.assertEqual(stat["no_umi_in_name"], 1)
        self.assertEqual(stat["kept"], 1)
        self.assertEqual(per_read[0]["umi"], "")
        self.assertEqual(sites[0]["umis"], "1")

    def test_umi_of_shapes(self):
        self.assertEqual(mji.umi_of("t0_12_ACGTACGTAC"), "ACGTACGTAC")
        self.assertEqual(mji.umi_of("SRR1.1"), "")
        self.assertEqual(mji.umi_of("noUnderscore"), "")
        self.assertEqual(mji.umi_of("a_b_c_d"), "d")


class FilterTest(MapperTest):
    def test_below_min_mapq_is_dropped(self):
        stat, reads, _ = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", mapq=29)], min_mapq=30)
        self.assertEqual(stat["below_min_mapq"], 1)
        self.assertEqual(reads, [])

    def test_min_mapq_boundary_is_inclusive(self):
        stat, reads, _ = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", mapq=30)], min_mapq=30)
        self.assertEqual(stat["kept"], 1)
        self.assertEqual(len(reads), 1)

    def test_secondary_dropped_by_default(self):
        stat, _, _ = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", secondary=True)])
        self.assertEqual(stat["secondary_or_supplementary"], 1)

    def test_supplementary_dropped_by_default(self):
        stat, _, _ = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", supplementary=True)])
        self.assertEqual(stat["secondary_or_supplementary"], 1)

    def test_secondary_kept_when_asked(self):
        stat, _, _ = self.call(
            [seg(self.header, name="t0_1_AAAAAAAAAA", secondary=True)],
            drop_secondary=False)
        self.assertEqual(stat["kept"], 1)

    def test_alt_contig_dropped_by_default(self):
        stat, _, _ = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA",
                ref="chr1_KI270706v1_alt", pos=100)])
        self.assertEqual(stat["non_primary_contig"], 1)

    def test_alt_contig_kept_when_asked(self):
        stat, _, _ = self.call(
            [seg(self.header, name="t0_1_AAAAAAAAAA",
                 ref="chr1_KI270706v1_alt", pos=100)],
            primary_chroms_only=False)
        self.assertEqual(stat["kept"], 1)

    def test_both_chromosome_naming_conventions_are_primary(self):
        """GRCh38 from UCSC says chr1, from Ensembl says 1. Both count."""
        for name in ["chr1", "1", "chrX", "X", "chrM", "MT"]:
            self.assertTrue(mji.keep_contig(name, True), name)

    def test_alt_and_decoy_names_are_not_primary(self):
        for name in ["chr1_KI270706v1_alt", "chrUn_GL000195v1",
                     "chrEBV", "HLA-A*01:01:01:01", "11_KI270721v1_random"]:
            self.assertFalse(mji.keep_contig(name, True), name)

    def test_nothing_is_primary_filtered_when_off(self):
        self.assertTrue(mji.keep_contig("chrEBV", False))

    def test_each_filter_is_counted_separately(self):
        stat, _, _ = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", unmapped=True),
            seg(self.header, name="t0_2_AAAAAAAAAA", mapq=1),
            seg(self.header, name="t0_3_AAAAAAAAAA", secondary=True),
            seg(self.header, name="t0_4_AAAAAAAAAA",
                ref="chr1_KI270706v1_alt", pos=10),
            seg(self.header, name="t0_5_AAAAAAAAAA", pos=1000),
        ])
        self.assertEqual(stat["scanned"], 5)
        self.assertEqual(stat["unmapped"], 1)
        self.assertEqual(stat["below_min_mapq"], 1)
        self.assertEqual(stat["secondary_or_supplementary"], 1)
        self.assertEqual(stat["non_primary_contig"], 1)
        self.assertEqual(stat["kept"], 1)


class FlushWindowTest(MapperTest):
    """Sites are buffered and flushed as the BAM's coordinate order moves past
    them, so memory scales with pile-up depth rather than with the number of
    loci in a 3.1 Gb genome. The risk in that trade is emitting one site as two
    rows, or losing one."""

    def test_distant_sites_on_one_contig_all_appear_once(self):
        reads = [seg(self.header, name=f"t0_{i}_AAAAAAAAAA", pos=p)
                 for i, p in enumerate([1000, 500_000, 90_000_000])]
        _, _, sites = self.call(reads)
        self.assertEqual([s["ins0"] for s in sites],
                         ["1000", "500000", "90000000"])
        self.assertEqual(len(sites), 3)

    def test_a_site_is_never_split_across_two_rows(self):
        """Reads for one site separated by many other reads must still merge.
        They cannot be separated by more than the flush window in a sorted BAM,
        which is the invariant that makes the window safe."""
        reads = [seg(self.header, name="t0_1_AAAAAAAAAA", pos=1000)]
        reads += [seg(self.header, name=f"t0_{i}_CCCCCCCCCC", pos=1000)
                  for i in range(2, 60)]
        _, _, sites = self.call(reads)
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["reads"], "59")
        self.assertEqual(sites[0]["umis"], "2")

    def test_contig_change_flushes_everything(self):
        _, _, sites = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", ref="chr1", pos=1000),
            seg(self.header, name="t0_2_AAAAAAAAAA", ref="chr2", pos=20),
        ])
        self.assertEqual([(s["ref"], s["ins0"]) for s in sites],
                         [("chr1", "1000"), ("chr2", "20")])

    def test_a_site_near_coordinate_zero_survives(self):
        """reference_start below the flush window must not skip the flush and
        lose the site at the end."""
        _, _, sites = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", pos=5, length=40)])
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["ins0"], "5")

    def test_a_reverse_read_site_ahead_of_the_sort_order_is_kept(self):
        """A reverse read's insertion is at the far end of its alignment, so
        it sits ahead of the position the BAM is sorted on."""
        _, _, sites = self.call([
            seg(self.header, name="t0_1_AAAAAAAAAA", pos=1000, length=300,
                reverse=True),
            seg(self.header, name="t0_2_CCCCCCCCCC", pos=1100, length=40),
        ])
        self.assertEqual(sorted(s["ins0"] for s in sites), ["1100", "1299"])

    def test_site_count_matches_the_rows_written(self):
        reads = [seg(self.header, name=f"t0_{i}_AAAAAAAAAA", pos=1000 * i)
                 for i in range(1, 40)]
        stat, _, sites = self.call(reads)
        self.assertEqual(stat["sites"], len(sites))
        self.assertEqual(stat["sites"], 39)


class OutputShapeTest(MapperTest):
    def test_per_read_columns(self):
        _, reads, _ = self.call([seg(self.header, name="t0_1_AAAAAAAAAA")])
        self.assertEqual(list(reads[0]), mji.PER_READ_HEADER)

    def test_per_site_columns(self):
        _, _, sites = self.call([seg(self.header, name="t0_1_AAAAAAAAAA")])
        self.assertEqual(list(sites[0]), mji.PER_SITE_HEADER)

    def test_per_read_keeps_the_alignment_span_and_cigar(self):
        _, reads, _ = self.call([seg(self.header, name="t0_1_AAAAAAAAAA",
                                     pos=1000, length=40)])
        self.assertEqual(reads[0]["aln_start0"], "1000")
        self.assertEqual(reads[0]["aln_end0_excl"], "1040")
        self.assertEqual(reads[0]["cigar"], "40M")

    def test_sites_are_sorted_within_a_contig(self):
        reads = [seg(self.header, name=f"t0_{i}_AAAAAAAAAA", pos=p)
                 for i, p in enumerate([1000, 1500, 2000])]
        _, _, sites = self.call(reads)
        self.assertEqual([int(s["ins0"]) for s in sites], [1000, 1500, 2000])

    def test_empty_bam_writes_headers_only(self):
        stat, reads, sites = self.call([])
        self.assertEqual(stat["scanned"], 0)
        self.assertEqual(reads, [])
        self.assertEqual(sites, [])


class IndexRequirementTest(MapperTest):
    """The script must never start a bwa index on the human genome by
    surprise: that is an hour of CPU and about 5 GB."""

    def test_missing_index_is_an_error_that_names_the_files(self):
        ref = self.tmp / "GRCh38.fa"
        ref.write_text(">chr1\nACGT\n")
        with self.assertRaises(SystemExit) as cm:
            mji.require_bwa_index(ref)
        msg = str(cm.exception)
        for ext in mji.BWA_INDEX_SUFFIXES:
            self.assertIn(ext, msg)
        self.assertIn("bwa index", msg)

    def test_a_complete_index_passes(self):
        ref = self.tmp / "GRCh38.fa"
        ref.write_text(">chr1\nACGT\n")
        for ext in mji.BWA_INDEX_SUFFIXES:
            (self.tmp / f"GRCh38.fa{ext}").write_text("x")
        self.assertIsNone(mji.require_bwa_index(ref))

    def test_a_partial_index_is_still_an_error(self):
        ref = self.tmp / "GRCh38.fa"
        ref.write_text(">chr1\nACGT\n")
        (self.tmp / "GRCh38.fa.bwt").write_text("x")
        with self.assertRaises(SystemExit) as cm:
            mji.require_bwa_index(ref)
        self.assertNotIn(".bwt,", str(cm.exception))
        self.assertIn(".sa", str(cm.exception))


@unittest.skipIf(IMPORT_ERROR, f"cannot import: {IMPORT_ERROR}")
class CommandLineTest(unittest.TestCase):
    def test_defaults_are_stricter_than_the_ecoli_script(self):
        args = mji.parse_args(["--fastq", "a.fastq.gz", "--ref", "r.fa",
                               "--outdir", "o"])
        self.assertEqual(args.min_mapq, 30)
        self.assertFalse(args.keep_secondary)
        self.assertFalse(args.all_contigs)

    def test_there_is_no_donor_side_or_donor_seq_option(self):
        """Both are meaningless here: the reads arrive already trimmed, so the
        insertion is always the read's 5' genomic base."""
        for flag, value in [("--donor-seq", "ACGT"), ("--donor-side", "5p")]:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    mji.parse_args(["--fastq", "a", "--ref", "r",
                                    "--outdir", "o", flag, value])

    def test_missing_fastq_is_an_error_before_anything_runs(self):
        with self.assertRaises(SystemExit) as cm:
            mji.main(["--fastq", "/nonexistent/x.fastq.gz",
                      "--ref", "/nonexistent/r.fa", "--outdir", "/tmp/out"])
        self.assertIn("no such FASTQ", str(cm.exception))

    def test_fastq_stem_strips_the_whole_double_extension(self):
        self.assertEqual(mji.fastq_stem(Path("293T_RE.fastq.gz")), "293T_RE")
        self.assertEqual(mji.fastq_stem(Path("293T_RE.fq.gz")), "293T_RE")
        self.assertEqual(mji.fastq_stem(Path("293T_RE.fastq")), "293T_RE")
        self.assertEqual(mji.fastq_stem(Path("/a/b/FT_LE.fq")), "FT_LE")

    def test_fastq_stem_leaves_an_unknown_extension_to_pathlib(self):
        self.assertEqual(mji.fastq_stem(Path("reads.txt")), "reads")


@unittest.skipIf(IMPORT_ERROR, f"cannot import: {IMPORT_ERROR}")
class GenomeScaleTest(unittest.TestCase):
    """The human genome is 3.1 Gb, which rules out three things the E. coli
    script does. These assert the behaviour, not the source text: an earlier
    version of this class grepped the file and matched its own docstring."""

    @classmethod
    def setUpClass(cls):
        import ast
        tree = ast.parse((HP / "map_junction_insertions.py").read_text())
        cls.imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                cls.imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                cls.imported.add(node.module.split(".")[0])

    def test_imports_nothing_that_would_load_the_genome(self):
        """No Biopython and no `core.sequence`: the first reads a FASTA into
        memory, the second returns only its first record and builds two copies
        of it on the way."""
        self.assertNotIn("Bio", self.imported)
        self.assertNotIn("core", self.imported)

    def commands_from(self, fn, *a, **kw):
        """Run `fn` with `mji.run` replaced, returning the commands issued."""
        seen = []
        original = mji.run
        mji.run = lambda cmd, **k: seen.append(cmd)
        try:
            fn(*a, **kw)
        finally:
            mji.run = original
        return [c if isinstance(c, str) else " ".join(map(str, c))
                for c in seen]

    def test_checking_the_index_never_runs_bwa(self):
        """A missing index must be an error, not an hour-long indexing job."""
        with tempfile.TemporaryDirectory() as d:
            ref = Path(d) / "GRCh38.fa"
            ref.write_text(">chr1\nACGT\n")
            with self.assertRaises(SystemExit):
                self.commands_from(mji.require_bwa_index, ref)

    def test_mapping_never_issues_a_bwa_index(self):
        cmds = self.commands_from(mji.map_reads, Path("r.fa"),
                                  Path("in.fastq.gz"), Path("out.bam"), 4)
        self.assertFalse([c for c in cmds if "bwa index" in c], cmds)

    def test_bwa_output_is_piped_into_sort_not_staged_as_sam(self):
        """A human-scale intermediate .sam is tens of GB for a file read once."""
        cmds = self.commands_from(mji.map_reads, Path("r.fa"),
                                  Path("in.fastq.gz"), Path("out.bam"), 4)
        mem = next(c for c in cmds if "bwa mem" in c)
        self.assertIn("| samtools sort", mem)
        self.assertNotIn(".sam", mem)
        self.assertNotIn(">", mem)

    def test_mapping_indexes_the_sorted_bam(self):
        cmds = self.commands_from(mji.map_reads, Path("r.fa"),
                                  Path("in.fastq.gz"), Path("out.bam"), 4)
        self.assertTrue(any("samtools index out.bam" in c for c in cmds), cmds)

    def test_cores_reach_both_tools(self):
        cmds = self.commands_from(mji.map_reads, Path("r.fa"),
                                  Path("in.fastq.gz"), Path("out.bam"), 7)
        mem = next(c for c in cmds if "bwa mem" in c)
        self.assertIn("-t 7", mem)
        self.assertIn("-@ 7", mem)


HAVE_TOOLS = all(shutil.which(t) for t in ("bwa", "samtools"))


@unittest.skipIf(IMPORT_ERROR, f"cannot import: {IMPORT_ERROR}")
@unittest.skipUnless(HAVE_TOOLS, "needs bwa and samtools on PATH")
class EndToEndTest(unittest.TestCase):
    """The whole script against a toy genome with planted insertions.

    The other classes feed `call_insertions` a BAM they built themselves, which
    leaves `map_reads` and the wiring in `main` untested. Indexing a 50 kb
    reference takes well under a second, so the real thing can run here -- and
    this is the test that would catch the insertion coordinate being off by the
    read length, which a hand-built BAM cannot, since it asserts against the
    position the read was cut from.
    """

    GENOME_LEN = 50_000
    FLANK = 60
    SITES = [5_000, 12_345, 30_000]

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        rng = random.Random(11)
        cls.genome = "".join(rng.choice("ACGT") for _ in range(cls.GENOME_LEN))

        cls.ref = tmp / "toy.fa"
        with open(cls.ref, "w") as f:
            f.write(">chr1\n")
            for i in range(0, cls.GENOME_LEN, 70):
                f.write(cls.genome[i:i + 70] + "\n")
        subprocess.run(["bwa", "index", str(cls.ref)], check=True,
                       capture_output=True)

        # Two reads per site sharing a UMI, one with a second UMI: 3 reads,
        # 2 molecules. Forward reads start at the insertion and read outward.
        cls.fastq = tmp / "toy_RE.fastq.gz"
        with gzip.open(cls.fastq, "wt") as f:
            for n, pos in enumerate(cls.SITES):
                flank = cls.genome[pos:pos + cls.FLANK]
                for k, umi in enumerate(["AAAAAAAAAA", "AAAAAAAAAA",
                                         "CCCCCCCCCC"]):
                    f.write(f"@t0_{n}{k}_{umi}\n{flank}\n+\n"
                            f"{'I' * cls.FLANK}\n")

        cls.outdir = tmp / "out"
        mji.main(["--fastq", str(cls.fastq), "--ref", str(cls.ref),
                  "--outdir", str(cls.outdir), "--cores", "1"])
        cls.sites = MapperTest.rows(cls.outdir / "toy_RE.sites.tsv")
        cls.reads = MapperTest.rows(cls.outdir / "toy_RE.insertions.tsv")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_planted_site_is_recovered_at_its_exact_position(self):
        self.assertEqual(sorted(int(s["ins0"]) for s in self.sites),
                         self.SITES)

    def test_each_site_reports_three_reads_and_two_umis(self):
        for s in self.sites:
            self.assertEqual(s["reads"], "3", s)
            self.assertEqual(s["umis"], "2", s)

    def test_no_extra_sites(self):
        self.assertEqual(len(self.sites), len(self.SITES))

    def test_all_reads_are_on_the_forward_strand(self):
        """They were cut from the plus strand, so a minus call would mean the
        coordinate logic picked the wrong end."""
        self.assertEqual({r["strand"] for r in self.reads}, {"+"})

    def test_per_read_rows_match_the_reads_in(self):
        self.assertEqual(len(self.reads), 3 * len(self.SITES))

    def test_the_bam_and_its_index_are_kept(self):
        self.assertTrue((self.outdir / "toy_RE.sorted.bam").exists())
        self.assertTrue((self.outdir / "toy_RE.sorted.bam.bai").exists())

    def test_reverse_complement_reads_land_on_the_minus_strand_at_the_site(self):
        """A flank read out of the other orientation must call the same
        insertion base, from `reference_end - 1`."""
        rc = {"A": "T", "C": "G", "G": "C", "T": "A"}
        tmp = Path(self._tmp.name)
        fq = tmp / "rev.fastq.gz"
        pos = self.SITES[1]
        flank = self.genome[pos - self.FLANK + 1:pos + 1]
        rev = "".join(rc[b] for b in reversed(flank))
        with gzip.open(fq, "wt") as f:
            f.write(f"@t0_1_GGGGGGGGGG\n{rev}\n+\n{'I' * len(rev)}\n")
        out = tmp / "rev_out"
        mji.main(["--fastq", str(fq), "--ref", str(self.ref),
                  "--outdir", str(out), "--cores", "1"])
        sites = MapperTest.rows(out / "rev.sites.tsv")
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["strand"], "-")
        self.assertEqual(int(sites[0]["ins0"]), pos)


if __name__ == "__main__":
    unittest.main()
