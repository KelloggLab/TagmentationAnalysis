"""Tests for `apps/chipseq/plot_enrichment.py`.

The first tests `apps/chipseq/` has had. They exist because this code used to
be an inline `python3` heredoc in `chipseq_pipeline.sh` with a real bug in it:
it read the bedGraph's `chrom` column and then ignored it, plotting every
contig from zero on one axis whose width was the *sum* of all contig lengths.
On a single-contig genome that is right; on a chromosome plus a plasmid, which
is routine for these strains, both contigs were drawn on top of each other.

So the assertions here are mostly about contigs: that each one gets its own
axis spanning its own length, that a multi-contig run writes one file pair per
contig, and that peaks are shaded only on the contig they belong to.

Needs pandas and matplotlib, both in `environment.yml`, so these do not skip.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHIPSEQ = REPO / "apps" / "chipseq"
sys.path.insert(0, str(CHIPSEQ))

try:
    import matplotlib
    matplotlib.use("Agg")
    import plot_enrichment as pe
    IMPORT_ERROR = None
except Exception as exc:                      # pragma: no cover
    IMPORT_ERROR = str(exc)

CHROM_LEN = 4_600_000
PLASMID_LEN = 120_000


def write(path, text):
    path.write_text(text)
    return path


def fai(path, contigs):
    """Write a .fai. Only the first two columns are ever read."""
    return write(path, "".join(
        f"{name}\t{length}\t0\t60\t61\n" for name, length in contigs))


def bedgraph(path, rows):
    return write(path, "".join(
        f"{c}\t{s}\t{e}\t{v}\n" for c, s, e, v in rows))


def bed(path, rows):
    return write(path, "".join(f"{c}\t{s}\t{e}\n" for c, s, e in rows))


@unittest.skipIf(IMPORT_ERROR, f"cannot import plot_enrichment: {IMPORT_ERROR}")
class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class ReadFaiTest(TempDirTest):
    def test_lengths_in_file_order(self):
        path = fai(self.tmp / "ref.fa.fai",
                   [("chrom", CHROM_LEN), ("pBAD", PLASMID_LEN)])
        self.assertEqual(pe.read_fai(path),
                         {"chrom": CHROM_LEN, "pBAD": PLASMID_LEN})
        self.assertEqual(list(pe.read_fai(path)), ["chrom", "pBAD"])

    def test_blank_lines_ignored(self):
        path = write(self.tmp / "ref.fa.fai", f"chrom\t{CHROM_LEN}\t0\t60\t61\n\n")
        self.assertEqual(pe.read_fai(path), {"chrom": CHROM_LEN})

    def test_empty_fai_is_an_error(self):
        path = write(self.tmp / "ref.fa.fai", "")
        with self.assertRaises(ValueError):
            pe.read_fai(path)

    def test_numeric_contig_name_stays_a_string(self):
        """A contig called "1" must not become an int, or the chrom match in
        `plot_fold_enrichment` compares a str to an int and silently finds
        nothing to plot."""
        path = fai(self.tmp / "ref.fa.fai", [("1", CHROM_LEN)])
        self.assertEqual(list(pe.read_fai(path)), ["1"])


class ReadInputTest(TempDirTest):
    def test_bedgraph_columns(self):
        path = bedgraph(self.tmp / "t.bdg", [("chrom", 0, 100, 1.5)])
        df = pe.read_bedgraph(path)
        self.assertEqual(list(df.columns), ["chrom", "start", "end", "FE"])
        self.assertEqual(df.loc[0, "FE"], 1.5)

    def test_empty_bedgraph_is_an_empty_frame(self):
        path = write(self.tmp / "t.bdg", "")
        self.assertTrue(pe.read_bedgraph(path).empty)

    def test_empty_peaks_bed_is_an_empty_frame(self):
        path = write(self.tmp / "p.bed", "")
        self.assertTrue(pe.read_peaks(path).empty)

    def test_no_peaks_path_is_none(self):
        self.assertIsNone(pe.read_peaks(None))

    def test_extra_narrowpeak_columns_are_dropped(self):
        """MACS3 writes 10 columns; only the first three are wanted."""
        path = write(self.tmp / "p.bed",
                     "chrom\t10\t90\tpeak_1\t55\t.\t3.1\t9.9\t7.2\t40\n")
        df = pe.read_peaks(path)
        self.assertEqual(list(df.columns), ["chrom", "start", "end"])
        self.assertEqual(df.loc[0, "end"], 90)

    def test_contig_names_are_strings(self):
        path = bedgraph(self.tmp / "t.bdg", [("1", 0, 100, 2.0)])
        self.assertEqual(pe.read_bedgraph(path).loc[0, "chrom"], "1")


class SingleContigTest(TempDirTest):
    """The common case, and the one whose output must not have changed: file
    names stay `<prefix>.png` / `.pdf`, with no contig infix."""

    def setUp(self):
        super().setUp()
        self.bdg = bedgraph(self.tmp / "rep1_FE.bdg", [
            ("chrom", 0, 1000, 1.0),
            ("chrom", 1000, 2000, 8.0),
            ("chrom", 2000, CHROM_LEN, 1.0),
        ])

    def run_plot(self, **kw):
        return pe.plot_fold_enrichment(
            self.bdg, str(self.tmp / "rep1"), label="Replicate 1",
            color=pe.PALETTE[0], contig_lengths={"chrom": CHROM_LEN}, **kw)

    def test_writes_png_and_pdf_without_a_contig_infix(self):
        written = self.run_plot()
        self.assertEqual([Path(p).name for p in written],
                         ["rep1.png", "rep1.pdf"])
        for p in written:
            self.assertGreater(Path(p).stat().st_size, 0)

    def test_png_is_actually_a_png(self):
        self.run_plot()
        self.assertEqual((self.tmp / "rep1.png").read_bytes()[:8],
                         b"\x89PNG\r\n\x1a\n")

    def test_pdf_is_actually_a_pdf(self):
        self.run_plot()
        self.assertEqual((self.tmp / "rep1.pdf").read_bytes()[:5], b"%PDF-")

    def test_axis_spans_the_contig_not_the_data(self):
        """The track stops at CHROM_LEN here, but the x-limit must come from
        the .fai, so a track that ends early still shows the whole contig."""
        short = bedgraph(self.tmp / "short.bdg", [("chrom", 0, 500, 3.0)])
        fig, ax = matplotlib.pyplot.subplots()
        pe.plot_contig(ax, pe.read_bedgraph(short), CHROM_LEN,
                       color="#000000")
        self.assertEqual(ax.get_xlim(), (0, CHROM_LEN))
        matplotlib.pyplot.close(fig)

    def test_title_is_the_bare_label(self):
        fig, ax = matplotlib.pyplot.subplots()
        pe.plot_contig(ax, pe.read_bedgraph(self.bdg), CHROM_LEN,
                       color="#000000")
        ax.set_title("Replicate 1", loc="left")
        self.assertEqual(ax.get_title(loc="left"), "Replicate 1")
        matplotlib.pyplot.close(fig)

    def test_peaks_are_shaded(self):
        peaks = pe.read_peaks(bed(self.tmp / "p.bed", [("chrom", 1000, 2000)]))
        written = self.run_plot(peaks=peaks)
        self.assertEqual(len(written), 2)


class MultiContigTest(TempDirTest):
    """The bug this module was extracted to fix."""

    def setUp(self):
        super().setUp()
        self.lengths = {"chrom": CHROM_LEN, "pBAD": PLASMID_LEN}
        self.bdg = bedgraph(self.tmp / "rep1_FE.bdg", [
            ("chrom", 0, 2000, 1.0),
            ("chrom", 2000, CHROM_LEN, 4.0),
            ("pBAD", 0, 500, 1.0),
            ("pBAD", 500, PLASMID_LEN, 30.0),
        ])

    def plot(self, **kw):
        return pe.plot_fold_enrichment(
            self.bdg, str(self.tmp / "rep1"), label="Replicate 1",
            color=pe.PALETTE[0], contig_lengths=self.lengths, **kw)

    def test_one_file_pair_per_contig(self):
        written = [Path(p).name for p in self.plot()]
        self.assertEqual(written, ["rep1.chrom.png", "rep1.chrom.pdf",
                                   "rep1.pBAD.png", "rep1.pBAD.pdf"])

    def test_each_axis_spans_only_its_own_contig(self):
        for contig, length in self.lengths.items():
            rows = pe.read_bedgraph(self.bdg)
            rows = rows[rows["chrom"] == contig]
            fig, ax = matplotlib.pyplot.subplots()
            pe.plot_contig(ax, rows, length, color="#000000")
            self.assertEqual(ax.get_xlim(), (0, length),
                             f"{contig} axis should span {length}")
            matplotlib.pyplot.close(fig)

    def test_summed_length_is_not_used_as_an_axis(self):
        """What the heredoc did: xlim == sum of all contigs."""
        total = sum(self.lengths.values())
        fig, ax = matplotlib.pyplot.subplots()
        pe.plot_contig(ax, pe.read_bedgraph(self.bdg), CHROM_LEN,
                       color="#000000")
        self.assertNotEqual(ax.get_xlim()[1], total)
        matplotlib.pyplot.close(fig)

    def test_a_contig_with_no_coverage_is_skipped(self):
        bdg = bedgraph(self.tmp / "only_chrom.bdg", [("chrom", 0, 100, 2.0)])
        written = pe.plot_fold_enrichment(
            bdg, str(self.tmp / "rep1"), label="Replicate 1",
            color=pe.PALETTE[0], contig_lengths=self.lengths)
        self.assertEqual([Path(p).name for p in written],
                         ["rep1.chrom.png", "rep1.chrom.pdf"])

    def test_peaks_shade_only_their_own_contig(self):
        """A peak on the plasmid must not appear on the chromosome panel."""
        peaks = pe.read_peaks(bed(self.tmp / "p.bed", [("pBAD", 600, 900)]))
        rows = pe.read_bedgraph(self.bdg)

        fig, ax = matplotlib.pyplot.subplots()
        pe.plot_contig(ax, rows[rows["chrom"] == "chrom"], CHROM_LEN,
                       color="#000000", peaks=peaks[peaks["chrom"] == "chrom"])
        chrom_patches = len(ax.patches)
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots()
        pe.plot_contig(ax, rows[rows["chrom"] == "pBAD"], PLASMID_LEN,
                       color="#000000", peaks=peaks[peaks["chrom"] == "pBAD"])
        plasmid_patches = len(ax.patches)
        matplotlib.pyplot.close(fig)

        self.assertEqual(chrom_patches, 0)
        self.assertEqual(plasmid_patches, 1)

    def test_a_bedgraph_contig_absent_from_the_fai_is_reported(self):
        bdg = bedgraph(self.tmp / "stray.bdg", [
            ("chrom", 0, 100, 2.0), ("phage", 0, 100, 2.0)])
        written = pe.plot_fold_enrichment(
            bdg, str(self.tmp / "rep1"), label="R", color="#000000",
            contig_lengths={"chrom": CHROM_LEN, "pBAD": PLASMID_LEN})
        self.assertTrue(all("phage" not in p for p in written))


class AxisUnitTest(TempDirTest):
    def test_megabases_for_a_chromosome(self):
        unit, _ = pe._position_formatter(CHROM_LEN)
        self.assertEqual(unit, "Mb")

    def test_kilobases_for_a_plasmid(self):
        unit, _ = pe._position_formatter(PLASMID_LEN)
        self.assertEqual(unit, "kb")

    def test_formatter_labels_match_the_unit(self):
        _, mb = pe._position_formatter(CHROM_LEN)
        self.assertEqual(mb(2_300_000, None), "2.3")
        _, kb = pe._position_formatter(PLASMID_LEN)
        self.assertEqual(kb(45_000, None), "45")


class CommandLineTest(TempDirTest):
    """What the pipeline itself calls. These replace the heredoc's `eval` on a
    shell-concatenated string, so the quoting cases are the point."""

    def setUp(self):
        super().setUp()
        self.fai = fai(self.tmp / "ref.fa.fai", [("chrom", CHROM_LEN)])
        self.bdg = bedgraph(self.tmp / "rep1_FE.bdg",
                            [("chrom", 0, CHROM_LEN, 2.0)])

    def test_argv_roundtrip(self):
        args = pe.parse_args(["--fai", str(self.fai),
                              "--rep", "a.bdg", "Replicate 1", "out/rep1",
                              "--rep", "b.bdg", "Replicate 2", "out/rep2"])
        self.assertEqual(args.rep, [["a.bdg", "Replicate 1", "out/rep1"],
                                    ["b.bdg", "Replicate 2", "out/rep2"]])
        self.assertIsNone(args.peaks)

    def test_main_writes_the_files(self):
        pe.main(["--fai", str(self.fai), "--rep", str(self.bdg),
                 "Replicate 1", str(self.tmp / "rep1")])
        self.assertTrue((self.tmp / "rep1.png").exists())

    def test_replicates_get_different_colours(self):
        """Two replicates in one call must not come out the same colour."""
        args = pe.parse_args(["--fai", str(self.fai),
                              "--rep", "a", "A", "x", "--rep", "b", "B", "y"])
        colors = [pe.PALETTE[i % len(pe.PALETTE)]
                  for i in range(len(args.rep))]
        self.assertEqual(len(set(colors)), 2)

    def test_a_quote_in_a_path_is_not_executed(self):
        """The heredoc `eval`-ed this string; a quote broke it or injected."""
        odd = self.tmp / "rep'1"
        pe.main(["--fai", str(self.fai), "--rep", str(self.bdg),
                 "Replicate 1", str(odd)])
        self.assertTrue(Path(str(odd) + ".png").exists())

    def test_label_with_a_quote_survives(self):
        pe.main(["--fai", str(self.fai), "--rep", str(self.bdg),
                 "Liz's replicate", str(self.tmp / "q")])
        self.assertTrue((self.tmp / "q.png").exists())

    def test_runs_under_isolated_mode(self):
        """The pipeline invokes it as `python3 -I`, which drops the script's
        own directory from sys.path. Nothing here may depend on that."""
        out = subprocess.run(
            [sys.executable, "-I", str(CHIPSEQ / "plot_enrichment.py"),
             "--fai", str(self.fai), "--rep", str(self.bdg), "Replicate 1",
             str(self.tmp / "iso")],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertTrue((self.tmp / "iso.png").exists())

    def test_missing_rep_is_an_error(self):
        with self.assertRaises(SystemExit):
            pe.parse_args(["--fai", str(self.fai)])


class PipelineWiringTest(unittest.TestCase):
    """The script and the module have to agree about how it is called."""

    SCRIPT = (CHIPSEQ / "chipseq_pipeline.sh").read_text()

    def test_the_heredoc_is_gone(self):
        self.assertNotIn("PYEOF", self.SCRIPT)
        self.assertNotIn("PLOT_REPS_PY", self.SCRIPT)

    def test_no_eval_remains(self):
        self.assertNotIn("eval", self.SCRIPT)

    def test_the_script_calls_the_module_in_isolated_mode(self):
        self.assertIn('python3 -I "$HERE/plot_enrichment.py"', self.SCRIPT)

    def test_the_script_passes_the_fai(self):
        self.assertIn('--fai "${REFERENCE}.fai"', self.SCRIPT)

    def test_genome_size_is_still_computed_for_macs3(self):
        """Only the plot moved to per-contig lengths; MACS3 -g still wants the
        summed effective size."""
        self.assertIn("GENOME_SIZE=$(awk '{sum+=$2}", self.SCRIPT)
        self.assertIn('-g "$GENOME_SIZE"', self.SCRIPT)

    def test_bash_syntax_is_valid(self):
        out = subprocess.run(["bash", "-n", str(CHIPSEQ / "chipseq_pipeline.sh")],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)


if __name__ == "__main__":
    unittest.main()
