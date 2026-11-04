"""Tests for the UdiTaS CBCL readers in `apps/uditas/cbcl_qc/`.

This code had no tests because it reads Illumina `.cbcl` files directly and
there was no way to hand it an input. `make_cbcl_fixtures.py` writes a
synthetic lane, which makes all of it testable: the nibble decoding, the
quality-bin mapping, the i7/i5 demultiplexing with its one-mismatch tolerance,
and the read classification `extract_salvage.py` performs.

Needs numpy, which the cbcl_qc scripts require anyway.
"""

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CBCL_QC = REPO / "apps" / "uditas" / "cbcl_qc"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CBCL_QC))

import make_cbcl_fixtures as fx  # noqa: E402

try:
    import numpy as np
    import extract_salvage as es
    import recheck_inserts as ri
    import verify_structure as vs
    IMPORT_ERROR = None
except Exception as exc:                      # pragma: no cover
    IMPORT_ERROR = str(exc)


@unittest.skipIf(IMPORT_ERROR, f"cannot import cbcl_qc modules: {IMPORT_ERROR}")
class PureHelperTest(unittest.TestCase):
    """The small functions, which are also the ones duplicated the most."""

    def test_reverse_complement_handles_n(self):
        self.assertEqual(es.rc("ACGT"), "ACGT")
        self.assertEqual(es.rc("AAGGAGTA"), "TACTCCTT")
        self.assertEqual(es.rc("ACGTN"), "NACGT")

    def test_rc_is_defined_identically_in_every_module(self):
        """It is copied into six files; until that is consolidated, at least
        hold the copies to the same behaviour."""
        for seq in ["ACGTN", "AAGGAGTA", "TAGATCGC", ""]:
            self.assertEqual(es.rc(seq), vs.rc(seq))
            self.assertEqual(es.rc(seq), ri.rc(seq))

    def test_one_mismatch_map_accepts_exact_barcodes(self):
        m = es.one_mismatch_map(["AAAACCCC", "GGGGTTTT"])
        self.assertEqual(m["AAAACCCC"], "AAAACCCC")
        self.assertEqual(m["GGGGTTTT"], "GGGGTTTT")

    def test_one_mismatch_map_accepts_a_single_substitution(self):
        m = es.one_mismatch_map(["AAAACCCC"])
        self.assertEqual(m["TAAACCCC"], "AAAACCCC")
        self.assertEqual(m["AAAACCCT"], "AAAACCCC")
        self.assertEqual(m["AAAANCCC"], "AAAACCCC")

    def test_one_mismatch_map_rejects_two_substitutions(self):
        m = es.one_mismatch_map(["AAAACCCC"])
        self.assertIsNone(m.get("TTAACCCC"))

    def test_one_mismatch_map_drops_ambiguous_variants(self):
        """Two barcodes one base apart: the variant between them resolves to
        neither, and must not be assigned to one of them arbitrarily."""
        m = es.one_mismatch_map(["AAAACCCC", "AAAACCCG"])
        self.assertEqual(m["AAAACCCC"], "AAAACCCC")
        self.assertEqual(m["AAAACCCG"], "AAAACCCG")
        self.assertIsNone(m.get("AAAACCCT"))
        self.assertIsNone(m.get("AAAACCCN"))

    def test_one_mismatch_map_agrees_across_its_three_copies(self):
        barcodes = [es.rc(i7) for _, _, _, i7 in es.SAMPLES]
        self.assertEqual(es.one_mismatch_map(barcodes),
                         vs.one_mismatch_map(barcodes))
        self.assertEqual(es.one_mismatch_map(barcodes),
                         ri.one_mismatch_map(barcodes))

    def test_real_sample_barcodes_are_all_distinguishable(self):
        """If any two i7s or i5s were within one mismatch of each other, reads
        would be silently dropped as ambiguous."""
        for idx in (2, 3):
            codes = sorted({s[idx] for s in es.SAMPLES})
            m = es.one_mismatch_map([es.rc(c) for c in codes])
            for c in codes:
                self.assertEqual(m.get(es.rc(c)), es.rc(c), c)

    def test_mismatch_count_ignores_no_calls(self):
        self.assertEqual(es.mism("ACGT", "ACGT"), 0)
        self.assertEqual(es.mism("ACGT", "ACGA"), 1)
        self.assertEqual(es.mism("ACGN", "ACGA"), 0, "N should not count")
        self.assertEqual(es.mism("ACGA", "ACGN"), 0, "either side")

    def test_mismatch_count_stops_at_the_shorter_sequence(self):
        self.assertEqual(es.mism("ACGT", "ACGTTTTT"), 0)


@unittest.skipIf(IMPORT_ERROR, f"cannot import cbcl_qc modules: {IMPORT_ERROR}")
class EarliestAdapterTest(unittest.TestCase):
    """recheck_inserts.earliest_adapter, the mismatch-tolerant scan."""

    def matrix(self, *seqs):
        """Reads as a base matrix, padded with N so rows are equal length.
        N never matches a probe base, so padding cannot create a hit."""
        width = max(len(s) for s in seqs)
        return np.array([list(s.ljust(width, "N").encode()) for s in seqs],
                        dtype=np.uint8)

    def test_finds_an_exact_mosaic_end(self):
        read = "A" * 30 + ri.ME + "C" * 30
        self.assertEqual(int(ri.earliest_adapter(self.matrix(read))[0]), 30)

    def test_tolerates_two_mismatches_in_the_mosaic_end(self):
        damaged = "T" + ri.ME[1:-1] + "A"
        read = "A" * 25 + damaged + "C" * 30
        self.assertEqual(int(ri.earliest_adapter(self.matrix(read))[0]), 25)

    def test_reports_the_earliest_of_several_hits(self):
        read = "A" * 10 + ri.ME + "A" * 10 + ri.ME + "A" * 10
        self.assertEqual(int(ri.earliest_adapter(self.matrix(read))[0]), 10)

    def test_finds_the_core_when_the_mosaic_end_is_truncated(self):
        """A mosaic end cut off by the read start still has its core, which is
        why the probe list includes the shorter fragments."""
        read = "TATACACATCT" + "G" * 60
        self.assertEqual(int(ri.earliest_adapter(self.matrix(read))[0]), 0)

    def test_returns_the_sentinel_when_nothing_matches(self):
        read = "ACAC" * 20
        self.assertEqual(int(ri.earliest_adapter(self.matrix(read))[0]), 10**6)

    def test_handles_several_reads_at_once(self):
        hit = "A" * 15 + ri.ME + "C" * 30
        miss = "ACAC" * 20
        got = ri.earliest_adapter(self.matrix(hit, miss, hit))
        self.assertEqual([int(x) for x in got], [15, 10**6, 15])


@unittest.skipIf(IMPORT_ERROR, f"cannot import cbcl_qc modules: {IMPORT_ERROR}")
class CbclDecodingTest(unittest.TestCase):
    """The synthetic lane has to decode back to exactly what went in, or
    nothing built on top of it means anything."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.truth = fx.build(Path(cls._tmp.name))
        cls.lane = cls.truth["lane_dir"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_header_reports_the_tiles_it_was_written_with(self):
        path = f"{self.lane}/C{fx.I1_START}.1/L001_1.cbcl"
        with open(path, "rb") as fh:
            header_size, tiles, qmap = es.read_header(fh)
        self.assertEqual(len(tiles), self.truth["n_tiles"])
        self.assertEqual([t[1] for t in tiles],
                         self.truth["clusters_per_tile"])
        self.assertEqual(qmap, {int(b): q for b, q in self.truth["qbins"].items()})
        # Data starts exactly where the header says it does.
        self.assertEqual(header_size, 12 + 8 * len(self.truth["qbins"]) + 4
                         + 16 * self.truth["n_tiles"])

    def test_decoded_cluster_counts_match_each_tile(self):
        for t, n in enumerate(self.truth["clusters_per_tile"]):
            bases, _ = es.read_block(self.lane, fx.I1_START, fx.I1_LEN, t)
            self.assertEqual(bases.shape, (n, fx.I1_LEN))

    def test_quality_bins_decode_to_phred_characters(self):
        _, quals = es.read_block(self.lane, fx.R2_START, 10, 0, want_qual=True)
        expected = chr(33 + min(self.truth["qbins"][str(fx.GOOD_BIN)], 60))
        chars = {chr(c) for c in quals.flatten()}
        self.assertEqual(chars, {expected},
                         "every synthetic base was written in the good bin")

    def test_decode_cycle_and_decode_tile_agree(self):
        """verify_structure reads tiles sequentially and extract_salvage seeks
        by offset -- two implementations of the same thing, in two files."""
        whole = vs.decode_cycle(f"{self.lane}/C{fx.I1_START}.1/L001_1.cbcl",
                                self.truth["n_tiles"])
        pieces = [es.decode_tile(f"{self.lane}/C{fx.I1_START}.1/L001_1.cbcl",
                                 t, False)[0]
                  for t in range(self.truth["n_tiles"])]
        self.assertTrue(np.array_equal(whole, np.concatenate(pieces)))

    def test_a_no_call_decodes_to_n(self):
        """Quality bin 0 is the no-call bin, and must come back as "N" rather
        than as whatever the two base bits happen to say."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "C1.1" / "L001_1.cbcl"
            fx.write_cbcl(path, [["A", "N", "C", "N", "G", "T"]])
            seq, _ = es.decode_tile(str(path), 0, False)
        self.assertEqual(seq.tobytes().decode(), "ANCNGT")

    def test_an_odd_cluster_count_round_trips(self):
        """Base calls are packed two per byte, so an odd count leaves a spare
        nibble that must be ignored rather than returned as a cluster."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "C1.1" / "L001_1.cbcl"
            fx.write_cbcl(path, [["A", "C", "G", "T", "A"]])
            seq, _ = es.decode_tile(str(path), 0, False)
        self.assertEqual(seq.tobytes().decode(), "ACGTA")

    def test_every_base_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "C1.1" / "L001_1.cbcl"
            fx.write_cbcl(path, [list("ACGTACGTACGT")])
            seq, _ = es.decode_tile(str(path), 0, False)
        self.assertEqual(seq.tobytes().decode(), "ACGTACGTACGT")

    def test_indexes_decode_to_the_barcodes_that_were_written(self):
        """Every assignable cluster's I1 should map to a known i7, and the
        barcode half of I2 to a known i5."""
        i7_map = es.one_mismatch_map([es.rc(i7) for _, _, _, i7 in es.SAMPLES])
        i5_map = es.one_mismatch_map([es.rc(i5) for _, _, i5, _ in es.SAMPLES])
        assigned = 0
        for t in range(self.truth["n_tiles"]):
            i1, _ = es.read_block(self.lane, fx.I1_START, fx.I1_LEN, t)
            i2, _ = es.read_block(self.lane, fx.I2_START, fx.I2_LEN, t)
            for k in range(i1.shape[0]):
                bc7 = i7_map.get(i1[k].tobytes().decode())
                bc5 = i5_map.get(i2[k, fx.UMI_LEN:].tobytes().decode())
                if bc7 is not None and bc5 is not None:
                    assigned += 1
        expected = sum(g["assigned"] for g in self.truth["expected_groups"].values())
        self.assertEqual(assigned, expected)

    def test_unassignable_reads_are_present_and_not_assigned(self):
        total = sum(self.truth["clusters_per_tile"])
        assigned = sum(g["assigned"] for g in self.truth["expected_groups"].values())
        self.assertEqual(total - assigned, self.truth["n_unassigned"])


@unittest.skipIf(IMPORT_ERROR, f"cannot import cbcl_qc modules: {IMPORT_ERROR}")
class ExtractSalvageTest(unittest.TestCase):
    """extract_salvage.py end to end over the synthetic lane."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        work = Path(cls._tmp.name)
        cls.truth = fx.build(work)
        cls.out = work / "salvage"
        proc = subprocess.run(
            [sys.executable, "-I", str(CBCL_QC / "extract_salvage.py"),
             cls.truth["lane_dir"], str(cls.truth["n_tiles"]), str(cls.out)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f"extract_salvage failed:\n{proc.stderr[-4000:]}")
        cls.summary = json.loads((cls.out / "salvage_summary.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_read_fates_match_the_fixture(self):
        for key, exp in self.truth["expected_groups"].items():
            got = self.summary["groups"][key]
            for field, n in exp.items():
                self.assertEqual(got[field], n, f"{key}.{field}")

    def test_groups_with_no_reads_are_still_reported(self):
        """All eight index combinations collapse to four groups, and the
        fixture only populates two of them."""
        self.assertEqual(sorted(self.summary["groups"]),
                         ["293T_LE", "293T_RE", "FT_LE", "FT_RE"])
        for key in ["293T_LE", "293T_RE"]:
            self.assertEqual(self.summary["groups"][key]["assigned"], 0)

    def test_it_records_the_lane_and_tile_count(self):
        self.assertEqual(self.summary["lane"], self.truth["lane_dir"])
        self.assertEqual(self.summary["tiles"], self.truth["n_tiles"])

    def test_one_fastq_per_group(self):
        for key in self.summary["groups"]:
            self.assertTrue((self.out / f"{key}.fastq.gz").exists(), key)

    def test_fastq_record_count_matches_the_kept_count(self):
        for key, group in self.summary["groups"].items():
            with gzip.open(self.out / f"{key}.fastq.gz", "rt") as fh:
                lines = sum(1 for _ in fh)
            self.assertEqual(lines, 4 * group["kept"], key)

    def test_fastq_carries_the_umi_in_the_read_name(self):
        with gzip.open(self.out / "FT_RE.fastq.gz", "rt") as fh:
            names = [l[1:].strip() for i, l in enumerate(fh) if i % 4 == 0]
        self.assertTrue(names)
        for name in names:
            tile, cluster, umi = name.split("_")
            self.assertTrue(tile.startswith("t"))
            self.assertTrue(cluster.isdigit())
            self.assertEqual(len(umi), fx.UMI_LEN, name)
            self.assertTrue(set(umi) <= set("ACGTN"), umi)

    def test_sequence_and_quality_lengths_agree(self):
        with gzip.open(self.out / "FT_RE.fastq.gz", "rt") as fh:
            lines = [l.rstrip("\n") for l in fh]
        for i in range(0, len(lines), 4):
            self.assertEqual(len(lines[i + 1]), len(lines[i + 3]),
                             lines[i])

    def test_trimmed_flank_is_at_least_the_minimum(self):
        for key in ["FT_RE", "FT_LE"]:
            with gzip.open(self.out / f"{key}.fastq.gz", "rt") as fh:
                seqs = [l.strip() for i, l in enumerate(fh) if i % 4 == 1]
            for s in seqs:
                self.assertGreaterEqual(len(s), es.MIN_FLANK, key)
                self.assertNotIn("N", s, "flank is truncated at the first N")

    def test_the_primer_and_transposon_end_are_trimmed_off(self):
        for arm in ("RE", "LE"):
            with gzip.open(self.out / f"FT_{arm}.fastq.gz", "rt") as fh:
                seqs = [l.strip() for i, l in enumerate(fh) if i % 4 == 1]
            for s in seqs:
                self.assertNotIn(es.PRIMER[arm], s)
                self.assertNotIn(es.ME[:15], s, "adapter should be cut away")

    def test_umis_are_collapsed_per_site(self):
        """Three reads were planted at one site with two distinct UMIs, so that
        site must count once and contribute two UMIs, not three."""
        for key, flank in self.truth["shared_site_flank"].items():
            group = self.summary["groups"][key]
            top = {t["flank"]: t["umis"] for t in group["top_sites"]}
            site = flank[:es.SITE_KEY]
            self.assertIn(site, top, key)
            self.assertEqual(top[site], 2,
                             f"{key}: 3 reads, 2 distinct UMIs -> 2")
            self.assertLess(group["umi_collapsed"], group["kept"],
                            "collapsing should reduce the count")

    def test_a_one_mismatch_barcode_is_still_assigned(self):
        """One read per arm was given an i7 with a single substitution. If the
        one-mismatch map stopped working, `assigned` would drop by one."""
        for key, exp in self.truth["expected_groups"].items():
            self.assertEqual(self.summary["groups"][key]["assigned"],
                             exp["assigned"], key)

    def test_still_donor_is_detected_on_the_re_arm(self):
        self.assertTrue(self.truth["still_donor_check_active"]["RE"])
        self.assertEqual(self.summary["groups"]["FT_RE"]["still_donor"], 2)

    def test_still_donor_check_is_dead_on_the_le_arm(self):
        """A real finding, recorded rather than endorsed.

        extract_salvage.py only applies its still-donor test when it has at
        least 12 bases of donor sequence past the transposon end to compare
        against. DONOR["RE"] supplies 20, but DONOR["LE"] is only 26 bases long
        and 19 of those are the transposon end, leaving 7 -- so on the LE arm
        the check never runs.

        The fixture plants two LE reads that continue into donor sequence past
        the terminus. They are reported as salvaged junctions, which is what
        this test pins. Extending DONOR["LE"] past the terminus would activate
        the check and should make this test fail.
        """
        self.assertFalse(self.truth["still_donor_check_active"]["LE"])
        self.assertLess(self.truth["donor_bases_past_terminus"]["LE"], 12)
        self.assertEqual(self.summary["groups"]["FT_LE"]["still_donor"], 0)
        # The two donor reads landed in `kept` instead.
        self.assertEqual(self.summary["groups"]["FT_LE"]["kept"],
                         self.summary["groups"]["FT_RE"]["kept"] + 2)


@unittest.skipIf(IMPORT_ERROR, f"cannot import cbcl_qc modules: {IMPORT_ERROR}")
class HardcodedRunLayoutTest(unittest.TestCase):
    """The cycle layout and barcodes are hardcoded in several files. The
    architecture plan moves them into the sample sheet; until then, the copies
    must agree, because the scripts are run against the same data."""

    def test_cycle_layout_agrees_across_modules(self):
        for mod in (vs, ri):
            self.assertEqual((mod.I1_START, mod.I1_LEN),
                             (es.I1_START, es.I1_LEN), mod.__name__)
            self.assertEqual((mod.I2_START, mod.I2_LEN),
                             (es.I2_START, es.I2_LEN), mod.__name__)
            self.assertEqual(mod.R2_START, es.R2_START, mod.__name__)
            self.assertEqual(mod.UMI_LEN, es.UMI_LEN, mod.__name__)

    def test_umi_plus_barcode_fills_the_index_read(self):
        """I2 is 18 cycles: a 10 nt UMI then the 8 nt i5 barcode."""
        barcode_len = len({s[2] for s in es.SAMPLES}.pop())
        self.assertEqual(es.UMI_LEN + barcode_len, es.I2_LEN)

    def test_i1_length_matches_the_i7_barcodes(self):
        for _, _, _, i7 in es.SAMPLES:
            self.assertEqual(len(i7), es.I1_LEN)

    def test_sample_barcode_pairs_are_unique(self):
        pairs = [(i5, i7) for _, _, i5, i7 in es.SAMPLES]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_every_index_pair_maps_to_exactly_one_group(self):
        combo = {(es.rc(i7), es.rc(i5)): (s, a)
                 for s, a, i5, i7 in es.SAMPLES}
        self.assertEqual(len(combo), len(es.SAMPLES))

    def test_primers_agree_between_modules(self):
        """extract_salvage keys primers by arm, verify_structure by the primer
        number in the sample name. Same two sequences, two spellings."""
        self.assertEqual(es.PRIMER["RE"], vs.PRIMERS["251"])
        self.assertEqual(es.PRIMER["LE"], vs.PRIMERS["253"])

    def test_recheck_primer_lengths_match_the_sequences(self):
        self.assertEqual(ri.PRIMER["RE"], len(es.PRIMER["RE"]))
        self.assertEqual(ri.PRIMER["LE"], len(es.PRIMER["LE"]))

    def test_mosaic_end_is_the_same_everywhere(self):
        self.assertEqual(es.ME, ri.ME)

    def test_transposon_end_length_fits_inside_the_donor(self):
        for arm in ("RE", "LE"):
            self.assertLessEqual(es.END_LEN[arm], len(es.DONOR[arm]), arm)


if __name__ == "__main__":
    unittest.main()
