"""Tests for apps/uditas/cbcl_qc/runconfig.py.

The config is now the single source of the cycle layout, the sample barcodes
and the LE/RE arm sequences, so a bad config is the one way the three CBCL
scripts can go wrong together. These tests write small configs to a temporary
directory and check that the loader rejects the inconsistent ones.

They also pin the real apps/uditas/run.yaml, because it is the record of how
run 261030 was processed.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CBCL_QC = REPO / "apps" / "uditas" / "cbcl_qc"
sys.path.insert(0, str(CBCL_QC))

IMPORT_ERROR = None
try:
    import runconfig as rc
except Exception as exc:                                  # pragma: no cover
    IMPORT_ERROR = exc

GOOD = {
    "run_id": "test_run",
    "platform": "test",
    "sample_sheet": "sheet.csv",
    "cycles": {"i1_start": 302, "i1_len": 8, "i2_start": 310, "i2_len": 18,
               "r2_start": 328, "r2_len": 220, "umi_len": 10},
    "assay": {
        "mosaic_end": "AGATGTGTATAAGAGACAG",
        "min_flank": 40,
        "site_key": 40,
        "still_donor_min_tail": 12,
        "arms": {
            "RE": {"primer": "AGCTGACGCAAGAATGAGGGT",
                   "donor": "TGTAGTGTTGTTGACGGCAAAGTTATGGTTCCAGTAACGGCAAAGTTATGGGT",
                   "end_len": 11},
            "LE": {"primer": "GCATAAGCTTGCCGTTGCGG",
                   "donor": "CAAAGTTATGGGTAAAGTCACAATCC",
                   "end_len": 19},
        },
    },
}

SHEET = """library,sample,arm,i5_index,i7_index
A_RE,A,RE,TAGATCGC,AGCGGAAT
A_LE,A,LE,TAGATCGC,GATCATGC
B_RE,B,RE,AAGGAGTA,AAGACGGA
B_LE,B,LE,AAGGAGTA,CGAGTCCT
"""


@unittest.skipIf(IMPORT_ERROR, f"cannot import runconfig: {IMPORT_ERROR}")
class LoaderTest(unittest.TestCase):
    """Writes a config pair into a temp dir, optionally mutated, and loads it."""

    def build(self, mutate=None, sheet=SHEET, suffix=".json"):
        d = Path(tempfile.mkdtemp())
        data = json.loads(json.dumps(GOOD))     # deep copy
        if mutate:
            mutate(data)
        (d / "sheet.csv").write_text(sheet)
        p = d / f"run{suffix}"
        p.write_text(json.dumps(data))
        return p

    def load(self, **kw):
        return rc.load(self.build(**kw))

    # --- the happy path -------------------------------------------------

    def test_a_consistent_config_loads(self):
        cfg = self.load()
        self.assertEqual(cfg.run_id, "test_run")
        self.assertEqual(len(cfg.samples), 4)
        self.assertEqual(sorted(cfg.arms), ["LE", "RE"])

    def test_json_needs_no_yaml_library(self):
        """The .json path must not import PyYAML, so these scripts still run
        on a bare Python with only numpy installed."""
        cfg = self.load(suffix=".json")
        self.assertEqual(cfg.source.suffix, ".json")

    def test_groups_are_sample_arm_pairs_in_sheet_order(self):
        self.assertEqual(self.load().groups,
                         [("A", "RE"), ("A", "LE"), ("B", "RE"), ("B", "LE")])

    def test_cycles_are_read_as_integers(self):
        c = self.load().cycles
        self.assertEqual((c.i1_start, c.i1_len), (302, 8))
        self.assertEqual(c.umi_len + 8, c.i2_len)

    def test_barcodes_are_upper_cased(self):
        """Only the barcode columns are lowered: an arm spelled "re" is a
        different arm, and the loader is right to reject it."""
        lower = "\n".join(
            line if i == 0 else ",".join(
                f.lower() if n >= 3 else f
                for n, f in enumerate(line.split(",")))
            for i, line in enumerate(SHEET.strip().splitlines()))
        cfg = rc.load(self.build(sheet=lower + "\n"))
        for s in cfg.samples:
            self.assertEqual(s.i5, s.i5.upper())
            self.assertEqual(s.i7, s.i7.upper())

    def test_arm_lookup_names_the_alternatives_when_missing(self):
        with self.assertRaises(rc.ConfigError) as cm:
            self.load().arm("MIDDLE")
        self.assertIn("LE", str(cm.exception))

    # --- the still-donor trap -------------------------------------------

    def test_a_long_tail_activates_the_still_donor_check(self):
        self.assertTrue(self.load().arm("RE").still_donor_active)

    def test_a_short_tail_deactivates_it(self):
        """The LE arm of the real run: 7 bases past the terminus, under the
        12 the check needs, so it cannot fire."""
        le = self.load().arm("LE")
        self.assertEqual(len(le.donor_past_terminus), 7)
        self.assertFalse(le.still_donor_active)

    def test_inactive_arms_are_reported(self):
        self.assertEqual(self.load().inactive_still_donor_arms(), ["LE"])

    def test_extending_the_donor_switches_the_check_on(self):
        """What fixing the bug looks like: more donor, no code change."""
        def mutate(d):
            d["assay"]["arms"]["LE"]["donor"] += "GGGTTTAAACCC"
        cfg = self.load(mutate=mutate)
        self.assertTrue(cfg.arm("LE").still_donor_active)
        self.assertEqual(cfg.inactive_still_donor_arms(), [])

    def test_the_threshold_is_configurable(self):
        def mutate(d):
            d["assay"]["still_donor_min_tail"] = 7
        self.assertTrue(self.load(mutate=mutate).arm("LE").still_donor_active)

    def test_donor_past_terminus_starts_after_end_len(self):
        re = self.load().arm("RE")
        self.assertEqual(re.donor[:re.end_len] + re.donor_past_terminus, re.donor)

    # --- rejections ------------------------------------------------------

    def test_umi_plus_barcode_must_fill_i2(self):
        def mutate(d):
            d["cycles"]["umi_len"] = 9
        with self.assertRaises(rc.ConfigError) as cm:
            self.load(mutate=mutate)
        self.assertIn("I2", str(cm.exception))

    def test_i7_length_must_match_i1(self):
        def mutate(d):
            d["cycles"]["i1_len"] = 10
        with self.assertRaises(rc.ConfigError) as cm:
            self.load(mutate=mutate)
        self.assertIn("I1", str(cm.exception))

    def test_duplicate_index_pairs_are_rejected(self):
        dupe = SHEET.replace("B_RE,B,RE,AAGGAGTA,AAGACGGA",
                             "B_RE,B,RE,TAGATCGC,AGCGGAAT")
        with self.assertRaises(rc.ConfigError) as cm:
            self.load(sheet=dupe)
        self.assertIn("twice", str(cm.exception))

    def test_duplicate_library_names_are_rejected(self):
        dupe = SHEET.replace("B_RE,B,RE", "A_RE,B,RE")
        with self.assertRaises(rc.ConfigError):
            self.load(sheet=dupe)

    def test_end_len_past_the_donor_is_rejected(self):
        def mutate(d):
            d["assay"]["arms"]["LE"]["end_len"] = 999
        with self.assertRaises(rc.ConfigError) as cm:
            self.load(mutate=mutate)
        self.assertIn("end_len", str(cm.exception))

    def test_an_arm_in_the_sheet_must_exist_in_the_config(self):
        odd = SHEET.replace("B_LE,B,LE", "B_XX,B,XX")
        with self.assertRaises(rc.ConfigError) as cm:
            self.load(sheet=odd)
        self.assertIn("XX", str(cm.exception))

    def test_non_dna_sequences_are_rejected(self):
        def mutate(d):
            d["assay"]["arms"]["RE"]["primer"] = "ACGTXACGT"
        with self.assertRaises(rc.ConfigError) as cm:
            self.load(mutate=mutate)
        self.assertIn("not DNA", str(cm.exception))

    def test_mixed_barcode_lengths_are_rejected(self):
        short = SHEET.replace("AAGGAGTA,AAGACGGA", "AAGGAG,AAGACGGA")
        with self.assertRaises(rc.ConfigError):
            self.load(sheet=short)

    def test_a_missing_section_names_the_key(self):
        def mutate(d):
            del d["assay"]["mosaic_end"]
        with self.assertRaises(rc.ConfigError) as cm:
            self.load(mutate=mutate)
        self.assertIn("mosaic_end", str(cm.exception))

    def test_a_missing_sheet_column_names_it(self):
        with self.assertRaises(rc.ConfigError) as cm:
            self.load(sheet="library,sample,arm,i5_index\nA,A,RE,TAGATCGC\n")
        self.assertIn("i7_index", str(cm.exception))

    def test_an_empty_sheet_is_rejected(self):
        with self.assertRaises(rc.ConfigError):
            self.load(sheet="library,sample,arm,i5_index,i7_index\n")

    def test_a_missing_file_is_rejected(self):
        with self.assertRaises(rc.ConfigError):
            rc.load("/nonexistent/run.yaml")

    def test_an_unknown_extension_is_rejected(self):
        p = self.build()
        odd = p.with_suffix(".txt")
        odd.write_text(p.read_text())
        with self.assertRaises(rc.ConfigError):
            rc.load(odd)

    def test_no_arms_is_rejected(self):
        def mutate(d):
            d["assay"]["arms"] = {}
        with self.assertRaises(rc.ConfigError):
            self.load(mutate=mutate)


@unittest.skipIf(IMPORT_ERROR, f"cannot import runconfig: {IMPORT_ERROR}")
class RealRunConfigTest(unittest.TestCase):
    """The committed run.yaml is the record of how run 261030 was processed."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = rc.load()

    def test_it_loads(self):
        self.assertEqual(self.cfg.run_id, "261030_VH00889_534_AAJLG3JM5")

    def test_cycle_layout_is_the_one_in_the_run_folder(self):
        c = self.cfg.cycles
        self.assertEqual((c.i1_start, c.i1_len), (302, 8))
        self.assertEqual((c.i2_start, c.i2_len), (310, 18))
        self.assertEqual(c.r2_start, 328)
        self.assertEqual(c.umi_len, 10)

    def test_eight_libraries_pool_into_four_groups(self):
        self.assertEqual(len(self.cfg.samples), 8)
        self.assertEqual(len(self.cfg.groups), 4)
        self.assertEqual(sorted(self.cfg.groups),
                         [("293T", "LE"), ("293T", "RE"),
                          ("FT", "LE"), ("FT", "RE")])

    def test_the_le_still_donor_check_is_still_off(self):
        """Recorded, not endorsed. Extend DONOR LE in run.yaml to fix it; this
        test is the tripwire that says the fix landed."""
        self.assertEqual(self.cfg.inactive_still_donor_arms(), ["LE"])

    def test_primer_lengths_match_what_recheck_inserts_used(self):
        self.assertEqual(len(self.cfg.arm("RE").primer), 21)
        self.assertEqual(len(self.cfg.arm("LE").primer), 20)

    def test_the_sheet_sits_next_to_the_config(self):
        self.assertTrue((self.cfg.source.parent / "sample_sheet.csv").exists())


if __name__ == "__main__":
    unittest.main()
