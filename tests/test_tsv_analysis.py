"""Tests for insertion-table analysis and for the CAST insertion geometry.

Covers `seqsuite.TsvMapHelperFunctions` (the parts that are pure
arithmetic on a table) and `seqsuite.NovoCAST_helper`. The geometry in
NovoCAST_helper is fixed by two constants, so its output is fully predictable
and is asserted here position by position -- which matters, because the
architecture plan moves those constants into `apps/ecoli_tagmentation` and the
same 71/10 are currently written inline in a second place.
"""

import random
import unittest

import pandas as pd

from seqsuite import NovoCAST_helper as novo
from seqsuite import TsvMapHelperFunctions as tsvhp
from seqsuite import postprocessHelpers as hp


def make_tsv(rows):
    """Build an insertion table with the real column names."""
    return pd.DataFrame(
        [{"read_id": f"r{i}", "ref": "toy_chr", "ins0": pos, "ins1": pos + 1,
          "strand": strand, "mapq": 60, "cigar": "70M",
          "aln_start0": pos, "aln_end0_excl": pos + 70}
         for i, (pos, strand) in enumerate(rows)])


class PercentWithinDistanceTest(unittest.TestCase):
    def test_all_inside_the_window(self):
        self.assertEqual(
            tsvhp.percent_within_distance([100, 101, 99], 100, 5), 100.0)

    def test_none_inside_the_window(self):
        self.assertEqual(
            tsvhp.percent_within_distance([0, 1000], 500, 10), 0.0)

    def test_half_inside_the_window(self):
        self.assertEqual(
            tsvhp.percent_within_distance([100, 100, 500, 900], 100, 10), 50.0)

    def test_window_boundary_is_inclusive(self):
        self.assertEqual(tsvhp.percent_within_distance([110], 100, 10), 100.0)
        self.assertEqual(tsvhp.percent_within_distance([111], 100, 10), 0.0)

    def test_symmetric_about_the_target(self):
        self.assertEqual(tsvhp.percent_within_distance([90], 100, 10), 100.0)
        self.assertEqual(tsvhp.percent_within_distance([89], 100, 10), 0.0)

    def test_zero_width_window(self):
        self.assertEqual(
            tsvhp.percent_within_distance([100, 101], 100, 0), 50.0)

    def test_empty_input_returns_zero(self):
        self.assertEqual(tsvhp.percent_within_distance([], 100, 10), 0.0)

    def test_accepts_a_dataframe_column(self):
        tsv = make_tsv([(100, "+"), (100, "+"), (9999, "+")])
        self.assertAlmostEqual(
            tsvhp.percent_within_distance(tsv["ins0"], 100, 50), 200 / 3)

    def test_needs_a_sized_collection_not_a_generator(self):
        """It calls len() on its input, so a generator raises. Worth pinning:
        it is an easy thing to break when this moves into ngslib."""
        with self.assertRaises(TypeError):
            tsvhp.percent_within_distance((p for p in [100]), 100, 10)


class TopNInsertionsFrequenciesTest(unittest.TestCase):
    INSERTIONS = [10] * 5 + [20] * 3 + [30] * 2  # 10 reads over 3 positions

    def test_frequencies_are_fractions_of_the_total(self):
        got = tsvhp.top_n_insertions_frequencies(self.INSERTIONS, 3)
        self.assertEqual(got, [(10, 0.5), (20, 0.3), (30, 0.2)])

    def test_sorted_by_descending_frequency(self):
        freqs = [f for _, f in
                 tsvhp.top_n_insertions_frequencies(self.INSERTIONS, 3)]
        self.assertEqual(freqs, sorted(freqs, reverse=True))

    def test_ties_are_broken_by_ascending_position(self):
        got = tsvhp.top_n_insertions_frequencies([50, 50, 10, 10, 99], 2)
        self.assertEqual([p for p, _ in got], [10, 50])

    def test_n_larger_than_the_number_of_positions(self):
        self.assertEqual(len(tsvhp.top_n_insertions_frequencies(self.INSERTIONS, 99)), 3)

    def test_frequencies_sum_to_one_when_n_covers_everything(self):
        got = tsvhp.top_n_insertions_frequencies(self.INSERTIONS, 99)
        self.assertAlmostEqual(sum(f for _, f in got), 1.0)

    def test_truncates_to_n(self):
        self.assertEqual(len(tsvhp.top_n_insertions_frequencies(self.INSERTIONS, 1)), 1)

    def test_empty_or_non_positive_n_returns_empty(self):
        self.assertEqual(tsvhp.top_n_insertions_frequencies([], 5), [])
        self.assertEqual(tsvhp.top_n_insertions_frequencies(self.INSERTIONS, 0), [])
        self.assertEqual(tsvhp.top_n_insertions_frequencies(self.INSERTIONS, -1), [])


class WhichAndBetweenTest(unittest.TestCase):
    def setUp(self):
        self.tsv = make_tsv([(100, "+"), (100, "-"), (200, "+"), (300, "+")])

    def test_which_selects_one_position_on_both_strands(self):
        self.assertEqual(len(tsvhp.which(self.tsv, 100)), 2)

    def test_which_on_an_absent_position_is_empty(self):
        self.assertTrue(tsvhp.which(self.tsv, 12_345).empty)

    def test_between_is_inclusive_of_both_ends(self):
        self.assertEqual(sorted(tsvhp.between(self.tsv, 100, 200)["ins0"]),
                         [100, 100, 200])

    def test_between_excludes_outside(self):
        self.assertEqual(list(tsvhp.between(self.tsv, 150, 250)["ins0"]), [200])


class PredictedInsertionGeometryTest(unittest.TestCase):
    """NovoCAST_helper.predict_insertion_from_spacer, checked against the
    constants rather than against a stored output."""

    GUIDE = "GATTACAGGCCTAATCGTAC"   # 20 bp, no internal repeats

    def setUp(self):
        rng = random.Random(7)
        filler = "".join(rng.choice("ACGT") for _ in range(3000))
        self.start = 1000
        self.genome = (filler[:self.start] + self.GUIDE
                       + filler[self.start + len(self.GUIDE):])
        self.end = self.start + len(self.GUIDE)

    def test_plus_strand_predicted_position(self):
        df = novo.predict_insertion_from_spacer(self.GUIDE, self.genome)
        row = df.iloc[0]
        self.assertEqual(row["gRNA_strand"], "+")
        self.assertEqual(row["proto_start"], self.start)
        self.assertEqual(row["proto_end"], self.end)
        # PAM is the 3 bases after the protospacer; the insertion sits
        # INSERTION_DISTANCE beyond the end of the PAM.
        self.assertEqual(row["PAM_start"], self.end)
        self.assertEqual(row["PAM_end"], self.end + 3)
        self.assertEqual(row["PAM"], self.genome[self.end:self.end + 3])
        predicted = self.end + 3 + novo.INSERTION_DISTANCE
        self.assertEqual(row["intervene_end"], predicted)
        self.assertEqual(row["ts_start"], predicted - novo.TARGET_SITE_WINDOW)
        self.assertEqual(row["ts_end"], predicted + novo.TARGET_SITE_WINDOW)

    def test_minus_strand_predicted_position(self):
        rc_guide = hp.reverse_complement(self.GUIDE)
        rng = random.Random(11)
        filler = "".join(rng.choice("ACGT") for _ in range(3000))
        start = 1500
        genome = filler[:start] + rc_guide + filler[start + len(rc_guide):]

        df = novo.predict_insertion_from_spacer(self.GUIDE, genome)
        row = df.iloc[0]
        self.assertEqual(row["gRNA_strand"], "-")
        # Everything mirrors: the PAM is upstream, the insertion further
        # upstream still.
        self.assertEqual(row["PAM_start"], start - 3)
        self.assertEqual(row["PAM_end"], start)
        predicted = start - 3 - novo.INSERTION_DISTANCE
        self.assertEqual(row["intervene_start"], predicted)
        self.assertEqual(row["ts_start"], predicted - novo.TARGET_SITE_WINDOW)

    def test_intervening_length_equals_the_insertion_distance(self):
        for guide in [self.GUIDE, hp.reverse_complement(self.GUIDE)]:
            rng = random.Random(len(guide))
            filler = "".join(rng.choice("ACGT") for _ in range(3000))
            genome = filler[:1200] + guide + filler[1200 + len(guide):]
            df = novo.predict_insertion_from_spacer(self.GUIDE, genome)
            self.assertEqual(df.iloc[0]["insertion_length"],
                             novo.INSERTION_DISTANCE)
            self.assertEqual(len(df.iloc[0]["intervening"]),
                             novo.INSERTION_DISTANCE)

    def test_target_site_window_is_two_sided(self):
        df = novo.predict_insertion_from_spacer(self.GUIDE, self.genome)
        self.assertEqual(len(df.iloc[0]["target_site"]),
                         2 * novo.TARGET_SITE_WINDOW)

    def test_pam_is_always_reported_on_the_top_strand(self):
        """Stated in the source as 'always list top strand' -- so for a minus
        hit the PAM is the genome as written, not its complement."""
        rc_guide = hp.reverse_complement(self.GUIDE)
        rng = random.Random(11)
        filler = "".join(rng.choice("ACGT") for _ in range(3000))
        start = 1500
        genome = filler[:start] + rc_guide + filler[start + len(rc_guide):]
        row = novo.predict_insertion_from_spacer(self.GUIDE, genome).iloc[0]
        self.assertEqual(row["PAM"], genome[start - 3:start])

    def test_only_the_first_hit_is_used(self):
        """It reads grna[0], and map_guide_to_genome lists plus-strand hits
        first, so a guide present on both strands is treated as plus."""
        both = self.GUIDE + "TTTT" + hp.reverse_complement(self.GUIDE)
        rng = random.Random(3)
        filler = "".join(rng.choice("ACGT") for _ in range(3000))
        genome = filler[:800] + both + filler[800 + len(both):]
        row = novo.predict_insertion_from_spacer(self.GUIDE, genome).iloc[0]
        self.assertEqual(row["gRNA_strand"], "+")
        self.assertEqual(row["proto_start"], 800)

    def test_still_reexported_from_postprocess_helpers(self):
        """Older notebooks call hp.predict_insertion_from_spacer. The alias has
        to keep working until those are updated."""
        self.assertIs(hp.predict_insertion_from_spacer,
                      novo.predict_insertion_from_spacer)


if __name__ == "__main__":
    unittest.main()
