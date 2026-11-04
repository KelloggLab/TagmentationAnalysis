"""Unit tests for the pure functions in `core.postprocessHelpers`.

These are the functions the architecture plan moves first (Phase 0: into
`ngslib/sequence.py`, `motifs.py` and `targets.py`), precisely because they do
no I/O and their answers can be worked out by hand. Every expected value below
is derived independently, not copied from a previous run -- except where a test
says it is locking in current behaviour, which is noted explicitly.
"""

import math
import unittest

import pandas as pd

from core import postprocessHelpers as hp


class ReverseComplementTest(unittest.TestCase):
    def test_known_pairs(self):
        self.assertEqual(hp.reverse_complement("ACGT"), "ACGT")
        self.assertEqual(hp.reverse_complement("AAAA"), "TTTT")
        self.assertEqual(hp.reverse_complement("GATTACA"), "TGTAATC")
        self.assertEqual(hp.reverse_complement(""), "")

    def test_is_its_own_inverse(self):
        seq = "ACGTTGCATCAGGTCCTTAAGGCAT"
        self.assertEqual(hp.reverse_complement(hp.reverse_complement(seq)), seq)


class AtRichnessTest(unittest.TestCase):
    def test_simple_fractions(self):
        self.assertEqual(hp.at_richness("AAAA"), 1.0)
        self.assertEqual(hp.at_richness("GGCC"), 0.0)
        self.assertEqual(hp.at_richness("ATGC"), 0.5)
        self.assertAlmostEqual(hp.at_richness("AATGC"), 3 / 5)

    def test_uracil_counts_as_thymine(self):
        self.assertEqual(hp.at_richness("AUGC"), 0.5)

    def test_gaps_and_whitespace_are_stripped(self):
        self.assertEqual(hp.at_richness("AT-GC"), 0.5)
        self.assertEqual(hp.at_richness("AT GC\n"), 0.5)

    def test_ambiguous_bases_excluded_by_default(self):
        # N is dropped, so the denominator is 4, not 5.
        self.assertEqual(hp.at_richness("ATGCN"), 0.5)

    def test_ambiguous_bases_included_when_asked(self):
        # Denominator becomes 5; N is not an A/T, so 2/5.
        self.assertAlmostEqual(hp.at_richness("ATGCN", ignore_ambiguous=False), 2 / 5)

    def test_empty_and_all_ambiguous_return_zero(self):
        self.assertEqual(hp.at_richness(""), 0.0)
        self.assertEqual(hp.at_richness("NNNN"), 0.0)

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            hp.at_richness(None)


class SlidingWindowTest(unittest.TestCase):
    def test_window_chunks(self):
        self.assertEqual(hp.sliding_window("ABCDE", 2),
                         ["AB", "BC", "CD", "DE"])

    def test_window_equal_to_length(self):
        self.assertEqual(hp.sliding_window("ACGT", 4), ["ACGT"])

    def test_window_longer_than_sequence_is_empty(self):
        self.assertEqual(hp.sliding_window("ACG", 5), [])

    def test_at_sliding_window_matches_manual_calculation(self):
        # windows: AAT=1.0, ATG=2/3, TGC=1/3
        got = hp.at_sliding_window("AATGC", 3)
        self.assertEqual(len(got), 3)
        self.assertAlmostEqual(got[0], 1.0)
        self.assertAlmostEqual(got[1], 2 / 3)
        self.assertAlmostEqual(got[2], 1 / 3)


class MapGuideToGenomeTest(unittest.TestCase):
    # guide ACGTACGT sits at 10 on the plus strand; its reverse complement
    # sits at 30, which is a minus-strand hit.
    GUIDE = "ACGTACGT"
    GENOME = ("TTTTTTTTTT"            # 0-9
              "ACGTACGT"              # 10-17  plus hit
              "GGGGGGGGGGGG"         # 18-29
              "ACGTACGT"              # 30-37  this is rc(GUIDE) == GUIDE here
              "CCCCCCCCCC")           # 38-47

    def test_finds_plus_strand_hit(self):
        hits = hp.map_guide_to_genome("TTTTTTTTTT", self.GENOME)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["start"], 0)
        self.assertEqual(hits[0]["end"], 10)
        self.assertEqual(hits[0]["strand"], "+")
        self.assertEqual(hits[0]["sequence"], "TTTTTTTTTT")

    def test_finds_minus_strand_hit(self):
        # GGGG... on the plus strand means CCCC... is a minus-strand match.
        hits = hp.map_guide_to_genome("GGGGGGGGGG", self.GENOME)
        strands = sorted(h["strand"] for h in hits)
        self.assertIn("+", strands)
        self.assertIn("-", strands)

    def test_case_is_normalised(self):
        upper = hp.map_guide_to_genome("acgtacgt", self.GENOME)
        self.assertTrue(upper)
        self.assertEqual(upper[0]["sequence"], "ACGTACGT")

    def test_no_hit_returns_empty_list(self):
        self.assertEqual(hp.map_guide_to_genome("AAAAAAAAAAAA", self.GENOME), [])

    def test_minus_hit_sequence_is_the_plus_strand_slice(self):
        """Documented quirk (see docs/API.md): for a '-' hit, `sequence` is the
        genome as written, i.e. the reverse complement of the guide -- not the
        guide itself. Anything consuming `sequence` has to know this."""
        genome = "AAAA" + hp.reverse_complement("GATTACAGATTACAGATTA") + "AAAA"
        guide = "GATTACAGATTACAGATTA"
        minus = [h for h in hp.map_guide_to_genome(guide, genome)
                 if h["strand"] == "-"]
        self.assertEqual(len(minus), 1)
        self.assertEqual(minus[0]["sequence"], hp.reverse_complement(guide))
        self.assertNotEqual(minus[0]["sequence"], guide)

    def test_overlapping_matches_are_all_reported(self):
        # 'ATAT' occurs at 0, 2 and 4 in 'ATATATAT' -- the scan steps by 1.
        hits = hp.map_guide_to_genome("ATAT", "ATATATAT")
        plus = sorted(h["start"] for h in hits if h["strand"] == "+")
        self.assertEqual(plus, [0, 2, 4])


class ExtractSequenceWindowsTest(unittest.TestCase):
    GENOME = "ACGT" * 25  # 100 bp

    def test_window_is_symmetric_and_inclusive_of_centre(self):
        w, = hp.extract_sequence_windows([50], self.GENOME, 5)
        self.assertEqual(w["position"], 50)
        self.assertEqual(w["start"], 45)
        self.assertEqual(w["end"], 56)          # pos + window + 1
        self.assertEqual(len(w["sequence"]), 11)
        self.assertEqual(w["sequence"], self.GENOME[45:56])

    def test_window_is_clipped_at_the_left_edge(self):
        w, = hp.extract_sequence_windows([2], self.GENOME, 5)
        self.assertEqual((w["start"], w["end"]), (0, 8))
        self.assertEqual(len(w["sequence"]), 8)

    def test_window_is_clipped_at_the_right_edge(self):
        w, = hp.extract_sequence_windows([98], self.GENOME, 5)
        self.assertEqual((w["start"], w["end"]), (93, 100))

    def test_out_of_range_positions_are_skipped_silently(self):
        got = hp.extract_sequence_windows([-1, 50, 100, 10_000], self.GENOME, 3)
        self.assertEqual([w["position"] for w in got], [50])


class DegenerateMatchingTest(unittest.TestCase):
    def test_exact_match_has_no_mismatches(self):
        self.assertEqual(hp.degenerate_hamming("ACGT", "ACGT"), 0)

    def test_n_matches_anything(self):
        self.assertEqual(hp.degenerate_hamming("ACGT", "NNNN"), 0)

    def test_counts_each_mismatch(self):
        self.assertEqual(hp.degenerate_hamming("ACGT", "ACGA"), 1)
        self.assertEqual(hp.degenerate_hamming("ACGT", "TGCA"), 4)

    def test_iupac_classes(self):
        # R = A/G, Y = C/T, W = A/T, S = C/G
        self.assertEqual(hp.degenerate_hamming("AG", "RR"), 0)
        self.assertEqual(hp.degenerate_hamming("CT", "YY"), 0)
        self.assertEqual(hp.degenerate_hamming("AT", "WW"), 0)
        self.assertEqual(hp.degenerate_hamming("CG", "SS"), 0)
        # R is A/G, so both bases of "CT" miss, while "CG" misses only once.
        self.assertEqual(hp.degenerate_hamming("CT", "RR"), 2)
        self.assertEqual(hp.degenerate_hamming("CG", "RR"), 1)

    def test_ttn_style_pam(self):
        # The CAST-relevant case: TTN matches TTA/TTC/TTG/TTT, not CTA.
        for base in "ACGT":
            self.assertEqual(hp.degenerate_hamming("TT" + base, "TTN"), 0)
        self.assertEqual(hp.degenerate_hamming("CTA", "TTN"), 1)

    def test_uracil_is_treated_as_thymine(self):
        self.assertEqual(hp.degenerate_hamming("ACGU", "ACGT"), 0)

    def test_match_count_is_the_complement_of_hamming(self):
        self.assertEqual(hp.degenerate_match_count("ACGT", "ACGA"), 3)
        self.assertEqual(hp.degenerate_match_count("ACGT", "ACGT"), 4)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            hp.degenerate_hamming("ACG", "ACGT")

    def test_unknown_code_raises(self):
        with self.assertRaises(ValueError):
            hp.degenerate_hamming("ACGT", "ACGZ")

    def test_scan_returns_best_hit_first(self):
        hits = hp.scan_degenerate_hamming("AAAACGTAAAA", "ACGT")
        self.assertEqual(hits[0][1], 0)          # zero mismatches
        self.assertEqual(hits[0][0], 3)          # at index 3
        self.assertEqual(hits[0][2], "ACGT")

    def test_scan_covers_every_window(self):
        seq, motif = "ACGTACGT", "ACG"
        self.assertEqual(len(hp.scan_degenerate_hamming(seq, motif)),
                         len(seq) - len(motif) + 1)


class PwmTest(unittest.TestCase):
    """Worked by hand, so the arithmetic itself is under test."""

    def setUp(self):
        # One position, strongly A. Counts A=9, C=0, G=0, T=0.
        self.counts = pd.DataFrame([[9, 0, 0, 0]], columns=list("ACGT"))

    def test_probabilities_are_laplace_smoothed(self):
        _, prob = hp.build_log_odds_pwm(self.counts)
        # (9+1) / (9+1+1+1+1) = 10/13 for A, 1/13 for the others.
        self.assertAlmostEqual(prob.loc[0, "A"], 10 / 13)
        self.assertAlmostEqual(prob.loc[0, "C"], 1 / 13)
        self.assertAlmostEqual(prob.sum(axis=1).iloc[0], 1.0)

    def test_log_odds_against_uniform_background(self):
        pwm, _ = hp.build_log_odds_pwm(self.counts)
        self.assertAlmostEqual(pwm.loc[0, "A"], math.log2((10 / 13) / 0.25))
        self.assertAlmostEqual(pwm.loc[0, "T"], math.log2((1 / 13) / 0.25))

    def test_zero_pseudocount_gives_a_flat_pwm_for_flat_counts(self):
        flat = pd.DataFrame([[5, 5, 5, 5]], columns=list("ACGT"))
        pwm, _ = hp.build_log_odds_pwm(flat, pseudocount=0.0)
        for base in "ACGT":
            self.assertAlmostEqual(pwm.loc[0, base], 0.0)

    def test_non_uniform_background(self):
        bg = {"A": 0.4, "C": 0.1, "G": 0.1, "T": 0.4}
        pwm, _ = hp.build_log_odds_pwm(self.counts, background=bg)
        self.assertAlmostEqual(pwm.loc[0, "A"], math.log2((10 / 13) / 0.4))

    def test_background_must_cover_the_alphabet(self):
        with self.assertRaises(ValueError):
            hp.build_log_odds_pwm(self.counts, background={"A": 0.5, "C": 0.5})

    def test_background_must_be_positive(self):
        with self.assertRaises(ValueError):
            hp.build_log_odds_pwm(
                self.counts, background={"A": 0.5, "C": 0.5, "G": 0.0, "T": 0.0})

    def test_natural_log_base(self):
        pwm, _ = hp.build_log_odds_pwm(self.counts, log_base=math.e)
        self.assertAlmostEqual(pwm.loc[0, "A"], math.log((10 / 13) / 0.25))

    def test_score_is_the_sum_over_positions(self):
        counts = pd.DataFrame([[9, 0, 0, 0], [0, 9, 0, 0]], columns=list("ACGT"))
        pwm, _ = hp.build_log_odds_pwm(counts)
        self.assertAlmostEqual(hp.score_sequence_pwm("AC", pwm),
                               pwm.loc[0, "A"] + pwm.loc[1, "C"])

    def test_consensus_scores_higher_than_anything_else(self):
        counts = pd.DataFrame([[9, 0, 0, 0], [0, 9, 0, 0]], columns=list("ACGT"))
        pwm, _ = hp.build_log_odds_pwm(counts)
        best = hp.score_sequence_pwm("AC", pwm)
        for other in ["AA", "CC", "TT", "GA"]:
            self.assertGreater(best, hp.score_sequence_pwm(other, pwm))

    def test_wrong_length_raises(self):
        pwm, _ = hp.build_log_odds_pwm(self.counts)
        with self.assertRaises(ValueError):
            hp.score_sequence_pwm("AC", pwm)

    def test_unknown_base_raises_by_default(self):
        pwm, _ = hp.build_log_odds_pwm(self.counts)
        with self.assertRaises(ValueError):
            hp.score_sequence_pwm("N", pwm)

    def test_unknown_base_penalty_is_applied(self):
        pwm, _ = hp.build_log_odds_pwm(self.counts)
        self.assertAlmostEqual(
            hp.score_sequence_pwm("N", pwm, unknown_penalty=-2.5), -2.5)

    def test_max_possible_score_is_the_row_maxima(self):
        counts = pd.DataFrame([[9, 0, 0, 0], [0, 9, 0, 0]], columns=list("ACGT"))
        pwm, _ = hp.build_log_odds_pwm(counts)
        self.assertAlmostEqual(hp.pwm_max_possible_score(pwm),
                               hp.score_sequence_pwm("AC", pwm))


if __name__ == "__main__":
    unittest.main()
