"""Tests for `apps/human_profiling/read_snapgene.py`.

It parses SnapGene's binary `.dna` format by hand -- a sequence of blocks, each
a type byte, a big-endian uint32 length, and that many bytes. Block 0 holds the
sequence after one flags byte; block 10 holds the features as XML.

A hand-rolled binary parser with no tests is a liability, and it is easy to
cover: the tests here build `.dna` files byte by byte, so no SnapGene file
needs to be committed.
"""

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "apps" / "human_profiling"
sys.path.insert(0, str(APP))

try:
    import read_snapgene as rs
    IMPORT_ERROR = None
except Exception as exc:                      # pragma: no cover
    IMPORT_ERROR = str(exc)

CIRCULAR, LINEAR = 1, 0


def block(btype, payload):
    return bytes([btype]) + struct.pack(">I", len(payload)) + payload


def sequence_block(seq, topology=CIRCULAR):
    return block(0, bytes([topology]) + seq.encode())


def features_block(features):
    """features: list of (name, type, range_string, directionality)."""
    items = "".join(
        f'<Feature name="{n}" type="{t}" directionality="{d}">'
        f'<Segment range="{r}"/></Feature>'
        for n, t, r, d in features)
    return block(10, f"<Features>{items}</Features>".encode())


def dna_file(path, seq, features=(), topology=CIRCULAR):
    path.write_bytes(block(9, b"SnapGene")
                     + sequence_block(seq, topology)
                     + features_block(features))
    return path


@unittest.skipIf(IMPORT_ERROR, f"cannot import read_snapgene: {IMPORT_ERROR}")
class BlockParsingTest(unittest.TestCase):
    def test_reads_consecutive_blocks(self):
        data = block(0, b"first") + block(10, b"second") + block(5, b"third")
        self.assertEqual(list(rs.blocks(data)),
                         [(0, b"first"), (10, b"second"), (5, b"third")])

    def test_empty_payload_is_allowed(self):
        self.assertEqual(list(rs.blocks(block(7, b""))), [(7, b"")])

    def test_empty_input_yields_nothing(self):
        self.assertEqual(list(rs.blocks(b"")), [])

    def test_a_truncated_trailing_header_is_ignored(self):
        """Fewer than five bytes left cannot be a block header, so the loop
        stops rather than raising."""
        data = block(0, b"ok") + b"\x01\x02"
        self.assertEqual(list(rs.blocks(data)), [(0, b"ok")])

    def test_length_is_big_endian(self):
        payload = b"x" * 258
        data = block(3, payload)
        self.assertEqual(data[1:5], b"\x00\x00\x01\x02")
        self.assertEqual(list(rs.blocks(data)), [(3, payload)])

    def test_a_truncated_payload_comes_back_short_rather_than_raising(self):
        """The slice is unbounded, so a declared length longer than the file
        yields a short payload. Worth knowing: the parser does not validate."""
        data = bytes([0]) + struct.pack(">I", 100) + b"only ten.."
        got = list(rs.blocks(data))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][1], b"only ten..")


@unittest.skipIf(IMPORT_ERROR, f"cannot import read_snapgene: {IMPORT_ERROR}")
class ConversionTest(unittest.TestCase):
    """Run the script and check the two files it writes."""

    SEQ = "acgtacgtaa" * 12          # 120 bp, lower case as SnapGene stores it

    def convert(self, seq=None, features=(), topology=CIRCULAR):
        d = Path(tempfile.mkdtemp())
        src = dna_file(d / "plasmid.dna", seq or self.SEQ, features, topology)
        prefix = d / "out"
        proc = subprocess.run(
            [sys.executable, "-I", str(APP / "read_snapgene.py"),
             str(src), str(prefix)],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return proc, prefix

    def test_sequence_is_written_upper_case(self):
        _, prefix = self.convert()
        lines = (prefix.with_suffix(".fasta")).read_text().splitlines()
        self.assertEqual("".join(lines[1:]), self.SEQ.upper())

    def test_fasta_header_records_length_and_topology(self):
        _, prefix = self.convert()
        header = (prefix.with_suffix(".fasta")).read_text().splitlines()[0]
        self.assertTrue(header.startswith(">out "))
        self.assertIn(f"{len(self.SEQ)} bp", header)
        self.assertIn("circular", header)

    def test_linear_topology_is_reported(self):
        _, prefix = self.convert(topology=LINEAR)
        header = (prefix.with_suffix(".fasta")).read_text().splitlines()[0]
        self.assertIn("linear", header)

    def test_fasta_wraps_at_seventy_columns(self):
        _, prefix = self.convert(seq="A" * 155)
        body = (prefix.with_suffix(".fasta")).read_text().splitlines()[1:]
        self.assertEqual([len(l) for l in body], [70, 70, 15])

    def test_features_are_written_sorted_by_start(self):
        feats = [("late", "misc_feature", "900-950", "1"),
                 ("early", "promoter", "10-40", "1"),
                 ("middle", "CDS", "100-400", "1")]
        _, prefix = self.convert(features=feats)
        rows = (prefix.parent / "out.features.tsv").read_text().splitlines()
        self.assertEqual(rows[0], "start\tend\tstrand\ttype\tname")
        self.assertEqual([r.split("\t")[4] for r in rows[1:]],
                         ["early", "middle", "late"])

    def test_feature_columns_are_parsed(self):
        _, prefix = self.convert(features=[("Cargo", "misc_feature", "1-600", "1")])
        row = (prefix.parent / "out.features.tsv").read_text().splitlines()[1]
        start, end, strand, ftype, name = row.split("\t")
        self.assertEqual((start, end, strand, ftype, name),
                         ("1", "600", "+", "misc_feature", "Cargo"))

    def test_directionality_two_is_the_minus_strand(self):
        _, prefix = self.convert(features=[("rev", "CDS", "10-20", "2")])
        row = (prefix.parent / "out.features.tsv").read_text().splitlines()[1]
        self.assertEqual(row.split("\t")[2], "-")

    def test_range_is_normalised_lowest_to_highest(self):
        """start/end come from min/max over the segment range, so a reversed
        range does not produce an inverted interval."""
        _, prefix = self.convert(features=[("odd", "CDS", "500-100", "1")])
        row = (prefix.parent / "out.features.tsv").read_text().splitlines()[1]
        self.assertEqual(row.split("\t")[:2], ["100", "500"])

    def test_no_features_still_writes_a_header_only_table(self):
        _, prefix = self.convert()
        rows = (prefix.parent / "out.features.tsv").read_text().splitlines()
        self.assertEqual(rows, ["start\tend\tstrand\ttype\tname"])

    def test_summary_line_counts_bases_and_features(self):
        proc, _ = self.convert(features=[("a", "CDS", "1-10", "1"),
                                         ("b", "CDS", "20-30", "1")])
        self.assertIn(f"{len(self.SEQ)} bp", proc.stdout)
        self.assertIn("circular", proc.stdout)
        self.assertIn("2 features", proc.stdout)

    def test_missing_sequence_block_is_a_clear_error(self):
        d = Path(tempfile.mkdtemp())
        src = d / "broken.dna"
        src.write_bytes(block(9, b"SnapGene") + features_block([]))
        proc = subprocess.run(
            [sys.executable, "-I", str(APP / "read_snapgene.py"),
             str(src), str(d / "out")],
            capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no sequence block", proc.stderr)


if __name__ == "__main__":
    unittest.main()
