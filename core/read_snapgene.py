"""Read a SnapGene `.dna` file.

The format is a sequence of blocks: one type byte, a big-endian uint32 length,
then that many bytes. Block 0 holds the sequence, after one flags byte whose
low bit says whether the construct is circular; block 10 holds the features as
XML.

`read_dna` is the library entry point and returns the parsed contents.
Running the module as a script writes them out as FASTA plus a feature table:

    python3 -I core/read_snapgene.py <in.dna> <out_prefix>
"""
from __future__ import annotations

import struct
import sys
import xml.etree.ElementTree as ET

#: Bases per line in the FASTA this writes.
FASTA_WIDTH = 70


def blocks(data):
    """Yield `(block_type, payload)` for each block in a .dna file's bytes."""
    i = 0
    while i + 5 <= len(data):
        btype = data[i]
        (n,) = struct.unpack(">I", data[i + 1:i + 5])
        yield btype, data[i + 5:i + 5 + n]
        i += 5 + n


def read_dna(path):
    """Parse a SnapGene `.dna` file.

    Returns `(sequence, topology, features)`, where `topology` is "circular",
    "linear" or "?" and `features` is a list of dicts with `name`, `type`,
    `start`, `end` and `strand`, sorted by start. Raises `ValueError` if the
    file has no sequence block.
    """
    with open(path, "rb") as f:
        data = f.read()

    seq = None
    topology = "?"
    feats = []
    for btype, payload in blocks(data):
        if btype == 0:
            topology = "circular" if payload[0] & 1 else "linear"
            seq = payload[1:].decode("ascii", "replace").upper()
        elif btype == 10:
            root = ET.fromstring(payload.decode("utf-8", "replace"))
            for fe in root.iter("Feature"):
                segs = [s for s in fe.iter("Segment")]
                rng = [int(x) for s in segs
                       for x in s.get("range", "0-0").split("-")]
                if not rng:
                    continue
                feats.append({
                    "name": fe.get("name", ""),
                    "type": fe.get("type", ""),
                    "start": min(rng),
                    "end": max(rng),
                    "strand": "-" if fe.get("directionality") == "2" else "+",
                })

    if seq is None:
        raise ValueError("no sequence block found")
    feats.sort(key=lambda f: f["start"])
    return seq, topology, feats


def write_fasta(path, seq, name, topology):
    with open(path, "w") as out:
        out.write(f">{name} {len(seq)} bp {topology}\n")
        for i in range(0, len(seq), FASTA_WIDTH):
            out.write(seq[i:i + FASTA_WIDTH] + "\n")


def write_features(path, feats):
    with open(path, "w") as out:
        out.write("start\tend\tstrand\ttype\tname\n")
        for f in feats:
            out.write(f"{f['start']}\t{f['end']}\t{f['strand']}"
                      f"\t{f['type']}\t{f['name']}\n")


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    try:
        seq, topology, feats = read_dna(src)
    except ValueError as exc:
        sys.exit(str(exc))

    write_fasta(prefix + ".fasta", seq, prefix.split("/")[-1], topology)
    write_features(prefix + ".features.tsv", feats)

    print(f"{len(seq)} bp, {topology}, {len(feats)} features")
    for f in feats:
        print(f"{f['start']:>6} {f['end']:>6} {f['strand']}  "
              f"{f['type']:<16} {f['name']}")


if __name__ == "__main__":
    main()
