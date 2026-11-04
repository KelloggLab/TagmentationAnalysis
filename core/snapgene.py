"""Read a SnapGene .dna file into FASTA plus a feature table.

The format is a sequence of blocks: one type byte, a big-endian uint32 length,
then that many bytes. Block 0 holds the sequence (after one flags byte) and
block 10 holds the features as XML.

usage: python3 -I core/snapgene.py <in.dna> <out_prefix>
"""
import struct
import sys
import xml.etree.ElementTree as ET


def blocks(data):
    i = 0
    while i + 5 <= len(data):
        btype = data[i]
        (n,) = struct.unpack(">I", data[i + 1:i + 5])
        yield btype, data[i + 5:i + 5 + n]
        i += 5 + n


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        data = f.read()

    seq = None
    feats = []
    topology = "?"
    for btype, payload in blocks(data):
        if btype == 0:
            topology = "circular" if payload[0] & 1 else "linear"
            seq = payload[1:].decode("ascii", "replace").upper()
        elif btype == 10:
            root = ET.fromstring(payload.decode("utf-8", "replace"))
            for fe in root.iter("Feature"):
                segs = [s for s in fe.iter("Segment")]
                rng = [int(x) for s in segs for x in s.get("range", "0-0").split("-")]
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
        sys.exit("no sequence block found")
    with open(prefix + ".fasta", "w") as out:
        out.write(f">{prefix.split('/')[-1]} {len(seq)} bp {topology}\n")
        for i in range(0, len(seq), 70):
            out.write(seq[i:i + 70] + "\n")

    feats.sort(key=lambda f: f["start"])
    with open(prefix + ".features.tsv", "w") as out:
        out.write("start\tend\tstrand\ttype\tname\n")
        for f in feats:
            out.write(f"{f['start']}\t{f['end']}\t{f['strand']}\t{f['type']}\t{f['name']}\n")

    print(f"{len(seq)} bp, {topology}, {len(feats)} features")
    for f in feats:
        print(f"{f['start']:>6} {f['end']:>6} {f['strand']}  {f['type']:<16} {f['name']}")


if __name__ == "__main__":
    main()
