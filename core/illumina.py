"""Read Illumina base calls straight out of a run folder's `.cbcl` files.

A CBCL file holds one cycle for one surface of one lane, as a header followed
by per-tile gzip members. Each cluster is packed into a nibble: the low two
bits are the base, the high two the quality bin, and bin 0 means no-call.

This is the only CBCL reader in the repository. It was previously copied, with
small variations, into three of the `apps/human_profiling` scripts; those
copies had drifted in what they returned and in how they located a tile.

Only numpy is imported, so the `python3 -I` scripts can use it without pulling
in pandas, matplotlib or Biopython.
"""
from __future__ import annotations

import gzip
import struct

import numpy as np

#: Nibble value -> base, for the low two bits.
BASES = np.frombuffer(b"ACGT", dtype=np.uint8)

#: Quality bin 0 is a no-call, and decodes to this character.
NO_CALL = ord("N")


def cbcl_path(lane_dir, cycle, surface=1):
    """Path to one cycle's CBCL within a lane directory."""
    return f"{lane_dir}/C{cycle}.{surface}/L001_{surface}.cbcl"


def read_header(f):
    """Parse a CBCL header from an open binary file.

    Returns `(header_size, tiles, qmap)`, where `tiles` is a list of
    `(tile_number, n_clusters, uncompressed_size, compressed_size)` and `qmap`
    maps quality bin -> Phred score. Leaves the file positioned after the
    header's tile table, so seek before reading tile data.
    """
    _version, header_size, _bits_base, _bits_qual, nbins = struct.unpack(
        "<HIBBI", f.read(12))
    pairs = struct.unpack("<" + "II" * nbins, f.read(8 * nbins))
    qmap = {pairs[2 * i]: pairs[2 * i + 1] for i in range(nbins)}
    (ntiles,) = struct.unpack("<I", f.read(4))
    tiles = [struct.unpack("<IIII", f.read(16)) for _ in range(ntiles)]
    return header_size, tiles, qmap


def _unpack(raw, nclust, qmap, want_qual):
    """Nibble-packed bytes -> (bases, quals or None), both uint8 arrays."""
    nib = np.empty(raw.size * 2, dtype=np.uint8)
    nib[0::2] = raw & 0x0F
    nib[1::2] = raw >> 4
    nib = nib[:nclust]
    seq = BASES[nib & 0x3].copy()
    qbin = nib >> 2
    seq[qbin == 0] = NO_CALL
    if not want_qual:
        return seq, None
    lut = np.full(4, 33, dtype=np.uint8)
    for b, q in qmap.items():
        if b < 4:
            lut[b] = 33 + min(q, 60)
    return seq, lut[qbin]


def decode_tile(path, tile_idx, want_qual=False):
    """Decode one tile of one cycle. Returns `(bases, quals or None)`."""
    with open(path, "rb") as f:
        header_size, tiles, qmap = read_header(f)
        offset = header_size + sum(t[3] for t in tiles[:tile_idx])
        _tile, nclust, _usize, csize = tiles[tile_idx]
        f.seek(offset)
        raw = np.frombuffer(gzip.decompress(f.read(csize)), dtype=np.uint8)
    return _unpack(raw, nclust, qmap, want_qual)


def decode_tiles(path, ntiles, want_qual=False):
    """Decode the first `ntiles` tiles of one cycle, concatenated.

    Equivalent to concatenating `decode_tile` over `range(ntiles)`, which
    `tests/test_uditas_cbcl.py` asserts directly.
    """
    seqs, quals = [], []
    for idx in range(ntiles):
        s, q = decode_tile(path, idx, want_qual)
        seqs.append(s)
        if want_qual:
            quals.append(q)
    return (np.concatenate(seqs),
            np.concatenate(quals) if want_qual else None)


def read_tile(lane_dir, start, length, tile_idx, want_qual=False, surface=1):
    """Cycles `start .. start+length-1` for one tile, as a clusters x cycles
    array. Returns `(bases, quals or None)`."""
    cols = [decode_tile(cbcl_path(lane_dir, c, surface), tile_idx, want_qual)
            for c in range(start, start + length)]
    return (np.stack([c[0] for c in cols], axis=1),
            np.stack([c[1] for c in cols], axis=1) if want_qual else None)


def read_tiles(lane_dir, start, length, ntiles, want_qual=False, surface=1):
    """As `read_tile`, but over the first `ntiles` tiles stacked together."""
    cols = [decode_tiles(cbcl_path(lane_dir, c, surface), ntiles, want_qual)
            for c in range(start, start + length)]
    return (np.stack([c[0] for c in cols], axis=1),
            np.stack([c[1] for c in cols], axis=1) if want_qual else None)
