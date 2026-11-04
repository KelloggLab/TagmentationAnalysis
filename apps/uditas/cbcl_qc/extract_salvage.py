"""Pull the salvageable junction reads straight out of the CBCL files.

Keeps reads with no Tn5 mosaic end near the primer, trims the primer and the
transposon end that follows it, and writes one gzipped FASTQ per biological
sample and primer arm, with the UMI in the read name. Also tallies distinct
flanking sequences, UMI-collapsed, which is an insertion-site count that needs
no reference genome.

usage: python3 -I extract_salvage.py <BaseCalls/L001> <ntiles> <outdir>
"""
import gzip
import json
import os
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

BASES = np.frombuffer(b"ACGT", dtype=np.uint8)

# -I drops the script's directory from sys.path; add just that one back.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runconfig  # noqa: E402

CFG = runconfig.load()

I1_START, I1_LEN = CFG.cycles.i1_start, CFG.cycles.i1_len
I2_START, I2_LEN = CFG.cycles.i2_start, CFG.cycles.i2_len
R2_START, N_R2 = CFG.cycles.r2_start, CFG.cycles.r2_len
UMI_LEN = CFG.cycles.umi_len

# 293T and FT are the two biological samples; the two i5 indices are index
# pairs of the same sample, so they pool. One i7 reads out of the RE end, the
# other out of the LE end.
SAMPLES = [(s.sample, s.arm, s.i5, s.i7) for s in CFG.samples]
PRIMER = {name: a.primer for name, a in CFG.arms.items()}
# Bases of transposon end between the primer's 3' end and the terminus,
# measured as the point where per-position agreement falls away.
END_LEN = {name: a.end_len for name, a in CFG.arms.items()}
DONOR = {name: a.donor for name, a in CFG.arms.items()}
ME = CFG.mosaic_end
MIN_FLANK = CFG.min_flank
SITE_KEY = CFG.site_key

# How much of the donor past the terminus to compare a flank against. The
# comparison only runs on arms where there is enough of it; see
# runconfig.Arm.still_donor_active.
STILL_DONOR_PROBE = 20


def rc(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


PROBES = (ME[:15], rc(ME)[:15])


def read_header(f):
    _v, header_size, _bb, _bq, nbins = struct.unpack("<HIBBI", f.read(12))
    raw = f.read(8 * nbins)
    pairs = struct.unpack("<" + "II" * nbins, raw)
    qmap = {pairs[2 * i]: pairs[2 * i + 1] for i in range(nbins)}
    (ntiles,) = struct.unpack("<I", f.read(4))
    tiles = [struct.unpack("<IIII", f.read(16)) for _ in range(ntiles)]
    return header_size, tiles, qmap


def decode_tile(path, tile_idx, want_qual):
    with open(path, "rb") as f:
        header_size, tiles, qmap = read_header(f)
        offset = header_size + sum(t[3] for t in tiles[:tile_idx])
        _tile, nclust, _usize, csize = tiles[tile_idx]
        f.seek(offset)
        raw = np.frombuffer(gzip.decompress(f.read(csize)), dtype=np.uint8)
    nib = np.empty(raw.size * 2, dtype=np.uint8)
    nib[0::2] = raw & 0x0F
    nib[1::2] = raw >> 4
    nib = nib[:nclust]
    seq = BASES[nib & 0x3].copy()
    qbin = nib >> 2
    seq[qbin == 0] = ord("N")
    if not want_qual:
        return seq, None
    lut = np.full(4, 33, dtype=np.uint8)
    for b, q in qmap.items():
        if b < 4:
            lut[b] = 33 + min(q, 60)
    return seq, lut[qbin]


def read_block(lane_dir, start, length, tile_idx, want_qual=False):
    cols = [decode_tile(f"{lane_dir}/C{c}.1/L001_1.cbcl", tile_idx, want_qual)
            for c in range(start, start + length)]
    bases = np.stack([c[0] for c in cols], axis=1)
    quals = np.stack([c[1] for c in cols], axis=1) if want_qual else None
    return bases, quals


def one_mismatch_map(barcodes):
    exact = set(barcodes)
    m = {bc: bc for bc in barcodes}
    ambiguous = set()
    for bc in barcodes:
        for i in range(len(bc)):
            for b in "ACGTN":
                v = bc[:i] + b + bc[i + 1:]
                if v in m and m[v] != bc:
                    ambiguous.add(v)
                else:
                    m.setdefault(v, bc)
    for v in ambiguous - exact:
        m.pop(v, None)
    return m


def mism(a, b):
    return sum(1 for x, y in zip(a, b) if x != y and x != "N" and y != "N")


def main():
    lane_dir, ntiles, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    i7_map = one_mismatch_map([rc(i7) for _, _, _, i7 in SAMPLES])
    i5_map = one_mismatch_map([rc(i5) for _, _, i5, _ in SAMPLES])
    combo = {(rc(i7), rc(i5)): (samp, arm) for samp, arm, i5, i7 in SAMPLES}
    groups = sorted({(s, a) for s, a, _, _ in SAMPLES})

    handles = {g: gzip.open(os.path.join(outdir, f"{g[0]}_{g[1]}.fastq.gz"), "wt")
               for g in groups}
    sites = {g: defaultdict(set) for g in groups}   # flank key -> set of UMIs
    stat = {g: Counter() for g in groups}

    for t in range(ntiles):
        i1, _ = read_block(lane_dir, I1_START, I1_LEN, t)
        i2, _ = read_block(lane_dir, I2_START, I2_LEN, t)
        r2, q2 = read_block(lane_dir, R2_START, N_R2, t, want_qual=True)
        print(f"tile {t}: {i1.shape[0]} clusters", file=sys.stderr)

        for k in range(i1.shape[0]):
            bc7 = i7_map.get(i1[k].tobytes().decode())
            bc5 = i5_map.get(i2[k, UMI_LEN:].tobytes().decode())
            if bc7 is None or bc5 is None:
                continue
            g = combo.get((bc7, bc5))
            if g is None:
                continue
            s = stat[g]
            s["assigned"] += 1
            seq = r2[k].tobytes().decode()

            # Salvageable = the adapter is not sitting right next to the primer.
            cut = len(seq)
            for pr in PROBES:
                p = seq.find(pr)
                if p != -1:
                    cut = min(cut, p)
            arm = g[1]
            start = len(PRIMER[arm]) + END_LEN[arm]
            if cut < start + MIN_FLANK:
                s["adapter_too_close"] += 1
                continue

            flank = seq[start:cut]
            n_pos = flank.find("N")
            if n_pos != -1:
                flank = flank[:n_pos]
            if len(flank) < MIN_FLANK:
                s["too_short_after_trim"] += 1
                continue
            # Still donor sequence past the terminus? Then it is not a junction.
            spec = CFG.arm(arm)
            tail = spec.donor_past_terminus[:STILL_DONOR_PROBE]
            if spec.still_donor_active and mism(flank[:len(tail)], tail) <= 2:
                s["still_donor"] += 1
                continue

            umi = i2[k, :UMI_LEN].tobytes().decode()
            qual = q2[k].tobytes().decode()[start:start + len(flank)]
            name = f"t{t}_{k}_{umi}"
            handles[g].write(f"@{name}\n{flank}\n+\n{qual}\n")
            sites[g][flank[:SITE_KEY]].add(umi)
            s["kept"] += 1

    for h in handles.values():
        h.close()

    summary = {"lane": lane_dir, "tiles": ntiles, "groups": {}}
    print(f"\n{'group':<10} {'assigned':>9} {'kept':>8} {'%kept':>6} "
          f"{'sites':>8} {'UMI-collapsed':>14} {'top site %':>11}")
    for g in groups:
        s, sd = stat[g], sites[g]
        kept = s["kept"]
        collapsed = sum(len(v) for v in sd.values())
        top = max((len(v) for v in sd.values()), default=0)
        key = f"{g[0]}_{g[1]}"
        summary["groups"][key] = {
            "assigned": s["assigned"], "kept": kept,
            "adapter_too_close": s["adapter_too_close"],
            "still_donor": s["still_donor"],
            "too_short_after_trim": s["too_short_after_trim"],
            "distinct_sites": len(sd), "umi_collapsed": collapsed,
            "top_sites": [
                {"flank": f, "umis": len(u), "reads": None}
                for f, u in sorted(sd.items(), key=lambda x: -len(x[1]))[:15]
            ],
        }
        print(f"{key:<10} {s['assigned']:>9} {kept:>8} "
              f"{100*kept/max(s['assigned'],1):>5.1f}% {len(sd):>8} "
              f"{collapsed:>14} {100*top/max(collapsed,1):>10.1f}%")

    with open(os.path.join(outdir, "salvage_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {outdir}/*.fastq.gz and salvage_summary.json")


if __name__ == "__main__":
    main()
