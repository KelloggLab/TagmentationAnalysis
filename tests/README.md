# Tests

These exist for one reason: so the reorganization in
[../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) can be checked by *"the same
input still gives the same table"*. The plan says to build them before Phase 1,
because that is the only practical way to refactor analysis code without
quietly changing results.

## Running them

Nothing to install — they use the standard library's `unittest`.

```bash
conda activate analyze_tagmentation
python -m unittest discover -s tests -v
```

111 tests, about 18 seconds. Most of that is the backbone-filter file, which
runs bwa once per test to keep the tests independent of each other; the pure
unit tests finish in well under a second. `pytest tests` also works if you
prefer it, but it is not required and is not in `environment.yml`.

The end-to-end tests need `cutadapt`, `bwa`, `samtools` and `pysam`. If any are
missing those classes **skip** rather than fail, so the rest still runs:

```
$ PATH=/usr/bin:/bin python -m unittest discover -s tests
OK (skipped=33)
```

## What is here

| File | Tests | Covers |
|---|--:|---|
| `make_fixtures.py` | — | Builds the synthetic dataset. Not a test; run it directly to inspect the fixtures |
| `test_sequence_functions.py` | 47 | `postprocessHelpers`: reverse complement, A/T richness, windows, guide mapping, IUPAC matching, PWMs |
| `test_tsv_analysis.py` | 27 | `TsvMapHelperFunctions` arithmetic and `NovoCAST_helper` geometry |
| `test_insertion_pipeline.py` | 16 | `pipeline/map_junction_insertions.py`, end to end, against planted truth |
| `test_backbone_filter.py` | 21 | `pipeline/filter_donor_backbone.sh`: what it drops, what it spares, caching, failure modes |

## The synthetic dataset

```bash
python tests/make_fixtures.py /tmp/fixtures
```

Writes a 50 kb random single-contig genome (`toy_chr`), a 1.5 kb donor plasmid
(`toy_donor.gb`, Cargo 1..600 and backbone 601..1500), 141 reads, and a
`truth.json` saying what the pipeline should report. It is seeded, so it is
byte-identical every time — one of the tests asserts that.

124 of the reads are junction reads built from the genome at nine known sites,
including one site planted on both strands and one singleton. Each read is
named `ins_<pos>_<strand>_<n>`, so **every output row is checked individually**
against the position and strand it was built from, not just in aggregate.

The remaining 17 are decoys that should all be discarded: 8 with no transposon
end, 5 with only 12 bp of genomic flank, and 4 with an unmappable poly-A flank.

### How the donor-backbone filter is made testable

The filter only ever sees reads that already mapped to the genome, so a read
has to be *both* genomic-mapping and donor-matching to exercise it. The fixture
arranges that by splicing two 400 bp stretches of the donor into the toy
genome — one from the backbone at 25,000 and one from the Cargo at 35,000 —
which mimics a pre-existing donor integration in the reference strain.

Four kinds of site then exist, and `truth.json` labels each one with its
`origin`:

| Origin | Flank | Expected |
|---|---|---|
| `genome` | unique genomic DNA | always kept |
| `backbone` | 70 bp of donor backbone | dropped |
| `backbone_partial` | 40 bp of backbone, then genome | dropped at `-m 30`, kept at `-m 50` |
| `cargo` | 70 bp of donor Cargo | **kept** — deliberately not screened |

The Cargo row is the important one: a genuine new integration reads from the
Cargo right up to the transposon end, so screening against the whole plasmid
would throw away real signal. The partial row pins the `-m` threshold from
both sides; bwa aligns those reads as `40M30S`, so 40 aligned bases is what
the threshold is compared against.

## A behaviour these tests changed

Writing them turned up a real leak. `cutadapt -g SEQ` is unanchored and
defaults to `-O 3`, so a read merely *beginning* with three bases resembling
the end of the donor was "trimmed", survived `--discard-untrimmed`, and was
reported as an insertion despite never containing a junction. One of the eight
donor-free decoys did exactly that.

`map_junction_insertions.py` now passes `-O len(donor_seq)` by default, so the
whole transposon end has to be present, and exposes `--min-overlap` to relax
it if reads genuinely start part-way into the donor. Two tests hold this down:
one asserts no decoy survives at the default, and one asserts the leak comes
back at `--min-overlap 3`, so it is clear the default is doing the work.

## Some tests pin quirks rather than ideals

Each is flagged in its docstring: `map_guide_to_genome` returns the plus-strand
slice as `sequence` for a minus hit; `percent_within_distance` calls `len()` so
it rejects generators; `predict_insertion_from_spacer` only ever uses the first
hit. These are locked in because notebooks depend on them, not because they are
good designs.

`test_backbone_filter.ToolResolutionTest` is narrower than it looks: it exists
because `find_tool` used bash 4's `${name^^}`, which is a syntax error on the
bash 3.2 that macOS ships — so the *error message* used to fail with "bad
substitution". That test keeps it fixed.

## Adding to this

When a module moves into `ngslib/`, keep the test pointed at the **public
behaviour**, not the import path: change the import, leave the assertions
alone. If an assertion has to change, that is a result change, and it needs a
reason written down in the commit message.

Still uncovered, roughly in order of value:

- `extract_tsd_centered_motifs_5p`, `characterize_insertion_from_position` /
  `_from_spacer` — the TSD and PAM-distance analysis
- `extract_spacers` and `find_pam_sites_with_optional_protospacer` — including
  the ignored `PAM` argument and the unsorted PAM range noted in `docs/API.md`
- `scripts/plot_profile.sh` and the plotting functions — hard to assert on, but
  "it runs and writes a non-empty file" would already catch a lot
- `chipseq/chipseq_pipeline.sh` — nothing, and it should get a fixture before
  the Python port in Phase 3
- `apps/uditas/` — nothing; `extract_salvage.py` is the one worth a fixture,
  and it needs a synthetic CBCL writer
