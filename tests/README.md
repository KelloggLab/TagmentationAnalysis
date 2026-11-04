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

90 tests, about 3 seconds. `pytest tests` also works if you prefer it, but it
is not required and is not in `environment.yml`.

The end-to-end tests need `cutadapt`, `bwa`, `samtools` and `pysam`. If any are
missing the whole class **skips** rather than fails, so the rest still runs:

```
$ PATH=/usr/bin:/bin python -m unittest discover -s tests
OK (skipped=13)
```

## What is here

| File | Covers |
|---|---|
| `make_fixtures.py` | Builds the synthetic dataset. Not a test; run it directly to inspect the fixtures |
| `test_insertion_pipeline.py` | `pipeline/map_junction_insertions.py`, end to end, against planted truth |
| `test_sequence_functions.py` | `postprocessHelpers`: reverse complement, A/T richness, windows, guide mapping, IUPAC matching, PWMs |
| `test_tsv_analysis.py` | `TsvMapHelperFunctions` arithmetic and `NovoCAST_helper` geometry |

## The synthetic dataset

```bash
python tests/make_fixtures.py /tmp/fixtures
```

Writes a 50 kb random single-contig genome (`toy_chr`), 113 reads, and a
`truth.json` saying what the pipeline should report. It is seeded, so it is
byte-identical every time — one of the tests asserts that.

96 of the reads are junction reads built from the genome at six known sites,
including one site planted on both strands and one singleton. Each read is
named `ins_<pos>_<strand>_<n>`, so **every output row is checked individually**
against the position and strand it was built from, not just in aggregate.

The remaining 17 are decoys that should be discarded: 8 with no transposon end,
5 with only 12 bp of genomic flank, and 4 with an unmappable poly-A flank.

## Two things the tests record rather than assert as correct

**One decoy leaks through.** `cutadapt -g SEQ` is unanchored, so it also trims
short partial matches at the 5′ end. One donor-free decoy begins with 3 bases
resembling the end of the donor, gets trimmed, therefore survives
`--discard-untrimmed`, and maps cleanly as ordinary genomic sequence. The test
pins this to one named read so the behaviour is visible. Anchoring the adapter
(`-g ^SEQ`) or requiring a minimum overlap would fix it, and would make that
test fail — correctly.

**Some tests pin quirks rather than ideals**, each flagged in its docstring:
`map_guide_to_genome` returns the plus-strand slice as `sequence` for a minus
hit; `percent_within_distance` calls `len()` so it rejects generators;
`predict_insertion_from_spacer` only ever uses the first hit. These are locked
in because notebooks depend on them, not because they are good designs.

## Adding to this

When a module moves into `ngslib/`, keep the test pointed at the **public
behaviour**, not the import path: change the import, leave the assertions
alone. If an assertion has to change, that is a result change, and it needs a
reason written down in the commit message.

Still uncovered, roughly in order of value:

- `pipeline/filter_donor_backbone.sh` — needs a toy donor GenBank with a
  `Cargo` feature and reads planted from the backbone
- `extract_tsd_centered_motifs_5p`, `characterize_insertion_from_position` /
  `_from_spacer` — the TSD and PAM-distance analysis
- `extract_spacers` and `find_pam_sites_with_optional_protospacer` — including
  the ignored `PAM` argument and the unsorted PAM range noted in `docs/API.md`
- `chipseq/chipseq_pipeline.sh` — nothing, and it should get a fixture before
  the Python port in Phase 3
- `apps/uditas/` — nothing; `extract_salvage.py` is the one worth a fixture
