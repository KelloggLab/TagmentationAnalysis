# Reorganization: what was done, what is left

The layout this plan aimed at is now the actual layout; see
[ARCHITECTURE.md](ARCHITECTURE.md). This file is kept only as the record of
the move and the short list of things it did not finish.

## Done

| Was | Now |
|---|---|
| `seqsuite/` | `core/` |
| `pipeline/` | `apps/ecoli_profiling/` |
| `scripts/plot_profile.sh` | `apps/ecoli_profiling/plot_profile.sh` |
| `chipseq/` | `apps/chipseq/` |
| `apps/uditas/`, with its `cbcl_qc/` and `salvage/` subfolders | `apps/human_profiling/`, flat, results in `results/261030/` |
| `legacy/` | deleted — bbmap notebook and Streamlit app |
| `tagmentation/` | deleted — an alias package, unused by any notebook |

Each app gained a README explaining how to run its pipeline. The three
forwarding modules at the repository root were repointed at `core/`, so the
~75 notebook import sites under `~/Projects/analyze_tagmentation` kept working
unchanged. All 209 tests pass with no skips in an environment that has
cutadapt, bwa, samtools and pysam.

## Left

Listed under "What has not been done" in [ARCHITECTURE.md](ARCHITECTURE.md):
the triplicated CBCL reader, `read_snapgene.py`'s home, ChIP-seq test
coverage, the 200→220 R2 cycle question, and the inactive LE still-donor check.
