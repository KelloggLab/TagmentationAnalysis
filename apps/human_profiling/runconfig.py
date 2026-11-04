"""Load the run layout, sample sheet and assay arms from one config file.

Everything that used to be a module-level constant in verify_structure.py,
recheck_inserts.py and extract_salvage.py lives here instead, so the three
scripts cannot drift apart and so a different run needs no code edit.

    cfg = load()                      # apps/human_profiling/run.yaml
    cfg = load("other_run.yaml")      # or JSON, picked by extension

YAML needs PyYAML; a .json config loads on the standard library alone, which
keeps these scripts runnable on a bare Python where only numpy is installed.

The loader validates the config rather than trusting it. Most checks are
arithmetic that used to be assumed -- that the UMI and the i5 barcode together
fill the I2 read, that the i7 barcodes are as long as I1, that no two samples
share an index pair. One is a real trap: a still-donor check needs a stretch of
donor past the transposon terminus to compare a read against, and if the arm
does not supply enough the check silently passes everything. See
`Arm.still_donor_active`.
"""
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "run.yaml"

DNA = set("ACGT")


class ConfigError(Exception):
    """The config is unusable: wrong shape, or internally inconsistent."""


@dataclass(frozen=True)
class Cycles:
    """1-based first cycle and length of each read, as the run reports them."""
    i1_start: int
    i1_len: int
    i2_start: int
    i2_len: int
    r2_start: int
    r2_len: int
    umi_len: int


@dataclass(frozen=True)
class Arm:
    """One primer arm (LE or RE) of the transposon.

    end_len is how many bases of transposon end sit between the primer's 3'
    end and the terminus, measured as the point where per-position agreement
    falls away. donor is the sequence the read runs through, starting at the
    primer's 3' end, so donor[end_len:] is what lies past the terminus.
    """
    name: str
    primer: str
    donor: str
    end_len: int
    still_donor_min_tail: int

    @property
    def donor_past_terminus(self) -> str:
        return self.donor[self.end_len:]

    @property
    def still_donor_active(self) -> bool:
        """Whether this arm can detect a read that is still donor plasmid.

        The check compares the read's flank against the donor past the
        terminus. Too short a tail and there is nothing to compare, so the
        check passes every read -- including the residual donor it exists to
        catch. False here means that filter is off for this arm.
        """
        return len(self.donor_past_terminus) >= self.still_donor_min_tail


@dataclass(frozen=True)
class Sample:
    """One index pair. Several may share a (sample, arm) group and pool."""
    library: str
    sample: str
    arm: str
    i5: str
    i7: str


@dataclass
class RunConfig:
    run_id: str
    platform: str
    cycles: Cycles
    arms: dict
    samples: list
    mosaic_end: str
    min_flank: int
    site_key: int
    source: Path = field(default=None)

    @property
    def groups(self):
        """(sample, arm) pairs in sheet order, deduplicated."""
        seen = []
        for s in self.samples:
            if (s.sample, s.arm) not in seen:
                seen.append((s.sample, s.arm))
        return seen

    def arm(self, name) -> Arm:
        try:
            return self.arms[name]
        except KeyError:
            raise ConfigError(f"no arm {name!r}; have {sorted(self.arms)}")

    def inactive_still_donor_arms(self):
        """Arms whose still-donor check cannot fire. Empty is the healthy case."""
        return [a.name for a in self.arms.values() if not a.still_donor_active]


def _read_mapping(path: Path):
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                f"{path} is YAML but PyYAML is not installed; either "
                f"'conda install pyyaml' or write the config as .json"
            ) from exc
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(f"{path}: expected .yaml, .yml or .json")
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _require(data, key, where):
    if key not in data:
        raise ConfigError(f"{where}: missing {key!r}")
    return data[key]


def _dna(seq, where):
    seq = str(seq).strip().upper()
    if not seq:
        raise ConfigError(f"{where}: empty sequence")
    bad = sorted(set(seq) - DNA)
    if bad:
        raise ConfigError(f"{where}: not DNA, found {''.join(bad)!r}")
    return seq


def read_sample_sheet(path: Path):
    """One Sample per row. Columns: library, sample, arm, i5_index, i7_index."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ConfigError(f"{path}: no sample rows")

    needed = {"library", "sample", "arm", "i5_index", "i7_index"}
    missing = needed - set(rows[0])
    if missing:
        raise ConfigError(f"{path}: missing columns {sorted(missing)}")

    samples = []
    for n, row in enumerate(rows, start=2):
        where = f"{path}:{n}"
        samples.append(Sample(
            library=row["library"].strip(),
            sample=row["sample"].strip(),
            arm=row["arm"].strip(),
            i5=_dna(row["i5_index"], f"{where} i5_index"),
            i7=_dna(row["i7_index"], f"{where} i7_index"),
        ))
    return samples


def load(path=None) -> RunConfig:
    """Read a config, its sample sheet, and check the three agree.

    path defaults to $UDITAS_RUN_CONFIG, then apps/human_profiling/run.yaml.
    """
    if path is None:
        path = os.environ.get("UDITAS_RUN_CONFIG") or DEFAULT_CONFIG
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigError(f"no such config: {path}")

    data = _read_mapping(path)

    c = _require(data, "cycles", str(path))
    try:
        cycles = Cycles(
            i1_start=int(_require(c, "i1_start", "cycles")),
            i1_len=int(_require(c, "i1_len", "cycles")),
            i2_start=int(_require(c, "i2_start", "cycles")),
            i2_len=int(_require(c, "i2_len", "cycles")),
            r2_start=int(_require(c, "r2_start", "cycles")),
            r2_len=int(_require(c, "r2_len", "cycles")),
            umi_len=int(_require(c, "umi_len", "cycles")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"cycles: {exc}") from exc

    assay = _require(data, "assay", str(path))
    min_tail = int(assay.get("still_donor_min_tail", 12))

    arms = {}
    for name, spec in _require(assay, "arms", "assay").items():
        where = f"assay.arms.{name}"
        end_len = int(_require(spec, "end_len", where))
        donor = _dna(_require(spec, "donor", where), f"{where}.donor")
        if end_len < 0 or end_len > len(donor):
            raise ConfigError(
                f"{where}: end_len {end_len} outside the {len(donor)} bp donor")
        arms[name] = Arm(
            name=name,
            primer=_dna(_require(spec, "primer", where), f"{where}.primer"),
            donor=donor,
            end_len=end_len,
            still_donor_min_tail=min_tail,
        )
    if not arms:
        raise ConfigError("assay.arms: no arms defined")

    sheet = Path(_require(data, "sample_sheet", str(path)))
    if not sheet.is_absolute():
        sheet = path.parent / sheet
    samples = read_sample_sheet(sheet)

    cfg = RunConfig(
        run_id=str(data.get("run_id", "")),
        platform=str(data.get("platform", "")),
        cycles=cycles,
        arms=arms,
        samples=samples,
        mosaic_end=_dna(_require(assay, "mosaic_end", "assay"), "assay.mosaic_end"),
        min_flank=int(assay.get("min_flank", 40)),
        site_key=int(assay.get("site_key", 40)),
        source=path,
    )
    _validate(cfg, sheet)
    return cfg


def _validate(cfg: RunConfig, sheet: Path):
    """Checks the three scripts used to assume. Raises on anything fatal."""
    i5_lens = {len(s.i5) for s in cfg.samples}
    i7_lens = {len(s.i7) for s in cfg.samples}
    if len(i5_lens) != 1:
        raise ConfigError(f"{sheet}: i5 barcodes have mixed lengths {sorted(i5_lens)}")
    if len(i7_lens) != 1:
        raise ConfigError(f"{sheet}: i7 barcodes have mixed lengths {sorted(i7_lens)}")

    i5_len, i7_len = i5_lens.pop(), i7_lens.pop()
    if cfg.cycles.umi_len + i5_len != cfg.cycles.i2_len:
        raise ConfigError(
            f"I2 is {cfg.cycles.i2_len} cycles but the UMI ({cfg.cycles.umi_len}) "
            f"plus an i5 barcode ({i5_len}) is {cfg.cycles.umi_len + i5_len}")
    if i7_len != cfg.cycles.i1_len:
        raise ConfigError(
            f"I1 is {cfg.cycles.i1_len} cycles but the i7 barcodes are {i7_len} bp")

    pairs = [(s.i5, s.i7) for s in cfg.samples]
    dupes = {p for p in pairs if pairs.count(p) > 1}
    if dupes:
        raise ConfigError(f"{sheet}: index pairs used twice: {sorted(dupes)}")

    libs = [s.library for s in cfg.samples]
    if len(set(libs)) != len(libs):
        raise ConfigError(f"{sheet}: duplicate library names")

    unknown = sorted({s.arm for s in cfg.samples} - set(cfg.arms))
    if unknown:
        raise ConfigError(
            f"{sheet}: arm(s) {unknown} not in assay.arms ({sorted(cfg.arms)})")
