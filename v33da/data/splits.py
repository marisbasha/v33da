"""Reusable dataset split helpers for V33DL experiments."""

from __future__ import annotations

from collections import Counter
import hashlib

from v33da.data.dataset import V33DLDataset


ALL_EXPERIMENTS = ("juvExpBP01", "juvExpBP02", "juvExpBP05")
_EXPERIMENT_ALIASES = {
    "bp01": "juvExpBP01",
    "juvexpbp01": "juvExpBP01",
    "bp02": "juvExpBP02",
    "juvexpbp02": "juvExpBP02",
    "bp05": "juvExpBP05",
    "juvexpbp05": "juvExpBP05",
}


class GeneralizationDataset:
    """Legacy unseen-bird split driven by the stored split_generalization column."""

    def __init__(self, gen_split: str):
        self.indices = []
        self.bases = {}
        for split in ("train", "val", "test"):
            ds = V33DLDataset(split)
            gen_col = [r.as_py() for r in ds.table.column("split_generalization")]
            for i, gs in enumerate(gen_col):
                if gs == gen_split:
                    self.indices.append((split, i))
            self.bases[split] = ds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        split, i = self.indices[idx]
        return self.bases[split][i]


class ExperimentFilteredDataset:
    """Filter samples by experiment across one or more base splits."""

    def __init__(self, base_splits: tuple[str, ...], experiments: tuple[str, ...], include: bool):
        self.indices = []
        self.bases = {}
        selected = set(experiments)
        for split in base_splits:
            ds = V33DLDataset(split)
            self.bases[split] = ds
            for i, exp in enumerate(ds._exps):
                keep = exp in selected
                if keep == include:
                    self.indices.append((split, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        split, i = self.indices[idx]
        return self.bases[split][i]


def _stable_bucket(sample_id: str, buckets: int = 10) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


class LeaveOneBirdDataset:
    """Within-experiment leave-one-bird-out split with deterministic train/val partition."""

    def __init__(self, experiment: str, heldout_bird: int, role: str):
        if role not in {"train", "val", "test"}:
            raise ValueError(f"Unknown role '{role}'.")
        self.indices = []
        self.bases = {}
        for split in ("train", "val", "test"):
            ds = V33DLDataset(split)
            self.bases[split] = ds
            for i, (exp, voc, sample_id) in enumerate(zip(ds._exps, ds._voc_idx, ds._ids)):
                if exp != experiment:
                    continue
                if voc == heldout_bird:
                    if role == "test":
                        self.indices.append((split, i))
                    continue
                bucket = _stable_bucket(str(sample_id))
                if role == "train" and bucket != 0:
                    self.indices.append((split, i))
                elif role == "val" and bucket == 0:
                    self.indices.append((split, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        split, i = self.indices[idx]
        return self.bases[split][i]


def resolve_experiment_name(name: str | None) -> str:
    if name is None:
        return "juvExpBP05"
    key = name.strip()
    if key in ALL_EXPERIMENTS:
        return key
    alias = _EXPERIMENT_ALIASES.get(key.lower())
    if alias is not None:
        return alias
    raise ValueError(
        f"Unknown experiment '{name}'. Expected one of {', '.join(ALL_EXPERIMENTS)} "
        "or aliases BP01/BP02/BP05."
    )


def iter_requested_experiments(spec: str | None) -> list[str]:
    if spec is None or spec.strip().lower() == "all":
        return list(ALL_EXPERIMENTS)
    return [resolve_experiment_name(spec)]


def short_experiment_name(experiment: str) -> str:
    return resolve_experiment_name(experiment).replace("juvExp", "")


def dataset_experiment_counts(ds) -> dict[str, int]:
    counts = Counter(ds[i]["experiment"] for i in range(len(ds)))
    return dict(sorted(counts.items()))


def experiment_vocalizer_ids(experiment: str) -> list[int]:
    experiment = resolve_experiment_name(experiment)
    birds = set()
    for split in ("train", "val", "test"):
        ds = V33DLDataset(split)
        birds.update(voc for exp, voc in zip(ds._exps, ds._voc_idx) if exp == experiment)
    return sorted(birds)


def iter_requested_birds(experiment: str, spec: str | int | None) -> list[int]:
    birds = experiment_vocalizer_ids(experiment)
    if spec is None or str(spec).strip().lower() == "all":
        return birds
    bird = int(spec)
    if bird not in birds:
        raise ValueError(f"Bird {bird} not found in {experiment}; available birds: {birds}")
    return [bird]


def load_split_triplet(
    split_name: str,
    heldout_experiment: str | None = None,
    heldout_bird: int | None = None,
):
    """Return train/val/test datasets for a named evaluation regime."""
    if split_name == "session":
        return (
            V33DLDataset("train"),
            V33DLDataset("val"),
            V33DLDataset("test"),
            {"key": "session", "label": "session"},
        )

    if split_name == "unseen_bird":
        return (
            GeneralizationDataset("train"),
            GeneralizationDataset("val"),
            GeneralizationDataset("test"),
            {"key": "unseen_bird", "label": "unseen_bird"},
        )

    if split_name == "heldout_experiment":
        heldout = resolve_experiment_name(heldout_experiment)
        tag = short_experiment_name(heldout).lower()
        return (
            ExperimentFilteredDataset(("train",), (heldout,), include=False),
            ExperimentFilteredDataset(("val",), (heldout,), include=False),
            ExperimentFilteredDataset(("train", "val", "test"), (heldout,), include=True),
            {"key": f"heldout_{tag}", "label": heldout, "heldout_experiment": heldout},
        )

    if split_name == "leave_one_bird_out":
        experiment = resolve_experiment_name(heldout_experiment)
        if heldout_bird is None:
            raise ValueError("leave_one_bird_out requires heldout_bird.")
        tag = short_experiment_name(experiment).lower()
        return (
            LeaveOneBirdDataset(experiment, int(heldout_bird), "train"),
            LeaveOneBirdDataset(experiment, int(heldout_bird), "val"),
            LeaveOneBirdDataset(experiment, int(heldout_bird), "test"),
            {
                "key": f"loo_{tag}_bird{int(heldout_bird)}",
                "label": experiment,
                "heldout_experiment": experiment,
                "heldout_bird": int(heldout_bird),
            },
        )

    raise ValueError(f"Unknown split '{split_name}'.")
