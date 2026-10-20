from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def append_experiment_record(
    path,
    record: Mapping,
) -> None:
    """Append one C2 experiment record as a JSONL line."""

    if not isinstance(record, Mapping):
        raise TypeError(
            "record must be a mapping"
        )

    log_path = Path(path)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        json.dump(
            dict(record),
            file,
            ensure_ascii=False,
            sort_keys=True,
        )

        file.write("\n")