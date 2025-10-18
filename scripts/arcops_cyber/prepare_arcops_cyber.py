"""
Utility script for Day 0 of the ATLAS Core paper effort.

Tasks:
1. Download ExCyTIn-Bench question/answer files from Hugging Face.
2. Materialise the ArcOps-Cyber dataset layout used throughout the paper.
3. Generate per-question JSON task specs that the runtime harness can ingest.

Usage:
    python scripts/arcops_cyber/prepare_arcops_cyber.py \
        --output-root paper_assets/arcops_cyber

The script is intentionally idempotent; reruns will skip downloads that already
exist locally unless --force is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

from urllib.request import urlopen
from urllib.error import HTTPError, URLError


INCIDENT_IDS: List[str] = [
    "incident_5",
    "incident_34",
    "incident_38",
    "incident_39",
    "incident_55",
    "incident_134",
    "incident_166",
    "incident_322",
]

FILENAME_SUFFIX = "qa_incident_o1-ga_c42.json"
DATASET_BASE_URL = (
    "https://huggingface.co/datasets/anandmudgerikar/excytin-bench/resolve/main/questions"
)


def _download(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return
    try:
        with urlopen(url) as response:  # noqa: S310 - trusted Hugging Face host
            if response.status != 200:
                raise RuntimeError(f"Failed to download {url} (status {response.status})")
            destination.write_bytes(response.read())
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download {url} ({exc})") from exc


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    slug = slug.strip("-")
    return slug or "answer"


def _write_tasks(
    incident: str, split: str, questions: Iterable[dict], destination: Path
) -> None:
    incident_dir = destination / split / incident
    incident_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = incident_dir / f"{incident}_{split}.json"

    aggregate_payload = []
    for index, payload in enumerate(questions, start=1):
        task_id = f"{incident}_{split}_q{index:03d}"
        task = {
            "task_id": task_id,
            "incident": incident,
            "split": split,
            "context": payload.get("context", "").strip(),
            "question": payload.get("question", "").strip(),
            "gold_answer": payload.get("answer", "").strip(),
            "solution_steps": payload.get("solution", []),
            "metadata": {
                "start_alert": payload.get("start_alert"),
                "end_alert": payload.get("end_alert"),
                "start_entities": payload.get("start_entities"),
                "end_entities": payload.get("end_entities"),
                "shortest_alert_path": payload.get("shortest_alert_path"),
            },
        }
        aggregate_payload.append(task)

        filename = f"{task_id}_{_slugify(task['question'][:32])}.json"
        task_path = incident_dir / filename
        task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")

    aggregate_path.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")


def prepare_arcops_cyber(output_root: Path, force_download: bool = False) -> None:
    raw_dir = output_root / "raw"
    tasks_dir = output_root / "tasks"
    index_records = []

    for split in ("train", "test"):
        for incident in INCIDENT_IDS:
            filename = f"{incident}_{FILENAME_SUFFIX}"
            source_url = f"{DATASET_BASE_URL}/{split}/{filename}"
            destination = raw_dir / split / filename
            _download(source_url, destination, force=force_download)
            data = json.loads(destination.read_text(encoding="utf-8"))
            _write_tasks(incident, split, data, tasks_dir)
            index_records.append(
                {
                    "incident": incident,
                    "split": split,
                    "num_questions": len(data),
                    "source_url": source_url,
                    "local_path": str(destination.relative_to(output_root)),
                }
            )

    index_path = output_root / "dataset_index.json"
    index_path.write_text(json.dumps(index_records, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ArcOps-Cyber dataset assets.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper_assets/arcops_cyber"),
        help="Directory where the dataset artifacts should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of source files even if they exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_arcops_cyber(args.output_root, force_download=args.force)
    print(f"ArcOps-Cyber assets written to {args.output_root}")


if __name__ == "__main__":
    main()
