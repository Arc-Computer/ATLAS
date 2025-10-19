"""Build a lightweight SQLite database from SecRL incident reports.

This utility downloads the public incident reports published alongside the
SecRL benchmark and extracts line-level evidence entries. The resulting
database acts as the read-only knowledge base the Student can query during
runtime, providing a reproducible alternative to the original MySQL logs.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator
from urllib.request import urlopen

INCIDENT_IDS = ["5", "34", "38", "39", "55", "134", "166", "322"]
BASE_URL = (
    "https://raw.githubusercontent.com/microsoft/SecRL/main/"
    "incident_reports/incident_{incident}_incident_report.txt"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "paper_assets" / "arcops_cyber"
REPORT_DIR = DATA_ROOT / "incident_reports"
DB_PATH = DATA_ROOT / "sqlite" / "secrl_incidents.db"


def download_report(incident: str, force: bool = False) -> Path:
    """Download a single incident report if not cached locally."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = REPORT_DIR / f"incident_{incident}.txt"
    if local_path.exists() and not force:
        return local_path

    url = BASE_URL.format(incident=incident)
    with urlopen(url) as response:  # noqa: S310 - trusted GitHub host
        if response.status != 200:
            raise RuntimeError(f"Unable to fetch {url} (status {response.status})")
        local_path.write_bytes(response.read())
    return local_path


def iter_evidence_lines(text: str) -> Iterator[str]:
    """Extract evidence bullets from an incident report."""

    bullet_pattern = re.compile(r"^(?:[\-\u2022]|•)\s*(.+)$")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = bullet_pattern.match(line)
        if match:
            yield match.group(1).strip()


def build_database(reports: Iterable[Path]) -> None:
    """Create the SQLite database from the downloaded reports."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_events (
                incident_id TEXT NOT NULL,
                line_index INTEGER NOT NULL,
                evidence TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX idx_incident ON incident_events(incident_id)")

        for report_path in reports:
            incident_id = report_path.stem.split("_")[1]
            with report_path.open("r", encoding="utf-8") as handle:
                content = handle.read()
            for index, evidence in enumerate(iter_evidence_lines(content), start=1):
                conn.execute(
                    "INSERT INTO incident_events (incident_id, line_index, evidence) VALUES (?, ?, ?)",
                    (incident_id, index, evidence),
                )
        conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SQLite evidence DB from SecRL reports.")
    parser.add_argument("--force-download", action="store_true", help="Redownload reports even if cached.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [download_report(incident, force=args.force_download) for incident in INCIDENT_IDS]
    build_database(reports)
    print(f"SQLite evidence database written to {DB_PATH}")


if __name__ == "__main__":
    main()
