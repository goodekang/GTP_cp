from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


def _gdc_cases_query_brca(size: int) -> dict:
    filters = {
        "op": "and",
        "content": [{"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-BRCA"]}}],
    }
    fields = [
        "submitter_id",
        "diagnoses.days_to_death",
        "diagnoses.vital_status",
        "diagnoses.days_to_last_follow_up",
    ]
    payload = {"filters": filters, "format": "JSON", "fields": ",".join(fields), "size": size}
    r = requests.post(GDC_CASES_ENDPOINT, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _extract_os(hit: dict) -> tuple[float | None, int | None]:
    diags = hit.get("diagnoses") or []
    if not diags:
        return None, None
    d0 = diags[0]
    vital = str(d0.get("vital_status") or "").lower()
    days_to_death = d0.get("days_to_death")
    days_to_last = d0.get("days_to_last_follow_up")
    if vital == "dead" and days_to_death is not None:
        return float(days_to_death), 1
    if days_to_last is not None:
        return float(days_to_last), 0
    return None, None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fetch TCGA-BRCA clinical OS from GDC cases endpoint.")
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument("--size", type=int, default=20000, help="Max number of case records to fetch from GDC.")
    args = p.parse_args(argv)

    paths = BRCAPaths(Path(args.root))
    ensure_dir(paths.clinical_dir)

    data = _gdc_cases_query_brca(size=int(args.size))
    hits = data.get("data", {}).get("hits", [])

    rows = []
    for h in hits:
        submitter_id = h.get("submitter_id")
        patient_id = str(submitter_id or "")[:12]
        os_days, os_event = _extract_os(h)
        rows.append(
            {
                "patient_id": patient_id,
                "case_submitter_id": submitter_id,
                "os_days": os_days,
                "os_event": os_event,
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["patient_id"]).sort_values("patient_id")
    df.to_csv(paths.clinical_csv, index=False)
    print(f"[ok] wrote clinical: {paths.clinical_csv} (patients={len(df)})")


if __name__ == "__main__":
    main()






