from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


@dataclass(frozen=True)
class GDCFileRecord:
    file_id: str
    file_name: str
    md5sum: str | None
    file_size: int | None
    data_format: str | None
    data_type: str | None
    experimental_strategy: str | None
    project_id: str
    case_id: str | None
    submitter_id: str | None  # TCGA barcode at case level


def _post_json(url: str, payload: dict[str, Any], timeout_s: int = 120) -> dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def query_tcga_brca_diagnostic_slides(*, size: int = 20000) -> list[GDCFileRecord]:
    """
    Returns a list of WSI-like pathology slide files for TCGA-BRCA from GDC.
    We keep the query broad and allow downstream filtering in CSV.
    """
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Biospecimen"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        ],
    }

    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "data_format",
        "data_type",
        "experimental_strategy",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
    ]

    payload = {
        "filters": filters,
        "format": "JSON",
        "fields": ",".join(fields),
        "size": size,
    }

    data = _post_json(GDC_FILES_ENDPOINT, payload)
    hits = data.get("data", {}).get("hits", [])

    out: list[GDCFileRecord] = []
    for h in hits:
        cases = (h.get("cases") or [])
        case0 = cases[0] if cases else {}
        out.append(
            GDCFileRecord(
                file_id=h.get("file_id"),
                file_name=h.get("file_name"),
                md5sum=h.get("md5sum"),
                file_size=h.get("file_size"),
                data_format=h.get("data_format"),
                data_type=h.get("data_type"),
                experimental_strategy=h.get("experimental_strategy"),
                project_id=(case0.get("project") or {}).get("project_id", "TCGA-BRCA"),
                case_id=case0.get("case_id"),
                submitter_id=case0.get("submitter_id"),
            )
        )
    return out


def download_manifest_json(records: list[GDCFileRecord]) -> str:
    """
    A minimal gdc-client manifest (TSV) can be generated from file_id list.
    This helper returns JSON lines for reproducible storage; TSV export is done in pipeline.
    """
    return "\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in records) + "\n"






