import json
import re
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic_patients"


def _visit_number(path: Path) -> int:
    match = re.search(r"visit(\d+)\.txt$", path.name)
    if not match:
        return 0
    return int(match.group(1))


def list_patients(data_root: Path = DATA_ROOT) -> list[str]:
    """Return available patient case directory names."""
    if not data_root.exists():
        return []

    return sorted(path.name for path in data_root.iterdir() if path.is_dir())


def load_visits(patient_id: str, data_root: Path = DATA_ROOT) -> list[Path]:
    """Return visit note paths for a patient, sorted by visit number."""
    patient_dir = data_root / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient case not found: {patient_id}")

    visits = patient_dir.glob("visit*.txt")
    return sorted(visits, key=_visit_number)


def read_visit(visit_path: str | Path) -> str:
    """Read a single visit note."""
    return Path(visit_path).read_text(encoding="utf-8")


def load_patient(patient_id: str, data_root: Path = DATA_ROOT) -> dict:
    """Load metadata and visit note contents for a patient case."""
    patient_dir = data_root / patient_id
    metadata_path = patient_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for patient case: {patient_id}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    visits = [
        {
            "visit_number": _visit_number(visit_path),
            "filename": visit_path.name,
            "content": read_visit(visit_path),
        }
        for visit_path in load_visits(patient_id, data_root)
    ]

    return {
        "patient_id": patient_id,
        "metadata": metadata,
        "visits": visits,
    }
