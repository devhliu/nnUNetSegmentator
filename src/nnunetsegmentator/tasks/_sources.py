"""
Upstream weight sources and shared numeric constants.

Single place for the release URLs of every task's weights and for the
thresholds reused across task postprocessing, so task modules stay declarative.
"""

from __future__ import annotations

from ..utils.model_layout import NON_DOWNLOADABLE_URL

__all__ = [
    "NON_DOWNLOADABLE_URL",
    "TOTALSEG_RELEASE_BASE",
    "totalseg_url",
    "MOOSE_RELEASE_BASE",
    "moose_url",
    "BOA_RELEASE_BASE",
    "boa_url",
    "TS2D_ZENODO_RECORDS",
    "ts2d_url",
    "BODY_EXTREMITIES_MIN_SIZE_MM3",
    "BOA_SMALL_OBJECT_SIZE",
]

# --- TotalSegmentator -------------------------------------------------------

TOTALSEG_RELEASE_BASE = "https://github.com/wasserth/TotalSegmentator/releases/download"


def totalseg_url(foldername: str, version: str) -> str:
    """Weights URL of a TotalSegmentator release folder."""
    return f"{TOTALSEG_RELEASE_BASE}/{version}/{foldername}.zip"


# --- MOOSE ------------------------------------------------------------------

MOOSE_RELEASE_BASE = "https://github.com/ENHANCE-PET/MOOSE/releases/download"


def moose_url(version: str, filename: str) -> str:
    """Weights URL of a MOOSE release archive."""
    return f"{MOOSE_RELEASE_BASE}/{version}/{filename}.zip"


# --- BOA (Body and Organ Analysis) ------------------------------------------

BOA_RELEASE_BASE = (
    "https://github.com/UMEssen/Body-and-Organ-Analysis/releases/download/v1.0.0-weights"
)


def boa_url(foldername: str) -> str:
    """Weights URL of a BOA body-composition-analysis release folder."""
    return f"{BOA_RELEASE_BASE}/{foldername}.zip"


# --- TotalSegmentator 2D (Zenodo) -------------------------------------------

# Zenodo record ids per released model family; files are
# ``{key}_{group}.zip`` and the record host serves them via ``?download=1``.
TS2D_ZENODO_RECORDS = {
    "ts2d-v2-ep4000b2": "16985939",
    "ts2d-v1-ep4000b2": "16574232",
    "tsxr-v2-ep1000b2": "17052912",
}


def ts2d_url(key: str, group: str) -> str:
    """Weights URL of one ts2d/tsxr group model on Zenodo."""
    record = TS2D_ZENODO_RECORDS[key]
    return f"https://zenodo.org/records/{record}/files/{key}_{group}.zip?download=1"


# --- Shared thresholds ------------------------------------------------------

# TotalSegmentator body task: remove body_extremities blobs below 50,000 mm3.
BODY_EXTREMITIES_MIN_SIZE_MM3 = 50000

# BOA body-parts task: minimum size of kept connected components/holes.
BOA_SMALL_OBJECT_SIZE = 3000