"""
DICOM utilities.

This module provides utilities for handling DICOM files, including:
- Conversion of a DICOM series to NIfTI using dicom2nifti (modality LUT applied)
- Primary-series selection when a directory bundles several acquisitions
- Extraction of PET SUV parameters
- DICOM tag parsing
"""

import logging
import contextlib
import io
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

# Try to import pydicom and dicom2nifti
try:
    import pydicom
    import dicom2nifti
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_dicom_file(filename: Union[str, Path]) -> bool:
    """
    Checks if a file is a DICOM file.
    """
    if not DICOM_AVAILABLE:
        return False
        
    try:
        pydicom.dcmread(str(filename), stop_before_pixels=True)
        return True
    except pydicom.errors.InvalidDicomError:
        logger.debug("File is not valid DICOM: %s", filename)
        return False
    except (OSError, AttributeError, ValueError) as exc:
        logger.warning("Failed to inspect DICOM file %s: %s", filename, exc)
        return False


def get_pet_suv_parameters(dicom_file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get DICOM parameters for SUV calculation from DICOM tags.
    """
    if not DICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM processing")

    ds = pydicom.dcmread(str(dicom_file_path), stop_before_pixels=True)
    
    def tag_to_float(tag):
        return float(tag) if tag is not None else None

    # Base parameters
    params = {
        'PatientWeight': tag_to_float(ds.get('PatientWeight')),
        'AcquisitionDate': ds.get('AcquisitionDate'),
        'AcquisitionTime': ds.get('AcquisitionTime'),
        'SeriesTime': ds.get('SeriesTime'),
        'DecayFactor': tag_to_float(ds.get('DecayFactor')),
        'DecayCorrection': ds.get('DecayCorrection'),
        'Units': ds.get('Units'),
        'RadionuclideTotalDose': None,
        'RadiopharmaceuticalStartTime': None,
        'RadionuclideHalfLife': None,
        'SUVScaleFactor': None
    }

    # Radiopharmaceutical Information
    if 'RadiopharmaceuticalInformationSequence' in ds:
        radio_info = ds.RadiopharmaceuticalInformationSequence[0]
        params['RadionuclideTotalDose'] = tag_to_float(radio_info.get('RadionuclideTotalDose'))
        params['RadiopharmaceuticalStartTime'] = radio_info.get('RadiopharmaceuticalStartTime')
        params['RadionuclideHalfLife'] = tag_to_float(radio_info.get('RadionuclideHalfLife'))
        
    # Philips private tag for SUV scale factor (0x7053, 0x1000)
    # This is often used for 'CNTS' units
    try:
        if (0x7053, 0x1000) in ds:
            params['SUVScaleFactor'] = float(ds[0x7053, 0x1000].value)
    except (TypeError, ValueError, KeyError) as exc:
        logger.debug("Failed to parse SUVScaleFactor private tag: %s", exc)

    return params


def compute_corrected_activity(params: Dict[str, Any]) -> Optional[float]:
    """
    Compute decay-corrected activity.
    """
    def tag_to_time_seconds(tag):
        if not tag:
            return None
        time_str = str(tag).split('.')[0]
        if len(time_str) < 6:
            return None
        h, m, s = int(time_str[0:2]), int(time_str[2:4]), int(time_str[4:6])
        return h * 3600 + m * 60 + s

    start_time = params.get('RadiopharmaceuticalStartTime')
    series_time = params.get('SeriesTime')
    total_dose = params.get('RadionuclideTotalDose')
    half_life = params.get('RadionuclideHalfLife')

    if not all([start_time, series_time, total_dose, half_life]):
        return total_dose

    t1 = tag_to_time_seconds(start_time)
    t2 = tag_to_time_seconds(series_time)
    
    if t1 is None or t2 is None:
        return total_dose

    time_diff = t2 - t1
    
    # Decay correction formula: A = A0 * 2^(-dt/T_half)
    corrected_activity = total_dose * pow(2.0, -(time_diff / half_life))
    
    return corrected_activity


def calculate_suv_conversion_factor(params: Dict[str, Any]) -> float:
    """
    Calculate factor to convert raw pixel values to SUV.
    Returns 1.0 if conversion not possible.
    """
    units = params.get('Units', '')
    
    if units == 'CNTS' and params.get('SUVScaleFactor'):
        return params['SUVScaleFactor']
        
    if units == 'BQML':
        weight_kg = params.get('PatientWeight')
        corrected_dose_bq = compute_corrected_activity(params)
        
        if weight_kg and corrected_dose_bq:
            # SUV = Activity Concentration / (Injected Dose / Body Weight)
            # Factor = 1 / (Injected Dose / Body Weight)
            # Dose in Bq, Weight in kg
            # Result usually required in g/ml or similar.
            # 1 SUV = 1 (Bq/ml) / (Bq/g) = 1 g/ml
            
            # If pixels are Bq/ml:
            # SUV = Pixels / (CorrectedDose / (Weight * 1000))  [Weight to g]
            # SUV = Pixels * (Weight * 1000 / CorrectedDose)
            
            # Wait, standard formula:
            # SUV = (Activity / Volume) / (Injected Dose / Body Weight)
            # If pixels are Bq/ml.
            # SUV = (Bq/ml) / (Bq/g) = g/ml
            
            # Conversion factor K:
            # SUV = K * Pixel
            # K = Body Weight (g) / Injected Dose (Bq)
            
            weight_g = weight_kg * 1000.0
            return weight_g / corrected_dose_bq
            
    return 1.0


def _series_uid(dataset: Any) -> str:
    """Return the SeriesInstanceUID as a string ('' when absent)."""
    return str(getattr(dataset, "SeriesInstanceUID", "") or "")


def _position_key(dataset: Any) -> Tuple[int, float]:
    """Sort key ordering slices along the acquisition (slice-normal) axis.

    The slice normal (row x col direction cosines) is used so the ordering is
    correct for any acquisition plane, including series whose column cosine is
    negated. Falls back to the z position, then the instance number.
    """
    ipp = getattr(dataset, "ImagePositionPatient", None)
    iop = getattr(dataset, "ImageOrientationPatient", None)
    if ipp is not None and iop is not None and len(ipp) >= 3 and len(iop) >= 6:
        row = np.asarray(iop[:3], dtype=float)
        col = np.asarray(iop[3:6], dtype=float)
        normal = np.cross(row, col)
        return (0, float(np.dot(np.asarray(ipp[:3], dtype=float), normal)))
    if ipp is not None and len(ipp) >= 3:
        return (1, float(ipp[2]))
    return (2, float(getattr(dataset, "InstanceNumber", 0) or 0))


def select_primary_series(
    dicom_files: Union[str, Path, List[Union[str, Path]]]
) -> Tuple[List[str], List[Any]]:
    """Resolve DICOM file(s)/directory to the dominant series.

    dicom2nifti does not group by SeriesInstanceUID, and a directory may bundle
    several series (e.g. a localizer plus the diagnostic scan). Files are
    grouped by SeriesInstanceUID and the series with the most slices is used;
    a warning is emitted when more than one series is present.

    Returns ``(file_paths, datasets)`` in slice order.
    """
    if not DICOM_AVAILABLE:
        raise ImportError("pydicom and dicom2nifti are required for DICOM conversion")

    if isinstance(dicom_files, (str, Path)):
        directory = Path(dicom_files)
        reader_files = sorted(str(p) for p in directory.glob("*.dcm"))
        if not reader_files:
            reader_files = sorted(
                str(p) for p in directory.iterdir()
                if p.is_file() and _has_dicom_magic(p)
            )
        dicom_files = reader_files

    pairs = [(str(f), pydicom.dcmread(str(f))) for f in dicom_files]
    if not pairs:
        raise ValueError("No DICOM files provided")

    groups: Dict[str, List[Tuple[str, Any]]] = {}
    for path, dataset in pairs:
        groups.setdefault(_series_uid(dataset), []).append((path, dataset))

    if len(groups) > 1:
        counts = {uid: len(members) for uid, members in groups.items()}
        primary_uid = max(counts, key=lambda uid: counts[uid])
        logger.warning(
            "Input contains %d DICOM series; using the largest (%d slices, "
            "SeriesInstanceUID=%s). Slice counts: %s",
            len(groups), counts[primary_uid], primary_uid, counts,
        )
    else:
        primary_uid = next(iter(groups))

    ordered = sorted(groups[primary_uid], key=lambda pair: _position_key(pair[1]))
    return [path for path, _ in ordered], [dataset for _, dataset in ordered]


def _has_dicom_magic(path: Path) -> bool:
    """Detect part-10 DICOM files without a .dcm extension via the DICM magic."""
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def convert_series_to_nifti(
    dicom_files: Union[str, Path, List[Union[str, Path]]],
    output_file: Union[str, Path],
    reorient_nifti: bool = True,
) -> Tuple[Any, List[str]]:
    """Convert one DICOM series to NIfTI using dicom2nifti.

    Unlike a hand-rolled reader, dicom2nifti applies the modality LUT
    (RescaleSlope/RescaleIntercept), so CT volumes carry correct HU values, and
    with ``reorient_nifti=True`` the data/affine are stored in the canonical LAS
    space used by TotalSegmentator/MOOSE.

    Returns ``(nibabel_image, primary_file_paths)``.
    """
    file_paths, datasets = select_primary_series(dicom_files)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = dicom2nifti.convert_dicom.dicom_array_to_nifti(
            datasets, str(output_file), reorient_nifti=reorient_nifti
        )

    return results["NII"], file_paths
