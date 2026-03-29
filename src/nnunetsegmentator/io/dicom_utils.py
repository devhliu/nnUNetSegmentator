"""
DICOM utilities.

This module provides utilities for handling DICOM files, including:
- Conversion to NIfTI using dcm2niix (via dicom2nifti)
- Extraction of PET SUV parameters
- DICOM tag parsing

Ported from LION/lionz/image_conversion.py
"""

import os
import logging
import contextlib
import io
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
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
        if not tag: return None
        time_str = str(tag).split('.')[0]
        if len(time_str) < 6: return None
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


def convert_dicom_to_nifti(dicom_dir: Union[str, Path], output_dir: Union[str, Path]) -> Path:
    """
    Convert a directory of DICOM files to NIfTI.
    Returns path to the directory containing NIfTI files.
    """
    if not DICOM_AVAILABLE:
        raise ImportError("dicom2nifti is required")
        
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use dicom2nifti
    # Note: LION uses a custom dcm2niix wrapper, but dicom2nifti is a pure python alternative 
    # (or wrapper around dcm2niix if installed) that is easier to manage as a dependency.
    # If dcm2niix binary is available, dicom2nifti will use it? 
    # Actually dicom2nifti is Python based but can call dcm2niix.
    
    # Suppress output
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dicom2nifti.convert_directory(dicom_dir, output_dir, compression=True, reorient=True)
        
    return output_dir
