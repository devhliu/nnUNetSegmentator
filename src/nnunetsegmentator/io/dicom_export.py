"""
DICOM Export Utilities

This module exports segmentations as DICOM-SEG (DICOM Segmentation) objects.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
from .. import image as sitk

logger = logging.getLogger(__name__)


# Deterministic fallback palette (RGB 0-255) for segments whose label has no
# colour in the central SNOMED mapping.
_DEFAULT_DISPLAY_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]

# NIfTI/framework (i, j, k) shape expected for each acquisition plane. The
# dicom2nifti output stores the DICOM column axis first (i), then rows (j),
# then slices (k).
_EXPECTED_INPUT_SHAPE = {
    "axial": lambda rows, cols, slices: (cols, rows, slices),
    "coronal": lambda rows, cols, slices: (cols, slices, rows),
    "sagittal": lambda rows, cols, slices: (slices, rows, cols),
}


def rgb_to_cielab_dicom(rgb) -> tuple:
    """Convert an 8-bit RGB triple to DICOM's 16-bit CIELab encoding.

    DICOM stores RecommendedDisplayCIELabValue as three unsigned 16-bit
    integers (L* scaled from 0-100, a*/b* from roughly -128..127).
    """
    r, g, b = (float(v) / 255.0 for v in rgb)

    def gamma_correct(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = gamma_correct(r), gamma_correct(g), gamma_correct(b)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    z /= 1.08883

    def f(t: float) -> float:
        delta = 6.0 / 29.0
        return t ** (1.0 / 3.0) if t > delta ** 3 else t / (3 * delta ** 2) + 4.0 / 29.0

    l_star = 116.0 * f(y) - 16.0
    a_star = 500.0 * (f(x) - f(y))
    b_star = 200.0 * (f(y) - f(z))

    l_dicom = int(round(l_star * 65535 / 100))
    a_dicom = int(round((a_star + 128) * 65535 / 255))
    b_dicom = int(round((b_star + 128) * 65535 / 255))
    return (
        max(0, min(65535, l_dicom)),
        max(0, min(65535, a_dicom)),
        max(0, min(65535, b_dicom)),
    )


def _infer_plane_from_iop(iop) -> Optional[str]:
    """Infer 'axial'/'coronal'/'sagittal'/'oblique' from direction cosines."""
    if iop is None or len(iop) < 6:
        return None
    try:
        row = np.asarray([float(v) for v in iop[:3]])
        col = np.asarray([float(v) for v in iop[3:6]])
    except (TypeError, ValueError):
        return None
    normal = np.abs(np.cross(row, col))
    if float(normal.max()) < 0.9:
        return "oblique"
    return ("sagittal", "coronal", "axial")[int(np.argmax(normal))]


def _needs_axial_row_flip(iop) -> bool:
    """Whether an axial mask's row axis must be reversed for DICOM output.

    Standard axial series use a column cosine of [0, 1, 0] (top of image =
    anterior); series with a negated column cosine must not be flipped.
    """
    if iop is None or len(iop) < 6:
        return True
    return float(iop[4]) >= 0


def _slice_spacing(datasets: List[Any]) -> Optional[float]:
    """Slice spacing (mm) from the first two images along the slice normal."""
    if len(datasets) < 2:
        return None
    iop = getattr(datasets[0], "ImageOrientationPatient", None)
    ipp0 = getattr(datasets[0], "ImagePositionPatient", None)
    ipp1 = getattr(datasets[1], "ImagePositionPatient", None)
    if iop is not None and len(iop) >= 6 and ipp0 is not None and ipp1 is not None:
        normal = np.cross(np.asarray(iop[:3], dtype=float), np.asarray(iop[3:6], dtype=float))
        delta = np.asarray(ipp1[:3], dtype=float) - np.asarray(ipp0[:3], dtype=float)
        return float(abs(np.dot(delta, normal)))
    thickness = float(getattr(datasets[0], "SliceThickness", 0) or 0)
    return thickness or None


def _reorient_to_dicom_frames(seg_array, plane, axial_row_flip):
    """Map a NIfTI-ordered (i, j, k) mask to highdicom's (frames, rows, cols).

    Transposes and flips the axes so the exported frames reproduce the
    original DICOM pixel grid.
    """
    if plane == "axial":
        frames = seg_array.transpose(1, 0, 2).transpose(2, 0, 1)
        if axial_row_flip:
            frames = frames[:, ::-1, :]
    elif plane == "coronal":
        frames = seg_array.transpose(0, 2, 1).transpose(2, 1, 0)[::-1, ::-1, :]
    elif plane == "sagittal":
        frames = seg_array.transpose(1, 2, 0).transpose(2, 1, 0)[:, ::-1, ::-1]
    else:
        raise ValueError(f"Unsupported plane for DICOM-SEG export: {plane!r}")
    return frames


class DICOMSEGExporter:
    """
    Export segmentations to DICOM-SEG format.
    
    DICOM-SEG is the standard DICOM object for storing segmentation results.
    """
    
    def __init__(self):
        """Initialize DICOM-SEG exporter."""
        try:
            import highdicom
            self.highdicom = highdicom
        except ImportError:
            raise ImportError(
                "highdicom is required for DICOM-SEG export. "
                "Install with: pip install highdicom"
            )
    
    def export(self,
               segmentation: Union[sitk.Image, np.ndarray],
               reference_dicom_dir: Path,
               output_path: Path,
               labels: Dict[int, str],
               series_description: str = "Segmentation",
               manufacturer: str = "nnunetsegmentator") -> Path:
        """
        Export segmentation to DICOM-SEG.

        Args:
            segmentation: Label map sharing the source DICOM geometry. A
                framework Image is used as-is; numpy arrays are expected in
                NIfTI (i, j, k) order, matching the volume the pipeline read.
            reference_dicom_dir: Directory (or explicit file list) of reference
                DICOM series
            output_path: Output DICOM-SEG file path
            labels: Dictionary mapping label values to label names
            series_description: Series description
            manufacturer: Manufacturer name
        
        Returns:
            Path to created DICOM-SEG file
        """
        from pydicom import dcmread
        from pydicom.uid import generate_uid
        import highdicom.seg as seg
        from highdicom.seg.enum import SegmentationTypeValues

        # Collect reference DICOM files (directory or explicit file list)
        dicom_files = self._collect_reference_files(reference_dicom_dir)
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {reference_dicom_dir}")

        # Order source images along the slice axis so frames line up with the
        # reoriented mask.
        ref_datasets = self._sort_source_images([dcmread(str(f)) for f in dicom_files])

        iop = getattr(ref_datasets[0], "ImageOrientationPatient", None)
        plane = _infer_plane_from_iop(iop)
        if plane in (None, "oblique"):
            raise ValueError(
                "DICOM-SEG export requires an orthogonal acquisition; got "
                f"plane={plane!r} (ImageOrientationPatient={iop})")

        rows = int(ref_datasets[0].Rows)
        cols = int(ref_datasets[0].Columns)
        slices = len(ref_datasets)

        if isinstance(segmentation, sitk.Image):
            seg_array = np.asarray(sitk.GetArrayFromImage(segmentation))
            seg_spacing = tuple(float(v) for v in segmentation.GetSpacing())
        else:
            seg_array = np.asarray(segmentation)
            seg_spacing = None
        if seg_array.ndim != 3:
            raise ValueError(
                f"Segmentation must be a 3D label map, got shape {seg_array.shape}")
        if not np.any(seg_array > 0):
            raise ValueError("Segmentation label map is empty (no foreground)")

        self._validate_geometry(
            seg_array.shape, seg_spacing, plane, rows, cols, slices, ref_datasets)

        frames = _reorient_to_dicom_frames(
            seg_array, plane, _needs_axial_row_flip(iop))

        # Only keep non-empty segments, renumbering them contiguously from 1
        # (highdicom BINARY segment numbers must be consecutive).
        present_labels = sorted(int(v) for v in np.unique(frames) if int(v) != 0)
        if not present_labels:
            raise ValueError("Segmentation label map is empty (no foreground)")

        dtype = np.uint8 if len(present_labels) <= np.iinfo(np.uint8).max else np.uint16
        pixel_array = np.zeros(frames.shape, dtype=dtype)

        algorithm_identification = self._get_algorithm_identification()
        segment_descriptions = []
        for number, label_value in enumerate(present_labels, start=1):
            label_name = str(labels.get(label_value, f"Segment_{label_value}"))
            mask = frames == label_value

            seg_desc = seg.SegmentDescription(
                segment_number=number,
                segment_label=label_name,
                segmented_property_category=self._get_snomed_category(label_name),
                segmented_property_type=self._get_snomed_type(label_name),
                algorithm_type=seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
            )
            seg_desc.RecommendedDisplayCIELabValue = list(
                self._get_display_color(label_name, number))
            segment_descriptions.append(seg_desc)
            pixel_array[mask] = number

        # Create DICOM-SEG
        seg_dataset = seg.Segmentation(
            source_images=ref_datasets,
            pixel_array=pixel_array,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=segment_descriptions,
            series_instance_uid=generate_uid(),
            series_number=100,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer,
            software_versions=self._get_package_version(),
            device_serial_number="N/A",
            series_description=series_description,
        )
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seg_dataset.save_as(str(output_path), little_endian=True, implicit_vr=False)
        
        logger.info(f"DICOM-SEG saved to {output_path}")
        return output_path
    
    @staticmethod
    def _collect_reference_files(reference) -> List[str]:
        """Resolve the reference series to a list of DICOM file paths."""
        if isinstance(reference, (list, tuple)):
            return [str(f) for f in reference]
        reference = Path(reference)
        if reference.is_file():
            return [str(reference)]
        reader = sitk.ImageSeriesReader()
        return list(reader.GetGDCMSeriesFileNames(str(reference)))

    @staticmethod
    def _sort_source_images(datasets: List[Any]) -> List[Any]:
        """Order source images along the slice-normal axis."""
        from .dicom_utils import _position_key

        return sorted(datasets, key=_position_key)

    @staticmethod
    def _validate_geometry(seg_shape, seg_spacing, plane, rows, cols, slices, ref_datasets) -> None:
        """Fail loudly when the mask grid disagrees with the reference series."""
        expected = _EXPECTED_INPUT_SHAPE[plane](rows, cols, slices)
        if tuple(int(v) for v in seg_shape) != expected:
            raise ValueError(
                f"Segmentation shape {tuple(int(v) for v in seg_shape)} does not match "
                f"the DICOM grid: expected {expected} (columns, rows, frames) for the "
                f"{plane} plane with rows={rows}, columns={cols}, frames={slices}")

        if seg_spacing is None or plane != "axial":
            # Spacing mapping is only well-defined for the axial case we support.
            return
        pixel_spacing = getattr(ref_datasets[0], "PixelSpacing", None)
        if pixel_spacing is None or len(pixel_spacing) < 2:
            return
        expected_spacing = (
            float(pixel_spacing[1]),
            float(pixel_spacing[0]),
            _slice_spacing(ref_datasets),
        )
        for axis, (got, want) in enumerate(zip(seg_spacing, expected_spacing)):
            if want is None:
                continue
            if not np.isclose(got, want, rtol=1e-2, atol=1e-4):
                raise ValueError(
                    f"Segmentation spacing {seg_spacing} is inconsistent with the "
                    f"reference DICOM geometry (axis {axis}: {got} vs {want})")

    def _get_display_color(self, label_name: str, number: int) -> tuple:
        """CIELab display colour for a segment (mapping RGB, else palette)."""
        entry = self._get_mapping_entry(label_name)
        rgb = getattr(entry, "rgb", None) if entry is not None else None
        if not rgb or len(rgb) != 3:
            rgb = _DEFAULT_DISPLAY_COLORS[(number - 1) % len(_DEFAULT_DISPLAY_COLORS)]
        return rgb_to_cielab_dicom(rgb)

    @staticmethod
    def _get_package_version() -> str:
        """Return the installed nnunetsegmentator version string."""
        from importlib.metadata import version as _pkg_version

        try:
            return _pkg_version("nnunetsegmentator")
        except Exception:  # noqa: BLE001 - package metadata unavailable
            return "0.0.0"

    @classmethod
    def _get_algorithm_identification(cls):
        """Build the algorithm identification sequence for automatic segments."""
        from highdicom.seg.content import AlgorithmIdentificationSequence
        from pydicom.sr.codedict import codes

        return AlgorithmIdentificationSequence(
            name="nnunetsegmentator",
            family=codes.DCM.ArtificialIntelligence,
            version=cls._get_package_version(),
        )

    def _get_mapping_entry(self, label_name: str):
        """Resolve a label name through the central SNOMED label mapper."""
        try:
            from ..mapping import LabelMapper
        except ImportError:
            return None
        try:
            return LabelMapper.get_instance().resolve(label_name)
        except (ValueError, KeyError):
            return None

    def _get_snomed_category(self, label_name: str) -> Any:
        """Get SNOMED CT category code for a label via the central mapping."""
        from highdicom.sr.coding import CodedConcept

        entry = self._get_mapping_entry(label_name)
        if entry is not None and entry.category is not None:
            return CodedConcept(entry.category.code, entry.category.scheme, entry.category.meaning)

        # Default to anatomical structure
        return CodedConcept("123037004", "SCT", "Anatomical Structure")

    def _get_snomed_type(self, label_name: str) -> Any:
        """Get SNOMED CT type code for a label via the central mapping."""
        from highdicom.sr.coding import CodedConcept

        entry = self._get_mapping_entry(label_name)
        if entry is not None and entry.type is not None:
            return CodedConcept(entry.type.code, entry.type.scheme, entry.type.meaning)

        # Default to morphologically abnormal structure
        return CodedConcept("49755003", "SCT", "Morphologically Abnormal Structure")
