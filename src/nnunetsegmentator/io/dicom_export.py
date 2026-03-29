"""
DICOM Export Utilities

This module provides functionality for exporting segmentations to DICOM formats:
- DICOM-SEG (DICOM Segmentation)
- RTSTRUCT (Radiotherapy Structure Set)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
from .. import image as sitk

logger = logging.getLogger(__name__)


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
               metadata: Optional[Dict[str, Any]] = None,
               series_description: str = "Segmentation",
               manufacturer: str = "nnunetsegmentator") -> Path:
        """
        Export segmentation to DICOM-SEG.
        
        Args:
            segmentation: Segmentation image (SimpleITK or numpy array)
            reference_dicom_dir: Directory containing reference DICOM series
            output_path: Output DICOM-SEG file path
            labels: Dictionary mapping label values to label names
            metadata: Additional metadata
            series_description: Series description
            manufacturer: Manufacturer name
        
        Returns:
            Path to created DICOM-SEG file
        """
        from pydicom import dcmread
        from pydicom.uid import generate_uid
        import highdicom.seg as seg
        from highdicom.seg.enum import SegmentationType
        
        # Load reference DICOM series
        reference_dicom_dir = Path(reference_dicom_dir)
        dicom_files = sorted(reference_dicom_dir.glob("*.dcm"))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {reference_dicom_dir}")
        
        # Read reference DICOM datasets
        ref_datasets = [dcmread(str(f)) for f in dicom_files]
        
        # Convert segmentation to numpy if needed
        if isinstance(segmentation, sitk.Image):
            seg_array = sitk.GetArrayFromImage(segmentation)
        else:
            seg_array = segmentation
        
        # Create segment descriptions
        segment_descriptions = []
        for label_value, label_name in labels.items():
            if label_value == 0:  # Skip background
                continue
            
            # Create segment description
            seg_desc = seg.SegmentDescription(
                segment_number=label_value,
                segment_label=label_name,
                segmented_property_category=self._get_snomed_category(label_name),
                segmented_property_type=self._get_snomed_type(label_name),
                algorithm_type=seg.SegmentationAlgorithmValues.AUTOMATIC,
            )
            segment_descriptions.append(seg_desc)
        
        # Create DICOM-SEG
        seg_dataset = seg.Segmentation(
            source_images=ref_datasets,
            pixel_array=seg_array,
            segmentation_type=SegmentationType.BINARY,
            segment_descriptions=segment_descriptions,
            series_instance_uid=generate_uid(),
            series_number=100,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer=manufacturer,
            series_description=series_description,
        )
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seg_dataset.save_as(str(output_path))
        
        logger.info(f"DICOM-SEG saved to {output_path}")
        return output_path
    
    def _get_snomed_category(self, label_name: str) -> Any:
        """Get SNOMED CT category code for a label."""
        from highdicom.sr.coding import CodedConcept
        
        # Map common anatomical structures to SNOMED CT
        # This is a simplified mapping - a complete implementation would have
        # comprehensive SNOMED CT mappings
        organ_codes = {
            "liver": ("78730000", "SCT", "Liver structure"),
            "spleen": ("78961009", "SCT", "Spleen structure"),
            "kidney_right": ("64033007", "SCT", "Kidney structure"),
            "kidney_left": ("64033007", "SCT", "Kidney structure"),
            "lung_upper_lobe_left": ("45654000", "SCT", "Upper lobe of left lung structure"),
            "lung_lower_lobe_left": ("90596000", "SCT", "Lower lobe of left lung structure"),
            "heart": ("80248007", "SCT", "Heart structure"),
            "brain": ("12738006", "SCT", "Brain structure"),
        }
        
        if label_name in organ_codes:
            code, system, meaning = organ_codes[label_name]
            return CodedConcept(code, system, meaning)
        
        # Default to anatomical structure
        return CodedConcept("123037004", "SCT", "Anatomical Structure")
    
    def _get_snomed_type(self, label_name: str) -> Any:
        """Get SNOMED CT type code for a label."""
        from highdicom.sr.coding import CodedConcept
        
        # Default to morphologically abnormal structure
        return CodedConcept("49755003", "SCT", "Morphologically Abnormal Structure")


class RTSTRUCTExporter:
    """
    Export segmentations to DICOM RTSTRUCT format.
    
    RTSTRUCT is commonly used in radiation therapy planning.
    """
    
    def __init__(self):
        """Initialize RTSTRUCT exporter."""
        try:
            from rt_utils import RTStructBuilder
            self.RTStructBuilder = RTStructBuilder
        except ImportError:
            raise ImportError(
                "rt_utils is required for RTSTRUCT export. "
                "Install with: pip install rt_utils"
            )
    
    def export(self,
               segmentation: Union[sitk.Image, np.ndarray],
               reference_dicom_dir: Path,
               output_path: Path,
               labels: Dict[int, str],
               metadata: Optional[Dict[str, Any]] = None,
               series_description: str = "RTSTRUCT",
               color_map: Optional[Dict[str, str]] = None) -> Path:
        """
        Export segmentation to RTSTRUCT.
        
        Args:
            segmentation: Segmentation image
            reference_dicom_dir: Directory containing reference DICOM series
            output_path: Output RTSTRUCT file path
            labels: Dictionary mapping label values to label names
            metadata: Additional metadata
            series_description: Series description
            color_map: Optional color mapping for structures
        
        Returns:
            Path to created RTSTRUCT file
        """
        reference_dicom_dir = Path(reference_dicom_dir)
        output_path = Path(output_path)
        
        # Convert segmentation to numpy if needed
        if isinstance(segmentation, sitk.Image):
            seg_array = sitk.GetArrayFromImage(segmentation)
            seg_image = segmentation
        else:
            seg_array = segmentation
            seg_image = sitk.GetImageFromArray(seg_array)
        
        # Create RTSTRUCT builder
        rtstruct = self.RTStructBuilder.create_new(
            dicom_series_path=str(reference_dicom_dir)
        )
        
        # Default colors
        default_colors = [
            "255,0,0",      # Red
            "0,255,0",      # Green
            "0,0,255",      # Blue
            "255,255,0",    # Yellow
            "255,0,255",    # Magenta
            "0,255,255",    # Cyan
            "255,128,0",    # Orange
            "128,0,255",    # Purple
        ]
        
        # Add each structure
        for idx, (label_value, label_name) in enumerate(labels.items()):
            if label_value == 0:  # Skip background
                continue
            
            # Extract binary mask for this label
            mask = (seg_array == label_value).astype(np.uint8)
            
            # Skip if mask is empty
            if mask.sum() == 0:
                logger.warning(f"Empty mask for {label_name}, skipping")
                continue
            
            # Get color
            if color_map and label_name in color_map:
                color = color_map[label_name]
            else:
                color = default_colors[idx % len(default_colors)]
            
            # Add structure
            try:
                rtstruct.add_roi(
                    mask=mask,
                    name=label_name,
                    color=color
                )
                logger.debug(f"Added structure: {label_name}")
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.error(f"Failed to add structure {label_name}: {exc}")
                raise RuntimeError(f"Failed to add ROI '{label_name}' to RTSTRUCT") from exc
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rtstruct.save(str(output_path))
        
        logger.info(f"RTSTRUCT saved to {output_path}")
        return output_path


def export_segmentation(
    segmentation: Union[sitk.Image, np.ndarray],
    output_path: Path,
    format: str = "nifti",
    reference_dicom_dir: Optional[Path] = None,
    labels: Optional[Dict[int, str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Export segmentation to various formats.
    
    Args:
        segmentation: Segmentation image
        output_path: Output path
        format: Output format (nifti, dicom_seg, rtstruct)
        reference_dicom_dir: Reference DICOM directory (required for DICOM formats)
        labels: Label mapping
        metadata: Additional metadata
    
    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    
    if format == "nifti":
        # Simple NIfTI export
        if isinstance(segmentation, np.ndarray):
            segmentation = sitk.GetImageFromArray(segmentation)
        
        sitk.WriteImage(segmentation, str(output_path))
        logger.info(f"NIfTI saved to {output_path}")
        return output_path
    
    elif format == "dicom_seg":
        if reference_dicom_dir is None:
            raise ValueError("reference_dicom_dir required for DICOM-SEG export")
        if labels is None:
            raise ValueError("labels required for DICOM-SEG export")
        
        exporter = DICOMSEGExporter()
        return exporter.export(
            segmentation=segmentation,
            reference_dicom_dir=reference_dicom_dir,
            output_path=output_path,
            labels=labels,
            metadata=metadata
        )
    
    elif format == "rtstruct":
        if reference_dicom_dir is None:
            raise ValueError("reference_dicom_dir required for RTSTRUCT export")
        if labels is None:
            raise ValueError("labels required for RTSTRUCT export")
        
        exporter = RTSTRUCTExporter()
        return exporter.export(
            segmentation=segmentation,
            reference_dicom_dir=reference_dicom_dir,
            output_path=output_path,
            labels=labels,
            metadata=metadata
        )
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_multilabel_to_binary_masks(
    segmentation: Union[sitk.Image, np.ndarray],
    output_dir: Path,
    labels: Dict[int, str],
    prefix: str = ""
) -> Dict[str, Path]:
    """
    Export multilabel segmentation as separate binary masks.
    
    Args:
        segmentation: Multilabel segmentation
        output_dir: Output directory
        labels: Label mapping
        prefix: Optional prefix for filenames
    
    Returns:
        Dictionary mapping label names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy if needed
    if isinstance(segmentation, sitk.Image):
        seg_array = sitk.GetArrayFromImage(segmentation)
        reference_image = segmentation
    else:
        seg_array = segmentation
        reference_image = sitk.GetImageFromArray(seg_array)
    
    output_files = {}
    
    for label_value, label_name in labels.items():
        if label_value == 0:  # Skip background
            continue
        
        # Extract binary mask
        mask = (seg_array == label_value).astype(np.uint8)
        
        # Skip if empty
        if mask.sum() == 0:
            continue
        
        # Create SimpleITK image
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.CopyInformation(reference_image)
        
        # Save
        filename = f"{prefix}{label_name}.nii.gz" if prefix else f"{label_name}.nii.gz"
        filepath = output_dir / filename
        sitk.WriteImage(mask_image, str(filepath))
        
        output_files[label_name] = filepath
    
    logger.info(f"Exported {len(output_files)} binary masks to {output_dir}")
    return output_files
