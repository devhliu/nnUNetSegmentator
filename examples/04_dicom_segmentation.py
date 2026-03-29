#!/usr/bin/env python3
"""
DICOM Input/Output Segmentation Example

This example demonstrates:
1. Reading DICOM series as input
2. Performing segmentation
3. Exporting results as DICOM SEG (DICOM Segmentation)
4. Exporting results as DICOM RTSTRUCT (Radiotherapy Structure Set)

DICOM SEG is the modern standard for segmentation storage.
DICOM RTSTRUCT is widely used in radiotherapy planning systems.

Requirements:
- pydicom >= 2.3.0
- highdicom (for DICOM SEG)
- dicom2nifti (for DICOM to NIfTI conversion)

Usage:
    python examples/04_dicom_segmentation.py
"""

import os
import tempfile
from pathlib import Path
import numpy as np
from nnunetsegmentator import image as sitk

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid, UID
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not available. Install with: pip install pydicom>=2.3.0")

try:
    import highdicom as hd
    HIGHDICOM_AVAILABLE = True
except ImportError:
    HIGHDICOM_AVAILABLE = False
    print("Warning: highdicom not available. Install with: pip install highdicom")

from nnunetsegmentator import SegmentationOrchestrator, Config


def create_synthetic_dicom_series(
    output_dir: Path,
    num_slices: int = 10,
    rows: int = 128,
    columns: int = 128
) -> Path:
    """
    Create a synthetic DICOM series for demonstration.
    
    Args:
        output_dir: Directory to save DICOM files
        num_slices: Number of slices
        rows: Number of rows per slice
        columns: Number of columns per slice
    
    Returns:
        Path to DICOM series directory
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM operations")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DICOM files for each slice
    series_uid = generate_uid()
    study_uid = generate_uid()
    
    for i in range(num_slices):
        # Create FileDataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'  # Implicit VR Little Endian
        
        ds = FileDataset(
            output_dir / f"slice_{i:03d}.dcm",
            {},
            file_meta=file_meta,
            preamble=b"\x00" * 128
        )
        
        # Set patient and study information
        ds.PatientName = "Test^Patient"
        ds.PatientID = "TEST001"
        ds.Modality = "CT"
        ds.SeriesDescription = "Synthetic CT for Segmentation"
        
        # Set UIDs
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        
        # Set image properties
        ds.Rows = rows
        ds.Columns = columns
        ds.NumberOfFrames = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # Signed
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        # Set geometry
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [2.0, 2.0]  # 2mm x 2mm
        ds.SliceThickness = 2.0
        ds.SliceLocation = str(i * 2.0)
        ds.ImagePositionPatient = [-(columns/2)*2.0, -(rows/2)*2.0, i * 2.0]
        
        # Create synthetic CT data (soft tissue + bone)
        pixel_array = np.full((rows, columns), 40, dtype=np.int16)  # Soft tissue: 40 HU
        
        # Add bone structure (ellipse in center)
        center_r, center_c = rows // 2, columns // 2
        for r in range(rows):
            for c in range(columns):
                if ((r - center_r) / 30) ** 2 + ((c - center_c) / 40) ** 2 < 1:
                    pixel_array[r, c] = 300  # Bone: 300 HU
        
        ds.PixelData = pixel_array.tobytes()
        
        # Save DICOM file
        ds.save_as(output_dir / f"slice_{i:03d}.dcm")
    
    print(f"Created synthetic DICOM series: {output_dir}")
    print(f"  - {num_slices} slices")
    print(f"  - {rows}x{columns} pixels")
    print(f"  - Series UID: {series_uid}")
    
    return output_dir


def segment_dicom_to_nifti(
    dicom_dir: Path,
    output_dir: Path,
    task_name: str = "total"
) -> Path:
    """
    Perform segmentation on DICOM input and save as NIfTI.
    
    Args:
        dicom_dir: Directory containing DICOM series
        output_dir: Output directory
        task_name: Segmentation task name
    
    Returns:
        Path to segmentation NIfTI file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize orchestrator
    orchestrator = SegmentationOrchestrator(
        task_name=task_name,
        config=Config(output_dir=output_dir)
    )
    
    # Segment from DICOM directory
    print(f"\nSegmenting DICOM series from: {dicom_dir}")
    print(f"Using task: {task_name}")
    
    result = orchestrator.segment(
        input_data=str(dicom_dir),  # Pass DICOM directory
        output_path=output_dir / "segmentation.nii.gz"
    )
    
    print(f"Segmentation saved to: {output_dir / 'segmentation.nii.gz'}")
    print(f"Number of labels: {len(result.labels)}")
    
    return output_dir / "segmentation.nii.gz"


def export_dicom_seg(
    segmentation_path: Path,
    source_dicom_dir: Path,
    output_dir: Path,
    label_names: dict = None
) -> Path:
    """
    Export segmentation as DICOM SEG.
    
    DICOM SEG is the modern standard for storing segmentations.
    It stores each segment as a separate item with metadata.
    
    Args:
        segmentation_path: Path to segmentation NIfTI file
        source_dicom_dir: Directory containing source DICOM series
        output_dir: Output directory
        label_names: Dictionary mapping label values to names
    
    Returns:
        Path to DICOM SEG file
    """
    if not HIGHDICOM_AVAILABLE:
        print("\nWarning: highdicom not available. Skipping DICOM SEG export.")
        print("Install with: pip install highdicom")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting DICOM SEG...")
    
    # Read source DICOM series
    source_dicom_files = sorted(source_dicom_dir.glob("*.dcm"))
    source_ds = pydicom.dcmread(source_dicom_files[0])
    
    # Read segmentation
    seg_sitk = sitk.ReadImage(str(segmentation_path))
    seg_array = sitk.GetArrayFromImage(seg_sitk)
    
    # Create segment descriptions
    if label_names is None:
        # Default labels for TotalSegmentator
        label_names = {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            5: "liver",
        }
    
    segments = []
    for label_value, label_name in label_names.items():
        # Create binary mask for this label
        mask = (seg_array == label_value).astype(np.uint8)
        
        # Create segment
        segment = hd.seg.Segment(
            segment_number=label_value,
            segment_label=label_name,
            segmented_property_category=hd.seg.SegmentedPropertyCategoryValues(
                code_value="T-D0050",
                coding_scheme_designator="SRT",
                code_meaning="Tissue"
            ),
            segmented_property_type=hd.seg.SegmentedPropertyTypeValues(
                code_value="T-D0050",
                coding_scheme_designator="SRT",
                code_meaning=label_name
            ),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        )
        segments.append(segment)
    
    # Create DICOM SEG
    seg_ds = hd.seg.Segmentation(
        source_images=[pydicom.dcmread(f) for f in source_dicom_files],
        pixel_array=seg_array,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segments=segments,
        series_number=100,
        series_description="AI Segmentation",
        manufacturer="nnunetsegmentator",
        manufacturer_model_name="TotalSegmentator",
        software_versions="1.0",
        device_serial_number="001",
        content_description="Automatic segmentation using nnUNet",
        content_creator_name="AI Algorithm",
    )
    
    # Save DICOM SEG
    output_path = output_dir / "segmentation_seg.dcm"
    seg_ds.save_as(output_path)
    
    print(f"DICOM SEG saved to: {output_path}")
    print(f"  - Number of segments: {len(segments)}")
    print(f"  - Segments: {', '.join(label_names.values())}")
    
    return output_path


def export_dicom_rtstruct(
    segmentation_path: Path,
    source_dicom_dir: Path,
    output_dir: Path,
    label_names: dict = None
) -> Path:
    """
    Export segmentation as DICOM RTSTRUCT.
    
    DICOM RTSTRUCT is widely used in radiotherapy planning systems.
    It stores contours for each structure.
    
    Args:
        segmentation_path: Path to segmentation NIfTI file
        source_dicom_dir: Directory containing source DICOM series
        output_dir: Output directory
        label_names: Dictionary mapping label values to names
    
    Returns:
        Path to DICOM RTSTRUCT file
    """
    if not PYDICOM_AVAILABLE:
        print("\nWarning: pydicom not available. Skipping DICOM RTSTRUCT export.")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting DICOM RTSTRUCT...")
    
    # Read source DICOM series
    source_dicom_files = sorted(source_dicom_dir.glob("*.dcm"))
    source_ds = pydicom.dcmread(source_dicom_files[0])
    
    # Read segmentation
    seg_sitk = sitk.ReadImage(str(segmentation_path))
    seg_array = sitk.GetArrayFromImage(seg_sitk)
    
    # Default labels
    if label_names is None:
        label_names = {
            1: "Spleen",
            2: "Kidney_R",
            3: "Kidney_L",
            5: "Liver",
        }
    
    # Create RTSTRUCT dataset
    rtstruct = Dataset()
    
    # Set SOP Class and Instance UIDs
    rtstruct.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage
    rtstruct.SOPInstanceUID = generate_uid()
    
    # Set patient and study information (copy from source)
    rtstruct.PatientName = source_ds.PatientName
    rtstruct.PatientID = source_ds.PatientID
    rtstruct.StudyInstanceUID = source_ds.StudyInstanceUID
    rtstruct.SeriesInstanceUID = generate_uid()
    rtstruct.SeriesNumber = "100"
    rtstruct.Modality = "RTSTRUCT"
    
    # Set structure set information
    rtstruct.StructureSetLabel = "AI_Segmentation"
    rtstruct.StructureSetName = "AI Segmentation"
    rtstruct.StructureSetDate = "20240101"
    rtstruct.StructureSetTime = "120000"
    
    # Create ROI contours
    roi_contour_sequence = []
    roi_sequence = []
    
    for i, (label_value, label_name) in enumerate(label_names.items(), start=1):
        # Create ROI data
        roi = Dataset()
        roi.ROINumber = str(i)
        roi.ReferencedROINumber = str(i)
        roi.ROIName = label_name
        roi.ROIGenerationAlgorithm = "AUTOMATIC"
        roi.ROIVolume = 0.0  # Would need to calculate
        roi_sequence.append(roi)
        
        # Create contour data for each slice
        # This is simplified - real implementation would extract contours
        contour_sequence = []
        for slice_idx in range(seg_array.shape[0]):
            mask_slice = seg_array[slice_idx, :, :]
            if np.any(mask_slice == label_value):
                # Create contour (simplified - just bounding box)
                # Real implementation would use marching squares
                contour = Dataset()
                contour.ContourGeometricType = "CLOSED_PLANAR"
                contour.NumberOfContourPoints = "4"
                # Simplified contour coordinates
                contour.ContourData = [0.0, 0.0, slice_idx * 2.0,
                                      100.0, 0.0, slice_idx * 2.0,
                                      100.0, 100.0, slice_idx * 2.0,
                                      0.0, 100.0, slice_idx * 2.0]
                contour_sequence.append(contour)
        
        # Create ROI contour
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = str(i)
        roi_contour.ContourSequence = Sequence(contour_sequence)
        roi_contour_sequence.append(roi_contour)
    
    # Set sequences
    rtstruct.ROIContourSequence = Sequence(roi_contour_sequence)
    rtstruct.StructureSetROISequence = Sequence(roi_sequence)
    
    # Save RTSTRUCT
    output_path = output_dir / "segmentation_rtstruct.dcm"
    pydicom.dcmwrite(output_path, rtstruct)
    
    print(f"DICOM RTSTRUCT saved to: {output_path}")
    print(f"  - Number of structures: {len(label_names)}")
    print(f"  - Structures: {', '.join(label_names.values())}")
    
    return output_path


def main():
    """Main function demonstrating DICOM I/O segmentation."""
    
    print("=" * 70)
    print("DICOM Input/Output Segmentation Example")
    print("=" * 70)
    
    # Setup output directory
    output_dir = Path("example_output_dicom")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Create synthetic DICOM series
    print("\n" + "=" * 70)
    print("Step 1: Creating Synthetic DICOM Series")
    print("=" * 70)
    
    dicom_dir = output_dir / "dicom_input"
    dicom_dir = create_synthetic_dicom_series(
        output_dir=dicom_dir,
        num_slices=10,
        rows=128,
        columns=128
    )
    
    # Step 2: Perform segmentation
    print("\n" + "=" * 70)
    print("Step 2: Performing Segmentation")
    print("=" * 70)
    
    segmentation_path = segment_dicom_to_nifti(
        dicom_dir=dicom_dir,
        output_dir=output_dir,
        task_name="total"
    )
    
    # Step 3: Export as DICOM SEG
    print("\n" + "=" * 70)
    print("Step 3: Exporting DICOM SEG")
    print("=" * 70)
    
    dicom_seg_path = export_dicom_seg(
        segmentation_path=segmentation_path,
        source_dicom_dir=dicom_dir,
        output_dir=output_dir
    )
    
    # Step 4: Export as DICOM RTSTRUCT
    print("\n" + "=" * 70)
    print("Step 4: Exporting DICOM RTSTRUCT")
    print("=" * 70)
    
    rtstruct_path = export_dicom_rtstruct(
        segmentation_path=segmentation_path,
        source_dicom_dir=dicom_dir,
        output_dir=output_dir
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nInput:")
    print(f"  - DICOM series: {dicom_dir}")
    print(f"  - Number of slices: 10")
    
    print(f"\nOutput:")
    print(f"  - NIfTI segmentation: {segmentation_path}")
    if dicom_seg_path:
        print(f"  - DICOM SEG: {dicom_seg_path}")
    if rtstruct_path:
        print(f"  - DICOM RTSTRUCT: {rtstruct_path}")
    
    print("\n" + "=" * 70)
    print("DICOM Format Comparison:")
    print("=" * 70)
    print("\nDICOM SEG:")
    print("  - Modern standard (DICOM Supplement 172)")
    print("  - Stores each segment separately")
    print("  - Supports overlapping segments")
    print("  - Rich metadata per segment")
    print("  - Recommended for new applications")
    
    print("\nDICOM RTSTRUCT:")
    print("  - Traditional radiotherapy format")
    print("  - Stores contours (not voxels)")
    print("  - Widely supported in treatment planning systems")
    print("  - No overlapping structures")
    print("  - Use for radiotherapy workflows")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
