#!/usr/bin/env python3
"""
Example: Multi-Modal Segmentation with GTRC-Net

This example demonstrates how to use nnunetsegmentator for multi-modal
PET/CT segmentation using GTRC-Net for glioblastoma treatment response.

Requirements:
- Install nnunetsegmentator
- Download GTRC-Net models (Git LFS required)
"""

import sys
from pathlib import Path
import numpy as np
from nnunetsegmentator import image as sitk

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnunetsegmentator import SegmentationOrchestrator, Config


def create_synthetic_pet_ct(shape=(128, 128, 128), spacing=(2.0, 2.0, 2.0)):
    """
    Create synthetic co-registered PET and CT images.
    
    PET image:
    - Background: 0 SUV
    - Normal tissue: 1.0 SUV
    - Tumor: 5.0 SUV (high uptake)
    
    CT image:
    - Background: -1000 HU (air)
    - Soft tissue: 40 HU
    - Tumor: 50 HU
    
    Args:
        shape: Image shape
        spacing: Voxel spacing in mm
        
    Returns:
        Tuple of (PET image, CT image)
    """
    # Create coordinate grids
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array(shape) // 2
    
    # Create CT
    ct_array = np.full(shape, -1000, dtype=np.float32)
    
    # Body
    body_mask = ((x - center[2])**2 / (shape[2]/3)**2 + 
                 (y - center[1])**2 / (shape[1]/3)**2 +
                 (z - center[0])**2 / (shape[0]/3)**2) <= 1
    ct_array[body_mask] = 40
    
    # Tumor (small sphere)
    tumor_center = center + np.array([0, 0, shape[2]//6])
    tumor_mask = ((x - tumor_center[2])**2 / 15**2 + 
                  (y - tumor_center[1])**2 / 15**2 +
                  (z - tumor_center[0])**2 / 15**2) <= 1
    ct_array[tumor_mask] = 50
    
    # Create PET
    pet_array = np.zeros(shape, dtype=np.float32)
    pet_array[body_mask] = 1.0  # Normal uptake
    pet_array[tumor_mask] = 5.0  # High tumor uptake
    
    # Convert to image objects
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetSpacing(spacing)
    
    pet_image = sitk.GetImageFromArray(pet_array)
    pet_image.SetSpacing(spacing)
    
    return pet_image, ct_image


def main():
    print("=" * 70)
    print("Example: Multi-Modal Segmentation with GTRC-Net")
    print("=" * 70)
    
    # Step 1: Create synthetic PET/CT
    print("\n1. Creating synthetic co-registered PET/CT...")
    pet_image, ct_image = create_synthetic_pet_ct()
    
    output_dir = Path("example_output_multimodal")
    output_dir.mkdir(exist_ok=True)
    
    # Save images
    pet_path = output_dir / "synthetic_pet.nii.gz"
    ct_path = output_dir / "synthetic_ct.nii.gz"
    sitk.WriteImage(pet_image, str(pet_path))
    sitk.WriteImage(ct_image, str(ct_path))
    print(f"   PET saved to: {pet_path}")
    print(f"   CT saved to: {ct_path}")
    
    # Step 2: Initialize GTRC-Net
    print("\n2. Initializing GTRC-Net...")
    config = Config(use_gpu=True, gpu_id=0)
    
    orchestrator = SegmentationOrchestrator(
        task_name="gtrc",
        config=config
    )
    
    # Step 3: Perform multi-modal segmentation
    print("\n3. Performing multi-modal segmentation...")
    print("   Note: Requires GTRC-Net models (Git LFS)")
    
    try:
        # Multi-modal input as dictionary
        result = orchestrator.segment(
            input_data={
                'pet': str(pet_path),
                'ct': str(ct_path)
            },
            output_path=str(output_dir / "gtrc_segmentation.nii.gz"),
            return_labels=True
        )
        
        # Step 4: Analyze results
        print("\n4. Segmentation results:")
        print(f"   Labels found: {list(result.labels.keys())}")
        
        for label_name, label_image in result.labels.items():
            label_array = sitk.GetArrayFromImage(label_image)
            volume = np.sum(label_array > 0) * np.prod(pet_image.GetSpacing())
            print(f"   - {label_name}: {volume:.1f} mm³")
        
        print("\n" + "=" * 70)
        print("Example completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure GTRC-Net models are downloaded:")
        print("  python scripts/download_models.py --task gtrc")


if __name__ == "__main__":
    main()
