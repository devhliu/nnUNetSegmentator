#!/usr/bin/env python3
"""
Example: Basic Segmentation with TotalSegmentator

This example demonstrates how to use nnunetsegmentator for basic
whole-body CT segmentation using TotalSegmentator.

Requirements:
- Install nnunetsegmentator
- Download TotalSegmentator models (automatic on first run)
"""

import sys
from pathlib import Path
import numpy as np
from nnunetsegmentator import image as sitk

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnunetsegmentator import SegmentationOrchestrator, Config
from nnunetsegmentator.core.registry import TaskRegistry


def create_synthetic_ct(shape=(128, 128, 128), spacing=(2.0, 2.0, 2.0)):
    """
    Create a synthetic CT image for demonstration.
    
    This creates a simple phantom with:
    - Background (air): -1000 HU
    - Soft tissue: 40 HU
    - Bone: 300 HU
    
    Args:
        shape: Image shape
        spacing: Voxel spacing in mm
        
    Returns:
        Image object
    """
    # Create array
    array = np.full(shape, -1000, dtype=np.float32)  # Air
    
    # Add elliptical body
    center = np.array(shape) // 2
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Body (soft tissue)
    body_mask = ((x - center[2])**2 / (shape[2]/3)**2 + 
                 (y - center[1])**2 / (shape[1]/3)**2 +
                 (z - center[0])**2 / (shape[0]/3)**2) <= 1
    array[body_mask] = 40  # Soft tissue
    
    # Add bone structure (spine)
    spine_mask = ((x - center[2])**2 / 10**2 + 
                  (y - center[1] - shape[1]/4)**2 / 10**2) <= 1
    array[spine_mask] = 300  # Bone
    
    # Convert to Image object
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin((0.0, 0.0, 0.0))
    
    return image


def main():
    print("=" * 70)
    print("Example: Basic Segmentation with TotalSegmentator")
    print("=" * 70)
    
    # Step 1: Create synthetic CT image
    print("\n1. Creating synthetic CT image...")
    ct_image = create_synthetic_ct()
    print(f"   Image size: {ct_image.GetSize()}")
    print(f"   Image spacing: {ct_image.GetSpacing()}")
    
    # Save synthetic image
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    
    input_path = output_dir / "synthetic_ct.nii.gz"
    sitk.WriteImage(ct_image, str(input_path))
    print(f"   Saved to: {input_path}")
    
    # Step 2: List available tasks
    print("\n2. Available tasks:")
    tasks = TaskRegistry.list_tasks()
    for name, desc in list(tasks.items())[:5]:  # Show first 5
        print(f"   - {name}: {desc}")
    
    # Step 3: Initialize orchestrator
    print("\n3. Initializing TotalSegmentator...")
    config = Config(
        use_gpu=True,
        gpu_id=0,
        log_level="INFO"
    )
    
    orchestrator = SegmentationOrchestrator(
        task_name="total",
        config=config
    )
    print("   Orchestrator initialized")
    
    # Step 4: Perform segmentation
    print("\n4. Performing segmentation...")
    print("   Note: This will download models on first run (~2GB)")
    
    try:
        result = orchestrator.segment(
            input_data=str(input_path),
            output_path=str(output_dir / "segmentation.nii.gz"),
            return_labels=True
        )
        
        # Step 5: Analyze results
        print("\n5. Segmentation results:")
        print(f"   Output shape: {result.segmentation.GetSize()}")
        print(f"   Number of labels: {len(result.labels)}")
        
        # Print some label names
        print("\n   Sample labels found:")
        for i, (label_name, label_image) in enumerate(list(result.labels.items())[:5]):
            label_array = sitk.GetArrayFromImage(label_image)
            volume = np.sum(label_array) * np.prod(ct_image.GetSpacing())
            print(f"   - {label_name}: {volume:.1f} mm³")
        
        # Step 6: Save individual labels
        print("\n6. Saving individual labels...")
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        for label_name, label_image in result.labels.items():
            label_path = labels_dir / f"{label_name}.nii.gz"
            sitk.WriteImage(label_image, str(label_path))
        
        print(f"   Saved {len(result.labels)} labels to: {labels_dir}")
        
        print("\n" + "=" * 70)
        print("Example completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during segmentation: {e}")
        print("This is expected if models are not downloaded yet.")
        print("Run: python scripts/download_models.py --task total")


if __name__ == "__main__":
    main()
