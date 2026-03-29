#!/usr/bin/env python3
"""
Example: Custom Pipeline Construction

This example demonstrates how to create custom processing pipelines
by combining different preprocessing, inference, and postprocessing steps.
"""

import sys
from pathlib import Path
import numpy as np
from nnunetsegmentator import image as sitk

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnunetsegmentator import SegmentationOrchestrator, Config
from nnunetsegmentator.pipeline import Pipeline
from nnunetsegmentator.pipeline.steps.preprocessing import (
    ResampleStep,
    ClipIntensityStep,
    NormalizeStep
)
from nnunetsegmentator.pipeline.steps.postprocessing import (
    LargestComponentStep,
    MorphologicalOpsStep
)


def create_test_image():
    """Create a simple test image."""
    array = np.random.rand(64, 64, 64).astype(np.float32) * 100
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((2.0, 2.0, 2.0))
    return image


def main():
    print("=" * 70)
    print("Example: Custom Pipeline Construction")
    print("=" * 70)
    
    # Step 1: Create custom pipeline
    print("\n1. Building custom pipeline...")
    
    custom_pipeline = Pipeline(name="custom_preprocessing")
    
    # Add preprocessing steps
    custom_pipeline.add_step(ResampleStep(
        name="resample_1mm",
        config={'spacing': (1.0, 1.0, 1.0)}
    ))
    
    custom_pipeline.add_step(ClipIntensityStep(
        name="clip_intensities",
        config={'lower': -100, 'upper': 100}
    ))
    
    custom_pipeline.add_step(NormalizeStep(
        name="normalize",
        config={'method': 'minmax'}
    ))
    
    print(f"   Pipeline: {custom_pipeline.name}")
    print(f"   Steps: {len(custom_pipeline)}")
    
    # Step 2: Create postprocessing pipeline
    print("\n2. Building postprocessing pipeline...")
    
    post_pipeline = Pipeline(name="custom_postprocessing")
    
    post_pipeline.add_step(LargestComponentStep(
        name="keep_largest",
        config={'keep_top_n': 3}
    ))
    
    post_pipeline.add_step(MorphologicalOpsStep(
        name="smooth",
        config={'operation': 'close', 'radius': 1}
    ))
    
    print(f"   Pipeline: {post_pipeline.name}")
    print(f"   Steps: {len(post_pipeline)}")
    
    # Step 3: Combine pipelines
    print("\n3. Combining pipelines...")
    full_pipeline = custom_pipeline | post_pipeline
    print(f"   Full pipeline steps: {len(full_pipeline)}")
    
    # Step 4: Use with orchestrator
    print("\n4. Using custom pipeline with orchestrator...")
    
    # Create test image
    test_image = create_test_image()
    output_dir = Path("example_output_custom")
    output_dir.mkdir(exist_ok=True)
    
    input_path = output_dir / "test_image.nii.gz"
    sitk.WriteImage(test_image, str(input_path))
    
    # Note: This would require a model to be specified
    # For demonstration, we just show the pipeline construction
    print("   Custom pipeline constructed successfully!")
    print("   To use with orchestrator:")
    print("     orchestrator = SegmentationOrchestrator(")
    print("         task_name='total',")
    print("         pipeline=full_pipeline")
    print("     )")
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
