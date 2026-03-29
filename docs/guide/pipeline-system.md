# Pipeline System Guide

## Pipeline Basics

### Creating a Pipeline

```python
from nnunetsegmentator.pipeline import Pipeline
from nnunetsegmentator.pipeline.steps.preprocessing import ResampleStep

# Create pipeline
pipeline = Pipeline(name='segmentation')

# Add steps
pipeline.add_step(ResampleStep('resample', {'spacing': [1.0, 1.0, 1.0]}))
```

### Pipeline Structure

```
Input → Preprocessing → Inference → Postprocessing → Output
```

## Pipeline Steps

### Preprocessing Steps

- `ResampleStep` - Resample image to target spacing
- `ClipIntensityStep` - Clip intensity values
- `NormalizeStep` - Normalize intensities
- `SUVThresholdStep` - Apply SUV threshold for PET
- `MultiChannelStackStep` - Stack multi-modal images

### Inference Steps

- `nnUNetInferenceStep` - Standard nnUNet inference
- `CascadeInferenceStep` - Cascade inference

### Postprocessing Steps

- `LargestComponentStep` - Keep largest components
- `MorphologicalOpsStep` - Morphological operations
- `FillHolesStep` - Fill holes
- `RemoveSmallObjectsStep` - Remove small objects

## Building Custom Pipelines

### Example Pipeline

```python
from nnunetsegmentator.pipeline import Pipeline
from nnunetsegmentator.pipeline.steps.preprocessing import (
    ResampleStep, NormalizeStep
)
from nnunetsegmentator.pipeline.steps.inference import nnUNetInferenceStep
from nnunetsegmentator.pipeline.steps.postprocessing import LargestComponentStep

# Build pipeline
pipeline = Pipeline(name='custom_segmentation')

# Preprocessing
pipeline.add_step(ResampleStep('resample', {'spacing': [1.0, 1.0, 1.0]}))
pipeline.add_step(NormalizeStep('normalize', {'method': 'ct'}))

# Inference
pipeline.add_step(nnUNetInferenceStep('inference', {'folds': [0, 1, 2]}))

# Postprocessing
pipeline.add_step(LargestComponentStep('largest', {'keep_top_n': 1}))
```

### Using Custom Pipeline

```python
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(
    task_name='gtrc',
    pipeline=pipeline
)
```

## Pipeline Composition

### Combining Pipelines

```python
preprocessing = Pipeline([resample, normalize])
inference = Pipeline([nnunet_inference])
postprocessing = Pipeline([largest_component])

full_pipeline = preprocessing | inference | postprocessing
```

## Best Practices

1. **Order matters** - Preprocessing → Inference → Postprocessing
2. **Test each step** - Verify pipeline components
3. **Use meaningful names** - For debugging
4. **Document pipeline** - For reproducibility
