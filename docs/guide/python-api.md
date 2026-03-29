# Python API Guide

## Basic Usage

### Initialization

```python
from nnunetsegmentator import SegmentationOrchestrator, Config

# Basic initialization
orchestrator = SegmentationOrchestrator(task_name="gtrc")

# With custom configuration
config = Config(use_gpu=True, num_workers=4)
orchestrator = SegmentationOrchestrator(task_name="gtrc", config=config)
```

### Single Image Segmentation

```python
# Basic segmentation
result = orchestrator.segment(
    input_data="input.nii.gz",
    output_path="output.nii.gz"
)

# With metrics
result = orchestrator.segment(
    input_data="input.nii.gz",
    output_path="output.nii.gz",
    compute_metrics=True
)
```

### Multi-Modal Segmentation

```python
# For tasks requiring multiple modalities
result = orchestrator.segment(
    input_data={
        'pet': 'pet.nii.gz',
        'ct': 'ct.nii.gz'
    },
    output_path='segmentation.nii.gz'
)
```

## Working with Results

```python
# Access segmentation
segmentation = result.segmentation

# Access individual labels
for label_name, label_image in result.labels.items():
    print(f"{label_name}: {label_image.GetSize()}")

# Access metrics
if result.metrics:
    print(f"Volume: {result.metrics['volume_mm3']:.2f} mm³")

# Convert to numpy
seg_array = result.get_array()
tumor_array = result.get_label_array('tumor_core')
```

## Batch Processing

```python
from pathlib import Path

# Process multiple files
input_files = list(Path("data/patients").glob("*.nii.gz"))

results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=4,
    progress_callback=lambda c, t, r: print(f"Progress: {c+1}/{t}")
)
```

## Configuration

```python
from nnunetsegmentator import Config

# Create configuration
config = Config(
    use_gpu=True,
    num_workers=4,
    batch_size=4,
    patch_size=[128, 128, 64]
)

# Load from file
config = Config.from_file("config.yaml")
```

## Next Steps

- [CLI Reference](cli-reference.md) - Command line usage
- [Batch Processing](batch-processing.md) - Advanced batch processing
- [Configuration](configuration.md) - Detailed configuration guide
