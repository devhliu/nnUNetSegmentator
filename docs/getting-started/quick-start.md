# Quick Start

## Basic Usage

### Python API

```python
from nnunetsegmentator import SegmentationOrchestrator

# Initialize orchestrator
orchestrator = SegmentationOrchestrator(task_name="gtrc")

# Segment a single image
result = orchestrator.segment(
    input_data="input.nii.gz",
    output_path="output.nii.gz"
)

# Access results
print(f"Segmentation shape: {result.segmentation.GetSize()}")
print(f"Labels: {list(result.labels.keys())}")
```

### Command Line Interface

```bash
# Basic segmentation
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc

# List available tasks
nnunetsegmentator list-tasks

# Show task information
nnunetsegmentator info -t gtrc
```

## Supported Tasks

| Task | Modality | Description |
|------|----------|-------------|
| gtrc | PET/CT | Glioblastoma segmentation |
| lion | PET | Lesion identification |
| deep_psma | PET | PSMA lesion segmentation |
| total | CT | 117 anatomical structures |
| dukeseg | CT | 140 anatomical structures |

## Multi-Modal Input

```python
# For tasks requiring multiple modalities (e.g., GTRC)
result = orchestrator.segment(
    input_data={
        'pet': 'pet.nii.gz',
        'ct': 'ct.nii.gz'
    },
    output_path='segmentation.nii.gz'
)
```

## Next Steps

- [Installation](installation.md) - Detailed setup instructions
- [User Guide](../guide/python-api.md) - Comprehensive usage guide
- [Tutorials](../tutorials/quick_start.ipynb) - Interactive examples
