# Installation

## Prerequisites

- Python >= 3.8
- pip package manager

## Installation Methods

### From PyPI (Recommended)

```bash
pip install nnunetsegmentator
```

### From Source

```bash
git clone https://github.com/devhliu/nnunetsegmentator.git
cd nnunetsegmentator
pip install -e ".[all]"
```

### Optional Dependencies

```bash
# For visualization
pip install nnunetsegmentator[visualization]

# For nnUNet models
pip install nnunetsegmentator[nnunet]

# For metrics exports
pip install nnunetsegmentator[metrics]

# For development
pip install nnunetsegmentator[dev]

# All dependencies
pip install nnunetsegmentator[all]
```

## Dependencies

### Core Dependencies
- Python >= 3.8
- numpy >= 1.20.0
- nibabel >= 3.2.0
- scipy >= 1.7.0
- pydicom >= 2.3.0
- nibabel >= 3.2.0
- PyYAML >= 5.4.0

### Optional Dependencies
- matplotlib >= 3.5.0 (visualization)
- pandas >= 1.3.0 (metrics export)
- TotalSegmentator >= 2.0.0 (vertebrae localization and TotalSegmentator tasks)

## nnUNet Setup

To use nnUNet models, configure the environment:

### Environment Variables

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### Python Configuration

```python
from nnunetsegmentator import Config

config = Config(
    nnunet_raw="/path/to/nnUNet_raw",
    nnunet_preprocessed="/path/to/nnUNet_preprocessed",
    nnunet_results="/path/to/nnUNet_results"
)
```

## Verify Installation

```python
from nnunetsegmentator import SegmentationOrchestrator, TaskRegistry

# List available tasks
tasks = TaskRegistry.list_tasks()
print("Available tasks:", list(tasks.keys()))

# Test basic functionality
orchestrator = SegmentationOrchestrator(task_name="gtrc")
print("Installation successful!")
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **nnUNet path errors**: Verify environment variables are set
3. **GPU issues**: Check CUDA installation and GPU availability

### Getting Help

- Check [GitHub Issues](https://github.com/devhliu/nnunetsegmentator/issues)
- Review [Tutorials](../tutorials/quick_start.ipynb)
