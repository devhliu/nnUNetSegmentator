# nnunetsegmentator Documentation

Welcome to nnunetsegmentator - a unified framework for medical image segmentation using nnUNet models.

## Quick Start

### Installation

```bash
pip install nnunetsegmentator
```

### Basic Usage

```python
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name="gtrc")
result = orchestrator.segment("input.nii.gz", "output.nii.gz")
```

## Documentation Structure

### 🚀 Getting Started
- [Installation](getting-started/installation.md) - Setup and dependencies
- [Quick Start](getting-started/quick-start.md) - First steps
- [Supported Tasks](tasks/overview.md) - Available segmentation tasks

### 📖 User Guide
- [Python API](guide/python-api.md) - Using the Python interface
- [CLI Reference](guide/cli-reference.md) - Command line usage
- [Batch Processing](guide/batch-processing.md) - Processing multiple images
- [Configuration](guide/configuration.md) - Customizing behavior

### 🔧 Advanced Topics
- [Architecture](architecture.md) - System design
- [Custom Tasks](guide/custom-tasks.md) - Creating your own tasks
- [Task–Model Workflows](guide/task-model-workflows.md) - Task/model design and extension workflow
- [Pipeline System](guide/pipeline-system.md) - Building custom pipelines
- [Model Management](guide/model-management.md) - Working with models

### 📚 Reference
- [API Reference](reference/api-reference.md) - Complete API documentation
- [Tasks Reference](reference/tasks-reference.md) - Task details
- [Task–Model Mapping Reference](reference/task-model-mapping-reference.md) - Current task/model mappings and constraints
- [Configuration Reference](reference/configuration-reference.md) - Config options

### 🧪 Tutorials
- [Quick Start Tutorial](tutorials/quick_start.ipynb) - Basic usage
- [Advanced Tutorial](tutorials/advanced.ipynb) - Advanced features
- [Custom Tasks Tutorial](tutorials/custom_tasks.ipynb) - Custom pipelines

## Supported Tasks

| Task | Modality | Description |
|------|----------|-------------|
| GTRC-Net | PET/CT | Glioblastoma segmentation |
| LION | PET | Lesion identification |
| DEEP-PSMA | PET | PSMA lesion segmentation |
| TotalSegmentator | CT | 117 anatomical structures |
| DukeSeg | CT | 140 anatomical structures |

## Community

- [GitHub](https://github.com/devhliu/nnunetsegmentator) - Source code
- [Issues](https://github.com/devhliu/nnunetsegmentator/issues) - Report bugs
- [Discussions](https://github.com/devhliu/nnunetsegmentator/discussions) - Ask questions

## License

MIT License - see [LICENSE](../LICENSE) for details.
