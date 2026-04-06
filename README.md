<div align="center">

# nnUNetSegmentator

**A Unified Framework for Medical Image Segmentation**

[![PyPI version](https://badge.fury.io/py/nnunetsegmentator.svg)](https://badge.fury.io/py/nnunetsegmentator)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](https://pytest.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://mypy-lang.org)

[📖 Documentation](docs/) • [🚀 Quick Start](#quick-start) • [📦 Installation](#installation) • [🎯 Examples](examples/) • [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**nnUNetSegmentator** is a comprehensive Python framework that provides a **unified interface** for multiple state-of-the-art nnUNet-based medical image segmentation models. It simplifies access to 22+ pretrained segmentation models across various modalities (CT, MR, PET/CT) without requiring deep expertise in each model's implementation.

### Why nnUNetSegmentator?

| Challenge | Solution |
|-----------|----------|
| 🔴 **Fragmented ecosystem** - Each model has different APIs, dependencies, and workflows | ✅ **Unified API** - Single consistent interface for all models |
| 🔴 **Complex setup** - Manual model downloads, environment configuration, path management | ✅ **Automatic management** - Models auto-downloaded, environments auto-configured |
| 🔴 **Steep learning curve** - Need to understand each model's preprocessing, inference, postprocessing | ✅ **Pre-built pipelines** - Ready-to-use processing pipelines for each task |
| 🔴 **Limited extensibility** - Hard to customize or combine models | ✅ **Modular design** - Composable pipeline steps, easy to extend |
| 🔴 **No production readiness** - Research code not suitable for clinical use | ✅ **Production ready** - Comprehensive error handling, logging, testing |

### Target Users

- **Researchers**: Quickly experiment with multiple segmentation models
- **Clinicians**: Deploy segmentation models in clinical workflows
- **Developers**: Build medical imaging applications with segmentation capabilities
- **Data Scientists**: Integrate segmentation into ML pipelines

---

## Key Features

### 🎯 Core Capabilities

- **22+ Segmentation Models**: GTRC-Net, LION, DEEP-PSMA, TotalSegmentator, DukeSeg, MRSegmentator, and more
- **Multi-Modality Support**: CT, MR (T1, T2, Dixon), PET, PET/CT, SPECT/CT
- **Flexible Pipelines**: Composable preprocessing, inference, and postprocessing steps
- **Multi-Format I/O**: NIfTI, DICOM, MHA, NRRD support
- **Batch Processing**: Parallel processing with threading or multiprocessing
- **Production Ready**: Comprehensive error handling, logging, and testing

### 🛠️ Developer Experience

- **Clean API**: Intuitive Python API with full type hints
- **CLI Tools**: Command-line interface for quick usage
- **Auto-Discovery**: Automatic task and model registration
- **Extensible**: Easy to add custom tasks, models, and pipeline steps
- **Well-Documented**: Comprehensive documentation with examples
- **Type-Safe**: Full type annotations for better IDE support

### 📦 Package Quality

- **Modern Packaging**: PEP 517/518 compliant with `pyproject.toml`
- **Dependency Management**: Minimal core dependencies, optional feature groups
- **Code Quality**: Black, isort, mypy, flake8 configured
- **Test Coverage**: Comprehensive test suite with pytest
- **CI/CD Ready**: Pre-commit hooks and continuous integration setup

---

## Supported Models

### Core Segmentation Models

| Model | Modality | Structures | Resolution | Citation |
|-------|----------|------------|------------|----------|
| **GTRC-Net** | PET/CT | Tumor burden (PSMA, FDG, LuPSMA) | 1.5mm | [DOI](https://doi.org/10.1148/ryai.240777) |
| **LION** | PET | Lesions (FDG, PSMA) | 1.0mm | [DOI](https://doi.org/10.5281/zenodo.12626789) |
| **DEEP-PSMA** | PET | PSMA lesions | 2.0mm | [GitHub](https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA) |
| **TotalSegmentator** | CT | 117 structures | 1.5mm | [DOI](https://doi.org/10.1148/ryai.230024) |
| **TotalSegmentator MR** | MR | 50 structures | 1.5mm | [DOI](https://doi.org/10.1148/ryai.230024) |
| **MRSegmentator** | MR/CT | 40 structures | 1.5mm | [DOI](https://doi.org/10.1148/ryai.240777) |
| **DukeSeg** | CT | 140 structures | 1.5mm | [arXiv](https://arxiv.org/abs/2405.11133) |
| **TotalSegmentator 2D** | CT | 117 structures | Multi-scale | [Zenodo](https://zenodo.org/records/16985939) |

### Specialized Tasks

- **Lung**: Vessels, nodules, airways
- **Liver**: Vessels, lesions, segments
- **Cardiac**: Heart chambers, cardiac structures
- **Abdominal**: Kidney cysts, tissue types
- **Neuro**: Cerebral bleed, brain structures
- **Musculoskeletal**: Vertebrae, body composition
- **Other**: Body segmentation, pleural effusion

**Total: 22 segmentation tasks** across multiple anatomical regions and modalities.

See [docs/tasks/overview.md](docs/tasks/overview.md) for complete details.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda
- Git and Git LFS (for some models)

### Quick Install

```bash
# Install from PyPI (recommended)
pip install nnunetsegmentator

# Or install from source
git clone https://github.com/devhliu/nnunetsegmentator.git
cd nnunetsegmentator
pip install -e ".[all]"
```

### Installation Options

```bash
# Minimal installation (core functionality only)
pip install nnunetsegmentator

# With nnUNet support
pip install nnunetsegmentator[nnunet]

# With visualization tools
pip install nnunetsegmentator[visualization]

# With development tools
pip install nnunetsegmentator[dev]

# Complete installation (all features)
pip install nnunetsegmentator[all]
```

### Model Setup

Models are automatically downloaded on first use. To pre-download:

```bash
# List available models
nnunetsegmentator list-models

# Download specific model
nnunetsegmentator download-model --task gtrc

# Download all models
nnunetsegmentator download-model --task all
```

For detailed installation instructions, see [docs/getting-started/installation.md](docs/getting-started/installation.md).

---

## Quick Start

### Python API

```python
from nnunetsegmentator import SegmentationOrchestrator

# Initialize with a task
orchestrator = SegmentationOrchestrator(task_name='total')

# Segment an image
result = orchestrator.segment(
    input_data='patient_ct.nii.gz',
    output_path='segmentation.nii.gz'
)

# Access results
print(f"Segmentation shape: {result.segmentation.shape}")
print(f"Number of labels: {len(result.labels)}")

# Access individual structures
liver_mask = result.get_label_array('liver')
kidney_left_mask = result.get_label_array('kidney_left')
```

### Multi-Modal Segmentation (PET/CT)

```python
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name='gtrc')

# Segment with co-registered PET and CT
result = orchestrator.segment(
    input_data={
        'pet': 'pet_image.nii.gz',
        'ct': 'ct_image.nii.gz'
    },
    output_path='tumor_segmentation.nii.gz'
)

# Access tumor burden
tumor_burden = result.get_label_array('tumor_burden')
```

### Command-Line Interface

```bash
# Basic segmentation
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t total

# Batch processing
nnunetsegmentator batch -i input_dir/ -o output_dir/ -t total --workers 4

# List available tasks
nnunetsegmentator list-tasks

# Show task information
nnunetsegmentator info --task gtrc
```

---

## Usage Examples

### Example 1: Basic CT Segmentation

```python
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name='total')
result = orchestrator.segment('ct_scan.nii.gz', 'output.nii.gz')

# Get volume statistics
for label_name, label_array in result.labels.items():
    volume_mm3 = result.compute_volume(label_name)
    print(f"{label_name}: {volume_mm3:.2f} mm³")
```

### Example 2: Custom Pipeline

```python
from nnunetsegmentator import SegmentationOrchestrator
from nnunetsegmentator.pipeline import Pipeline
from nnunetsegmentator.pipeline.steps.preprocessing import ResampleStep

# Create custom pipeline
pipeline = Pipeline(name="custom")
pipeline.add_step(ResampleStep('resample', {'spacing': (2.0, 2.0, 2.0)}))

# Use custom pipeline
orchestrator = SegmentationOrchestrator(task_name='total', pipeline=pipeline)
result = orchestrator.segment('input.nii.gz', 'output.nii.gz')
```

### Example 3: Batch Processing

```python
from pathlib import Path
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name='total')

# Process multiple files
input_files = list(Path("data/").glob("*.nii.gz"))
results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=4
)

print(f"Processed: {sum(r is not None for r in results)}/{len(input_files)}")
```

See [examples/](examples/) directory for more examples.

---

## Architecture

```
nnunetsegmentator/
├── src/nnunetsegmentator/          # Main package
│   ├── core/                       # Core components
│   │   ├── orchestrator.py         # Main segmentation orchestrator
│   │   ├── registry.py             # Task and model registry
│   │   └── config.py               # Configuration management
│   ├── io/                         # Input/Output handling
│   │   ├── readers.py              # NIfTI, DICOM readers
│   │   └── writers.py              # Output writers
│   ├── pipeline/                   # Processing pipelines
│   │   ├── base.py                 # Base pipeline classes
│   │   └── steps/                  # Pipeline steps
│   │       ├── preprocessing.py    # Preprocessing steps
│   │       ├── inference.py        # Inference steps
│   │       └── postprocessing.py   # Postprocessing steps
│   ├── tasks/                      # Task definitions
│   │   ├── gtrc.py                 # GTRC-Net task
│   │   ├── lion.py                 # LION task
│   │   ├── totalsegmentator.py     # TotalSegmentator task
│   │   └── ...                     # Other tasks
│   ├── cli/                        # Command-line interface
│   └── utils/                      # Utility functions
├── docs/                           # Documentation
├── examples/                       # Example scripts
├── tests/                          # Test suite
├── scripts/                        # Utility scripts
└── pyproject.toml                  # Package configuration
```

### Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add new tasks, models, and pipeline steps
3. **Type Safety**: Full type hints for better developer experience
4. **Separation of Concerns**: Clear separation between IO, processing, and inference
5. **Configuration Driven**: Behavior controlled through configuration files

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

---

## API Reference

### Core Classes

#### `SegmentationOrchestrator`

Main class for managing segmentation tasks.

```python
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(
    task_name='total',           # Task name
    config=None,                 # Optional configuration
    model_path=None,             # Optional model path override
    pipeline=None                # Optional custom pipeline
)

# Single image segmentation
result = orchestrator.segment(
    input_data='image.nii.gz',
    output_path='output.nii.gz',
    return_labels=True,
    compute_metrics=True
)

# Batch processing
results = orchestrator.segment_batch(
    input_list=['img1.nii.gz', 'img2.nii.gz'],
    output_dir='./output',
    num_workers=4
)
```

#### `SegmentationResult`

Container for segmentation results.

```python
result = orchestrator.segment('image.nii.gz', 'output.nii.gz')

# Access segmentation array
segmentation = result.get_array()  # numpy.ndarray

# Access individual labels
for label_name, label_image in result.labels.items():
    label_array = result.get_label_array(label_name)

# Compute metrics
volume = result.compute_volume('liver')  # in mm³

# Access metadata
print(result.metadata['spacing'])
print(result.metadata['origin'])
```

#### `Config`

Configuration management.

```python
from nnunetsegmentator import Config

# Create from file
config = Config.from_file('config.yaml')

# Create programmatically
config = Config(
    use_gpu=True,
    num_workers=4,
    model_dir='~/.nnunetsegmentator/models'
)

# Setup nnUNet environment
config.setup_nnunet_environment()
```

See [docs/reference/api-reference.md](docs/reference/api-reference.md) for complete API documentation.

---

## Configuration

### Configuration File

Create `config.yaml`:

```yaml
# Paths
data_dir: ./data
output_dir: ./output
model_dir: ~/.nnunetsegmentator/models

# Processing
num_workers: 4
use_gpu: true
gpu_id: 0
batch_size: 1

# nnUNet environment
nnunet_raw: /path/to/nnunet_workspace/raw
nnunet_preprocessed: /path/to/nnunet_workspace/preprocessed
nnunet_results: /path/to/nnunet_workspace/results

# Logging
log_level: INFO
log_file: ./logs/segmentation.log

# Custom settings
custom:
  preprocessing:
    resample_spacing: [1.0, 1.0, 1.0]
  postprocessing:
    keep_top_n: 3
```

### Environment Variables

```bash
# Model storage location
export NNUNETSEGMENTATOR_MODEL_ROOTPATH=/custom/model/path

# nnUNet workspace
export NNUNETSEGMENTATOR_NNUNET_WORKSPACE=/path/to/workspace

# Or set nnUNet paths directly
export nnUNet_raw=/path/to/raw
export nnUNet_preprocessed=/path/to/preprocessed
export nnUNet_results=/path/to/results
```

See [docs/guide/configuration.md](docs/guide/configuration.md) for detailed configuration guide.

---

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/devhliu/nnunetsegmentator.git
cd nnunetsegmentator

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/nnunetsegmentator --cov-report=html

# Type checking
mypy src/nnunetsegmentator

# Code formatting
black src/nnunetsegmentator
isort src/nnunetsegmentator

# Linting
flake8 src/nnunetsegmentator
```

### Project Structure

- **Source Code**: `src/nnunetsegmentator/`
- **Tests**: `tests/`
- **Documentation**: `docs/`
- **Examples**: `examples/`
- **Scripts**: `scripts/`

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Static type checking
- **flake8**: Linting
- **pytest**: Testing
- **pre-commit**: Git hooks

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 **Report bugs**: Open an issue with detailed reproduction steps
- 💡 **Suggest features**: Open an issue with clear feature description
- 📝 **Improve documentation**: Fix typos, add examples, clarify explanations
- 🔧 **Submit code**: Fix bugs, add features, improve performance
- ✅ **Write tests**: Add tests for new functionality or existing code

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Citation

If you use nnUNetSegmentator in your research, please cite:

```bibtex
@software{nnunetsegmentator2024,
  author = {devhliu},
  title = {nnUNetSegmentator: A Unified Framework for Medical Image Segmentation},
  year = {2024},
  url = {https://github.com/devhliu/nnunetsegmentator},
  version = {0.2.0}
}
```

**Important**: Please also cite the original papers for the specific models you use. See [Acknowledgments](#acknowledgments) for citation information.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The underlying segmentation models have their own licenses. Please respect the original authors' licensing terms when using specific models.

---

## Acknowledgments

This framework integrates models from the following research groups. **We gratefully acknowledge their work**:

| Model | Authors | Citation | License |
|-------|---------|----------|---------|
| **GTRC-Net** | Peter MacCallum Cancer Centre | [DOI](https://doi.org/10.1148/ryai.240777) | Apache-2.0 |
| **LION** | ENHANCE-PET Consortium | [DOI](https://doi.org/10.5281/zenodo.12626789) | Apache-2.0 |
| **DEEP-PSMA** | Peter MacCallum Cancer Centre | [GitHub](https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA) | Apache-2.0 |
| **TotalSegmentator** | Wasserthal et al. | [DOI](https://doi.org/10.1148/ryai.230024) | Apache-2.0 |
| **MRSegmentator** | Häntze et al. | [DOI](https://doi.org/10.1148/ryai.240777) | Apache-2.0 |
| **DukeSeg** | Duke University CVIT | [arXiv](https://arxiv.org/abs/2405.11133) | MIT |
| **TotalSegmentator 2D** | RISC-MI | [Zenodo](https://zenodo.org/records/16985939) | Apache-2.0 |

**If you use any of these models, please cite the original publications.**

---

## Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/devhliu/nnunetsegmentator/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/devhliu/nnunetsegmentator/discussions)
- 💬 **Questions**: [GitHub Discussions](https://github.com/devhliu/nnunetsegmentator/discussions)

---

<div align="center">

**Made with ❤️ by devhliu**

[⬆ Back to Top](#nnunetsegmentator)

</div>
