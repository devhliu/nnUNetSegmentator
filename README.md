# nnunetsegmentator

[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](https://pytest.org)
[![Type Checking](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://www.mypy-lang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)

A comprehensive framework for managing multiple nnUNet-based segmentation tasks with support for complex processing pipelines, multiple I/O formats, and both Python and CLI interfaces.

## Purpose & Scope

**nnunetsegmentator** provides a **universal nnUNetv2-based segmentation framework** that unifies multiple state-of-the-art medical image segmentation models under a single, easy-to-use interface. This repository serves as a comprehensive toolkit for researchers and clinicians to access various pretrained segmentation models without needing to understand the underlying implementation details of each model.

### What Makes It Unique

- **Unified Interface**: Access 21+ segmentation models with a single API
- **Flexible Pipelines**: Build custom processing workflows from reusable steps
- **Production Ready**: Battle-tested in clinical research environments
- **Developer Friendly**: Clean architecture, comprehensive documentation, and extensive examples
- **Extensible**: Easy to add new tasks, models, and pipeline steps

### Supported Tasks & Models

This framework currently supports **21 segmentation tasks** across multiple modalities (CT, MR, PET/CT):

**Core Segmentation Models:**
- **GTRC-Net** - Glioblastoma treatment response (PET/CT)
- **LION** - PET lesion segmentation (FDG/PSMA)
- **DEEP-PSMA** - PSMA PET lesion segmentation
- **TotalSegmentator** - Whole-body CT/MR segmentation (117/50 structures)
- **TotalSegmentator 2D** - Fast 2D projection-based segmentation
- **DukeSeg** - Comprehensive segmentation (140 structures)

**Specialized Tasks:**
- Lung vessels, lung nodules, liver vessels, liver lesions
- Kidney cysts, heart chambers, cerebral bleed
- Tissue types, body segmentation, vertebrae bodies
- Pleural/pericardial effusion

For a complete list, see [Available Tasks](#available-tasks) below or [docs/tasks/overview.md](docs/tasks/overview.md).

### Acknowledgments & Attribution

**IMPORTANT**: All segmentation models, pretrained weights, and algorithms integrated in this framework are developed by their respective original authors. This repository merely provides a unified interface to access these models.

**We gratefully acknowledge the following original works:**

- **GTRC-Net**: Peter MacCallum Cancer Centre - [GitHub](https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained)
- **LION**: ENHANCE-PET Consortium - [GitHub](https://github.com/ENHANCE-PET/LION) | DOI: [10.5281/zenodo.12626789](https://doi.org/10.5281/zenodo.12626789)
- **DEEP-PSMA**: Peter MacCallum Cancer Centre - [GitHub](https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA)
- **TotalSegmentator**: Wasserthal et al. - [GitHub](https://github.com/wasserth/TotalSegmentator) | DOI: [10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024)
- **TotalSegmentator 2D**: RISC-MI - [GitHub](https://github.com/risc-mi/totalsegmentator2D) | Zenodo: [16985939](https://zenodo.org/records/16985939)
- **DukeSeg**: Duke University CVIT - [GitLab](https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git) | arXiv: [2405.11133](https://arxiv.org/abs/2405.11133)

**If you use any of these models in your research, please cite the original publications as listed above.** This framework is provided as a convenience tool and does not claim ownership of any underlying models or algorithms.

## Features

- ✅ **Unified API**: Single interface for multiple segmentation tasks
- ✅ **Flexible Pipelines**: Composable preprocessing, inference, and postprocessing steps
- ✅ **Multi-format Support**: NIfTI and DICOM I/O
- ✅ **Batch Processing**: Parallel processing with threading or multiprocessing
- ✅ **Task Registry**: Easy registration and discovery of segmentation tasks
- ✅ **CLI Interface**: Command-line tools for easy usage
- ✅ **nnUNet Integration**: Seamless integration with nnUNetv2
- ✅ **Automatic Model Management**: Models are automatically downloaded and stored
- ✅ **Type Safety**: Full type hints for better IDE support
- ✅ **Production Ready**: Comprehensive error handling and logging

## Installation

```bash
# Clone the repository
git clone https://github.com/devhliu/nnunetsegmentator.git
cd nnunetsegmentator

# Install in development mode (recommended)
pip install -e ".[all]"

# Or install with specific features
pip install -e ".[dev]"      # Development dependencies
pip install -e ".[docs]"     # Documentation dependencies
```

### Prerequisites

- Python >= 3.8
- pip or conda
- Git and Git LFS (for some models)

### Download Pretrained Models

Before using the segmentation tasks, you need to download the pretrained model weights:

```bash
# Install download dependencies
pip install requests tqdm

# For GTRC-Net and DEEP-PSMA, install Git LFS
# Ubuntu/Debian:
sudo apt-get install git-lfs
git lfs install

# macOS:
brew install git-lfs
git lfs install

# Windows:
# Download from https://git-lfs.github.com/

# Download models
cd scripts
python download_models.py --list  # List available models
python download_models.py --task all --output-dir ~/.nnunetsegmentator/models
```

### Install Local Model Files (archive or directory)

You can install task model files from local paths (unzipped directory or archive).
Identifiers must use canonical format: Dataset<数字>_<model_name>.

```bash
nnunetsegmentator install-models \
  --task total \
  --model Dataset291_total_organs=/local/total_organs.zip \
  --model Dataset292_total_vertebrae=/local/total_vertebrae/
```

### Migrate Existing nnUNet Workspace Results

If you have models under `~/.nnunetsegmentator/nnunet_workspace/results`,
convert them into canonical model storage:

```bash
python scripts/migrate_workspace_models.py --dry-run
python scripts/migrate_workspace_models.py
```

**Model Sources:**
- **GTRC-Net**: [GitHub (Git LFS)](https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained)
- **LION**: AWS S3 (auto-downloaded)
- **DEEP-PSMA**: [GitHub (Git LFS)](https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA)

For detailed download instructions, see [scripts/README.md](scripts/README.md).

## Quick Start

### Python API

```python
from nnunetsegmentator import SegmentationOrchestrator, Config

# Initialize with a task
orchestrator = SegmentationOrchestrator(
    task_name='gtrc',
    config=Config(use_gpu=True, num_workers=4)
)

# Segment single image
result = orchestrator.segment(
    input_data='path/to/image.nii.gz',
    output_path='path/to/output.nii.gz',
    return_labels=True,
    compute_metrics=True
)

# Access results
print(f"Segmentation shape: {result.segmentation.GetSize()}")
print(f"Labels: {list(result.labels.keys())}")

if result.metrics:
    print(f"Volume: {result.metrics['volume_mm3']:.2f} mm³")

# Process individual labels
for label_name, label_image in result.labels.items():
    label_array = result.get_label_array(label_name)
    volume = np.sum(label_array > 0)
    print(f"{label_name}: {volume} voxels")
```

### Multi-Modal Segmentation (PET/CT)

```python
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name='gtrc')

# Multi-modal input (PET and CT must be co-registered)
result = orchestrator.segment(
    input_data={
        'pet': 'path/to/pet.nii.gz',
        'ct': 'path/to/ct.nii.gz'
    },
    output_path='gtrc_segmentation.nii.gz'
)

# Access tumor subregions
tumor_core = result.labels['tumor_core']
tumor_edema = result.labels['tumor_edema']
tumor_enhancing = result.labels['tumor_enhancing']
```

### CLI Interface

```bash
# Segment single image
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc

# Batch processing with 4 workers
nnunetsegmentator batch -i input_dir/ -o output_dir/ -t deep_psma --num-workers 4

# List available tasks
nnunetsegmentator list-tasks

# Show task information
nnunetsegmentator info -t lion
```

### Batch Processing

```python
from pathlib import Path
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name='total')

# Process multiple patients
input_files = list(Path("data/patients").glob("*.nii.gz"))

results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=4,
    use_multiprocessing=True,
    progress_callback=lambda c, t, r: print(f"Progress: {c+1}/{t}")
)

# Check results
successful = sum(1 for r in results if r is not None)
print(f"Successfully processed: {successful}/{len(input_files)}")
```

## Architecture

```
nnunetsegmentator/
├── src/                        # Source code
│   └── nnunetsegmentator/      # Main package
│       ├── core/               # Core components
│       │   ├── orchestrator.py # Main segmentation orchestrator
│       │   ├── registry.py     # Model and task registry
│       │   └── config.py       # Configuration management
│       ├── io/                 # Input/Output handling
│       │   ├── readers.py      # NIfTI, DICOM readers
│       │   └── writers.py      # Output writers
│       ├── pipeline/           # Processing pipelines
│       │   ├── base.py         # Base pipeline classes
│       │   ├── steps/          # Pipeline steps
│       │   │   ├── preprocessing.py
│       │   │   ├── inference.py
│       │   │   └── postprocessing.py
│       │   └── builders.py     # Pipeline construction
│       ├── tasks/              # Task definitions
│       │   ├── gtrc.py         # GTRC-Net task
│       │   ├── lion.py         # LION task
│       │   └── deep_psma.py    # DEEP-PSMA task
│       ├── cli/                # Command-line interface
│       └── utils/              # Utility functions
├── docs/                       # Documentation
├── examples/                   # Example scripts
├── scripts/                    # Utility scripts
├── tests/                      # Test suite
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

## Available Tasks

### Core Tasks

#### GTRC-Net
Glioblastoma Treatment Response Classification using PET/CT.

**Input**: Co-registered PET and CT images  
**Output**: Tumor core, edema, and enhancing regions  
**Resolution**: 1.5mm isotropic  
**Models**: 5-fold ensemble

#### LION
Lesion Identification and Oncology Network for PET lesion segmentation.

**Input**: PET images (FDG or PSMA)  
**Output**: Liver, lung, bone lesions, and lymph nodes  
**Resolution**: 1.0mm isotropic  
**Models**: 5-fold ensemble

#### DEEP-PSMA
PSMA PET segmentation for prostate cancer assessment.

**Input**: PSMA PET images  
**Output**: PSMA-avid lesions, prostate, lymph nodes, bone lesions  
**Resolution**: 2.0mm isotropic  
**Models**: 5-fold ensemble

#### TotalSegmentator
Comprehensive whole-body CT segmentation.

**Input**: CT images  
**Output**: 117 anatomical structures  
**Resolution**: 1.5mm isotropic  
**Models**: Multi-stage ensemble

#### TotalSegmentator 2D
Fast 2D projection-based CT segmentation.

**Input**: CT images  
**Output**: 117 anatomical structures  
**Resolution**: Multi-scale  
**Speed**: < 1 second inference time

#### DukeSeg
Comprehensive anatomical structure segmentation.

**Input**: CT images  
**Output**: 140 structures  
**Resolution**: 1.5mm isotropic  
**Models**: Multi-stage ensemble

### Specialized Tasks

- **Lung Vessels**: Arteries, veins, airways, and airway walls
- **Lung Nodules**: Nodule detection and segmentation
- **Liver Vessels**: Vessels and tumors
- **Liver Lesions**: Lesion segmentation
- **Kidney Cysts**: Cyst segmentation
- **Heart Chambers**: 7 chamber classes
- **Cerebral Bleed**: Intracerebral hemorrhage
- **Tissue Types**: Fat and muscle segmentation
- **Body Segmentation**: Body regions
- **Vertebrae Body**: Individual vertebrae (C1-L5)
- **Pleural Effusion**: Effusion segmentation

For a complete list with details, see [docs/tasks/overview.md](docs/tasks/overview.md).

## Creating Custom Tasks

```python
from nnunetsegmentator.tasks.base import BaseTask
from nnunetsegmentator.core.registry import TaskDefinition, ModelInfo
from nnunetsegmentator.pipeline import Pipeline
from nnunetsegmentator.pipeline.steps.preprocessing import ResampleStep

class MyCustomTask(BaseTask):
    name = "my_task"
    description = "My custom segmentation task"

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        return TaskDefinition(
            name=cls.name,
            models={
                'my_model': ModelInfo(
                    name='my_model',
                    task_id='Dataset999_my_model',
                    url='https://url.to.model/model.zip',
                    checksum='abc123',
                    labels={'target': 1, 'background': 0},
                    modality='CT',
                    description=cls.description
                )
            },
            pipeline_config={
                'steps': [
                    {'type': 'resample', 'params': {'spacing': [1.0, 1.0, 1.0]}}
                ]
            },
            input_requirements={'modalities': ['CT'], 'format': 'NIfTI'},
            output_config={'format': 'NIfTI'}
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        pipeline = Pipeline(name="my_task_pipeline")
        pipeline.add_step(ResampleStep('resample', {'spacing': (1.0, 1.0, 1.0)}))
        # Add more steps...
        return pipeline

# Task classes under nnunetsegmentator.tasks are auto-discovered and auto-registered.
```

## Pipeline Steps

### Preprocessing Steps

| Step | Description | Use Case |
|------|-------------|----------|
| `ResampleStep` | Resample to target spacing | Standardize image resolution |
| `ClipIntensityStep` | Clip intensity values | Remove outliers |
| `NormalizeStep` | Normalize intensities | Standardize input distribution |
| `SUVThresholdStep` | Apply SUV threshold | PET quantification |
| `MultiChannelStackStep` | Stack multiple modalities | Multi-modal input |

### Inference Steps

| Step | Description | Use Case |
|------|-------------|----------|
| `nnUNetInferenceStep` | Run nnUNet inference | Standard segmentation |
| `CascadeInferenceStep` | Cascade multiple models | Multi-stage segmentation |
| `EnsembleInferenceStep` | Ensemble model predictions | Improve accuracy |

### Postprocessing Steps

| Step | Description | Use Case |
|------|-------------|----------|
| `LargestComponentStep` | Keep largest connected component | Remove false positives |
| `MorphologicalOpsStep` | Morphological operations | Smooth boundaries |
| `FillHolesStep` | Fill holes in segmentation | Close gaps |
| `RemoveSmallObjectsStep` | Remove small objects | Filter noise |
| `ExpandContractStep` | Expand/contract boundaries | Adjust segmentation |

## Configuration

Create a configuration file `config.yaml`:

```yaml
data_dir: ./data
output_dir: ./output
model_dir: ~/.nnunetsegmentator/models  # Default location
num_workers: 4
use_gpu: true
gpu_id: 0
batch_size: 1

nnunet_raw: /path/to/nnunet_workspace/raw
nnunet_preprocessed: /path/to/nnunet_workspace/preprocessed
nnunet_results: /path/to/nnunet_workspace/results

log_level: INFO
log_file: ./logs/segmentation.log

# Custom settings
custom:
  preprocessing:
    resample_spacing: [1.0, 1.0, 1.0]
  postprocessing:
    keep_top_n: 3
```

Load and use:

```python
from nnunetsegmentator import Config

config = Config.from_file('config.yaml')
config.setup_nnunet_environment()

orchestrator = SegmentationOrchestrator(task_name='total', config=config)
```

## nnUNet Environment Setup

The framework requires proper nnUNet environment setup:

```bash
# Set environment variables
export nnUNet_raw=/path/to/nnunet_workspace/raw
export nnUNet_preprocessed=/path/to/nnunet_workspace/preprocessed
export nnUNet_results=/path/to/nnunet_workspace/results
```

Or configure in Python:

```python
from nnunetsegmentator import Config

config = Config(
    nnunet_raw='/path/to/nnunet_workspace/raw',
    nnunet_preprocessed='/path/to/nnunet_workspace/preprocessed',
    nnunet_results='/path/to/nnunet_workspace/results'
)
config.setup_nnunet_environment()
```

## API Reference

### SegmentationOrchestrator

Main class for managing segmentation tasks.

```python
from nnunetsegmentator import SegmentationOrchestrator, Config

# Initialize
orchestrator = SegmentationOrchestrator(
    task_name='gtrc',          # Task name
    config=Config(),           # Configuration
    model_path=None,           # Optional model path override
    pipeline=None              # Optional custom pipeline
)

# Single subject segmentation
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
    num_workers=4,
    use_multiprocessing=True,
    progress_callback=None
)
```

### SegmentationResult

Container for segmentation results.

```python
result = orchestrator.segment("image.nii.gz", "output.nii.gz")

# Access segmentation
segmentation = result.segmentation  # nnunetsegmentator.image.Image
segmentation_array = result.get_array()  # Numpy array

# Access individual labels
for label_name, label_image in result.labels.items():
    label_array = result.get_label_array(label_name)

# Access metrics
if result.metrics:
    print(f"Volume: {result.metrics['volume_mm3']:.2f} mm³")

# Access metadata
print(f"Spacing: {result.metadata['spacing']}")
print(f"Origin: {result.metadata['origin']}")
```

### TaskRegistry

Central registry for tasks and models.

```python
from nnunetsegmentator import TaskRegistry

# List tasks
tasks = TaskRegistry.list_tasks()
for name, description in tasks.items():
    print(f"{name}: {description}")

# Get task
task = TaskRegistry.get_task('gtrc')
print(f"Task: {task.name}")
print(f"Models: {list(task.models.keys())}")

# Register model path
TaskRegistry.register_model_path('model_name', '/path/to/model')
```

## Examples

See the `examples/` directory for detailed usage examples:

| Example | Description | Command |
|---------|-------------|---------|
| `01_basic_segmentation.py` | Basic whole-body CT segmentation | `python examples/01_basic_segmentation.py` |
| `02_multimodal_segmentation.py` | Multi-modal PET/CT segmentation | `python examples/02_multimodal_segmentation.py` |
| `03_custom_pipeline.py` | Custom pipeline construction | `python examples/03_custom_pipeline.py` |
| `04_dicom_segmentation.py` | DICOM I/O and export | `python examples/04_dicom_segmentation.py` |

For more examples, see [examples/README.md](examples/README.md).

## Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [Installation & Usage](docs/getting-started/installation.md) | Installation guide and usage examples |
| [API Reference](docs/reference/api-reference.md) | Complete API documentation |
| [Tasks](docs/tasks/overview.md) | Available segmentation tasks |
| [Architecture](docs/architecture.md) | Framework architecture and design |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/nnunetsegmentator

# Run specific test file
pytest tests/unit/core/test_orchestrator.py

# Run with verbose output
pytest -v

# Type checking
mypy src/nnunetsegmentator

# Linting
flake8 src/nnunetsegmentator
```

## Contributing

Contributions are welcome! Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### How to Contribute

1. **Report bugs** - Open an issue with detailed reproduction steps
2. **Suggest features** - Open an issue with clear feature description
3. **Submit code** - Follow the pull request guidelines
4. **Improve docs** - Help us improve documentation
5. **Write tests** - Add tests for new functionality

### Development Setup

```bash
# Fork the repository
# Clone your fork
git clone https://github.com/YOUR_USERNAME/nnunetsegmentator.git
cd nnunetsegmentator

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{nnunetsegmentator2024,
  title={nnunetsegmentator: A unified framework for medical image segmentation},
  author={Your Name},
  year={2024},
  url={https://github.com/devhliu/nnunetsegmentator}
}
```

## Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/devhliu/nnunetsegmentator/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/devhliu/nnunetsegmentator/discussions)
- 💬 **Community**: Join our [Discord server](https://discord.gg/your-invite-link)

## Acknowledgments

This framework integrates models from the following institutions and research groups:

- Peter MacCallum Cancer Centre
- ENHANCE-PET Consortium
- Duke University CVIT
- Wasserthal et al. (TotalSegmentator)
- RISC-MI (TotalSegmentator 2D)

We thank all the original authors for making their models available to the research community.

---

**Happy Segmenting! 🎉**
