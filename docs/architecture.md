# Architecture

This document describes the architecture of the nnunetsegmentator framework.

## Overview

nnunetsegmentator is a unified framework for managing multiple nnUNet-based medical image segmentation tasks. It provides a composable pipeline architecture, task registry system, and both Python and CLI interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                     nnunetsegmentator                           │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface  │  Python API  │  Configuration                 │
├─────────────────────────────────────────────────────────────────┤
│                     SegmentationOrchestrator                    │
├─────────────────────────────────────────────────────────────────┤
│  TaskRegistry   │  Pipeline    │  I/O Handlers                  │
├─────────────────────────────────────────────────────────────────┤
│  Preprocessing  │  Inference   │  Postprocessing                │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. SegmentationOrchestrator

The main entry point for all segmentation operations. It coordinates:
- Task selection and model loading
- Pipeline execution
- Input/output handling
- Batch processing

```python
from nnunetsegmentator import SegmentationOrchestrator, Config

orchestrator = SegmentationOrchestrator(
    task_name="gtrc",
    config=Config()
)

result = orchestrator.segment("input.nii.gz", "output.nii.gz")
```

### 2. TaskRegistry

A singleton registry that manages all segmentation tasks and their associated models.

```python
from nnunetsegmentator import TaskRegistry

# List available tasks
tasks = TaskRegistry.list_tasks()

# Get specific task
task = TaskRegistry.get_task("gtrc")

# Register custom task
TaskRegistry.register(my_task_definition)
```

### 3. Pipeline System

A composable pipeline architecture for building processing workflows.

#### Pipeline Structure

```
Input → Preprocessing → Inference → Postprocessing → Output
```

#### Pipeline Components

| Component | Description |
|-----------|-------------|
| `PipelineContext` | Shared state passed through pipeline steps |
| `PipelineStep` | Abstract base for individual processing steps |
| `Pipeline` | Container for ordered steps with execution logic |

#### Pipeline Steps

**Preprocessing Steps:**
- `ResampleStep` - Resample image to target spacing
- `ClipIntensityStep` - Clip intensity values to range
- `NormalizeStep` - Normalize intensities (CT, PET, minmax)
- `SUVThresholdStep` - Apply SUV threshold for PET
- `MultiChannelStackStep` - Stack multi-modal images

**Inference Steps:**
- `nnUNetInferenceStep` - Standard nnUNet inference
- `CascadeInferenceStep` - Cascade/iterative inference

**Postprocessing Steps:**
- `LargestComponentStep` - Keep largest connected components
- `MorphologicalOpsStep` - Morphological operations (open/close)
- `FillHolesStep` - Fill holes in segmentation
- `RemoveSmallObjectsStep` - Remove small objects
- `ExpandContractStep` - Expand/contract boundaries

#### Pipeline Composition

Pipelines can be combined using the `|` operator:

```python
preprocessing = Pipeline([resample, normalize])
inference = Pipeline([nnunet_inference])
postprocessing = Pipeline([largest_component, smooth])

full_pipeline = preprocessing | inference | postprocessing
```

### 4. Configuration System

The `Config` class manages all framework settings:

```python
from nnunetsegmentator import Config

# Default configuration
config = Config()

# Load from file
config = Config.from_file("config.yaml")

# Customize
config.update(
    num_workers=4,
    use_gpu=True,
    output_dir="./results"
)
```

### 5. I/O System

Handles reading and writing of medical images:

**Readers:**
- `ImageReader` - Read NIfTI and DICOM images
- `MultiModalReader` - Read multi-modal image sets

**Writers:**
- `ImageWriter` - Write segmentations to NIfTI

## Module Structure

```
nnunetsegmentator/
├── core/
│   ├── orchestrator.py    # Main orchestrator
│   ├── registry.py        # Task and model registry
│   └── config.py          # Configuration management
├── pipeline/
│   ├── base.py            # Pipeline base classes
│   ├── builders.py        # Pipeline construction
│   └── steps/
│       ├── preprocessing.py
│       ├── inference.py
│       └── postprocessing.py
├── tasks/
│   ├── base.py            # Base task class
│   ├── gtrc.py            # GTRC-Net task
│   ├── lion.py            # LION task
│   └── deep_psma.py       # DEEP-PSMA task
├── io/
│   ├── readers.py         # Image readers
│   └── writers.py         # Image writers
├── utils/
│   ├── metrics.py         # Segmentation metrics
│   ├── resampling.py      # Resampling utilities
│   └── visualization.py   # Visualization tools
└── cli/
    └── main.py            # CLI entry point
```

## Data Flow

```
1. Input Loading
   └── ImageReader/MultiModalReader
       └── Returns: image.Image + metadata

2. Pipeline Execution
   └── PipelineContext flows through steps
       └── Each step transforms context

3. Result Extraction
   └── SegmentationResult
       ├── segmentation: image.Image
       ├── labels: Dict[str, image.Image]
       ├── metadata: Dict
       └── metrics: Dict

4. Output Writing
   └── ImageWriter
       └── Writes NIfTI files
```

## Task Definition

Each task is defined by a `TaskDefinition` containing:

```python
TaskDefinition(
    name="gtrc",
    models={
        'gtrc': ModelInfo(
            name='gtrc',
            task_id='Task501_GTRC',
            url='https://...',
            checksum='abc123',
            labels={'tumor_core': 1, 'tumor_edema': 2},
            modality='PETCT',
            description='...'
        )
    },
    pipeline_config={...},
    input_requirements={...},
    output_config={...}
)
```

## Extending the Framework

### Adding a New Task

1. Create a new task class inheriting from `BaseTask`:

```python
from nnunetsegmentator.tasks.base import BaseTask

class MyTask(BaseTask):
    name = "my_task"
    description = "My custom segmentation task"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        return TaskDefinition(...)
    
    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        return Pipeline([...])
```

2. Ensure your new task module is inside `src/nnunetsegmentator/tasks/`. It will be auto-discovered and registered:

```python
from nnunetsegmentator.tasks import register_all_tasks

register_all_tasks(refresh=True)
```

### Adding a Custom Pipeline Step

1. Create a step class inheriting from `PipelineStep`:

```python
from nnunetsegmentator.pipeline.base import PipelineStep, PipelineContext

class MyCustomStep(PipelineStep):
    def execute(self, context: PipelineContext) -> PipelineContext:
        # Process the image
        array = context.get_array()
        processed = self._process(array)
        context.set_array(processed)
        return context
```

2. Add to pipeline:

```python
pipeline.add_step(MyCustomStep('my_step', {'param': 'value'}))
```

## Design Principles

1. **Composability**: Pipelines are built from composable steps
2. **Extensibility**: Easy to add new tasks and processing steps
3. **Type Safety**: Full type hints for IDE support
4. **Separation of Concerns**: Clear module boundaries
5. **Configuration Flexibility**: YAML/JSON config support
