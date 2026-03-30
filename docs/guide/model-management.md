# Model Management Guide

## Model Locations

### Default Locations

Models are stored in: `~/.nnunetsegmentator/models`

Canonical layout:

- `task_id`: `Dataset<数字>_<model_name>`
- install key: same as `task_id`
- path: `~/.nnunetsegmentator/models/{task_name}/{task_id}`
- payload directly under `{task_id}` (no wrapper dir)

### Custom Model Directory

```python
from nnunetsegmentator import Config

config = Config(
    model_dir="/path/to/custom/models"
)
```

## Downloading Models

### Using Download Script

```bash
# List available models
python scripts/download_models.py --list

# Download specific task
python scripts/download_models.py --task gtrc --output-dir ~/.nnunetsegmentator/models

# Download all models
python scripts/download_models.py --task all --output-dir ~/.nnunetsegmentator/models
```

### Manual Download

1. Visit model source URLs
2. Download model files
3. Extract to model directory
4. Set nnUNet_results environment variable

## Installing Local Model Files

You can install local model files (zipped or unzipped) into the standard model storage path.

### CLI

```bash
nnunetsegmentator install-models \
  --task total \
  --model Dataset291_total_organs=/local/total_organs.zip \
  --model Dataset292_total_vertebrae=/local/total_vertebrae/
```

### Python API

```python
from nnunetsegmentator import TaskRegistry

installed = TaskRegistry.install_task_models_from_local(
    task_name="total",
    model_sources={
        "Dataset291_total_organs": "/local/total_organs.zip",
        "Dataset292_total_vertebrae": "/local/total_vertebrae/",
    },
    force=False,
)
print(installed)
```

### Script API

```bash
python scripts/model_manager.py \
  --install-local \
  --task total \
  --model Dataset291_total_organs=/local/total_organs.zip
```

## Migrating from nnUNet Workspace Results

If you already have trained/exported nnUNet models under
`~/.nnunetsegmentator/nnunet_workspace/results`, migrate them into canonical
storage with:

```bash
# Preview (no writes)
python scripts/migrate_workspace_models.py --dry-run

# Execute migration
python scripts/migrate_workspace_models.py
```

Or use model manager:

```bash
python scripts/model_manager.py --migrate-workspace --dry-run
python scripts/model_manager.py --migrate-workspace --force
```

Notes:
- Source default: `~/.nnunetsegmentator/nnunet_workspace/results`
- Target default: `~/.nnunetsegmentator/models/{task_name}/{task_id}`
- Migration auto-detects nnUNet payload roots and normalizes wrappers so payload
  is directly under `{task_id}`.
- `--force` overwrites existing canonical targets.

## Model Registration

### Registering Custom Models

```python
from nnunetsegmentator import TaskRegistry, Config

# Register custom model
TaskRegistry.register_task(
    task_name='my_model',
    model_path='/path/to/model',
    config=Config(use_gpu=True),
    labels={'structure': 1}
)
```

### Loading Models from File

```python
import yaml
from nnunetsegmentator import TaskRegistry

with open('model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

TaskRegistry.register_task(**config)
```

## Model Versioning

### Track Model Versions

```python
task_config = {
    'task_name': 'my_task',
    'model_path': '/path/to/model',
    'model_version': 'v1.0.0',
    'training_date': '2024-01-15',
    'labels': {'structure': 1}
}
```

### Model Metadata

```python
from nnunetsegmentator import TaskRegistry

task = TaskRegistry.get_task('gtrc')
print(f"Model version: {task.model_version}")
print(f"Training date: {task.training_date}")
```

## Model Performance

### Optimizing Model Loading

```python
from nnunetsegmentator import SegmentationOrchestrator

# Load once, use multiple times
orchestrator = SegmentationOrchestrator(task_name='gtrc')

# Process multiple images
for input_file in input_files:
    result = orchestrator.segment(input_file, output_file)
```

### GPU Memory Management

```python
config = Config(
    use_gpu=True,
    batch_size=1,  # Smaller batch for memory efficiency
    num_workers=2  # Reduce workers
)
```

## Best Practices

1. **Version control models** - Track model versions
2. **Use consistent naming** - For model files
3. **Monitor GPU memory** - Adjust batch size
4. **Cache models** - Load once, reuse multiple times
5. **Document model sources** - For reproducibility
