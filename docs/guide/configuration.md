# Configuration Guide

## Config Basics

### Creating Configuration

```python
from nnunetsegmentator import Config

# Default configuration
config = Config()

# Custom configuration
config = Config(
    use_gpu=True,
    num_workers=4,
    batch_size=4,
    log_level="INFO"
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_gpu` | bool | True | Use GPU for inference |
| `num_workers` | int | 1 | Number of parallel workers |
| `batch_size` | int | 1 | Batch size for inference |
| `log_level` | str | "INFO" | Logging level |
| `data_dir` | Path | "./data" | Default data directory |
| `output_dir` | Path | "./output" | Default output directory |
| `model_dir` | Path | ~/.nnunetsegmentator/models | Model directory |
| `nnunet_results` | Path | None | nnUNet results directory |

## Loading Configuration

### From YAML File

```yaml
# config.yaml
use_gpu: true
num_workers: 4
batch_size: 4
log_level: "INFO"
data_dir: "./data"
output_dir: "./output"
model_dir: "~/.nnunetsegmentator/models"
nnunet_results: "/path/to/nnunet_results"
```

```python
from nnunetsegmentator import Config

config = Config.from_file("config.yaml")
```

### From Dictionary

```python
config_dict = {
    'use_gpu': True,
    'num_workers': 4,
    'batch_size': 4
}
config = Config.from_dict(config_dict)
```

## Using Configuration

### With Orchestrator

```python
from nnunetsegmentator import SegmentationOrchestrator, Config

config = Config(use_gpu=True, num_workers=4)
orchestrator = SegmentationOrchestrator(
    task_name="gtrc",
    config=config
)
```

### Updating Configuration

```python
config = config.update(
    use_gpu=False,
    num_workers=2
)
```

## Environment Variables

```python
config = Config(
    nnunet_raw="/path/to/nnunet_workspace/raw",
    nnunet_preprocessed="/path/to/nnunet_workspace/preprocessed",
    nnunet_results="/path/to/nnunet_workspace/results"
)
config.setup_nnunet_environment()
```

## Best Practices

1. **Use config files** for reproducibility
2. **Set nnUNet paths** explicitly
3. **Adjust num_workers** based on CPU cores
4. **Monitor GPU memory** with batch_size
5. **Set appropriate log_level** for debugging
