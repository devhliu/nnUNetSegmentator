# Configuration Reference

## Configuration Options

### Basic Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_gpu` | bool | True | Use GPU for inference |
| `num_workers` | int | 1 | Number of parallel workers |
| `batch_size` | int | 1 | Batch size for inference |
| `log_level` | str | "INFO" | Logging level |

### Path Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `data_dir` | Path | "./data" | Default data directory |
| `output_dir` | Path | "./output" | Default output directory |
| `model_dir` | Path | ~/.nnunetsegmentator/models | Model weights directory |
| `cache_dir` | Path | ~/.nnunetsegmentator/cache | Cache directory |

### nnUNet Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `nnunet_raw` | Path | None | nnUNet raw data directory |
| `nnunet_preprocessed` | Path | None | nnUNet preprocessed data directory |
| `nnunet_results` | Path | None | nnUNet results directory |

### Advanced Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `patch_size` | list | None | Patch size for inference |
| `min_size` | int | 100 | Minimum object size |
| `confidence_threshold` | float | 0.5 | Confidence threshold |
| `mixed_precision` | bool | False | Use mixed precision |

## Configuration File

### YAML Format

```yaml
# config.yaml
use_gpu: true
num_workers: 4
batch_size: 4
log_level: "INFO"

data_dir: "./data"
output_dir: "./output"
model_dir: "~/.nnunetsegmentator/models"
cache_dir: "~/.nnunetsegmentator/cache"

nnunet_raw: "/path/to/nnunet_workspace/raw"
nnunet_preprocessed: "/path/to/nnunet_workspace/preprocessed"
nnunet_results: "/path/to/nnunet_workspace/results"

patch_size: [128, 128, 64]
min_size: 100
confidence_threshold: 0.5
mixed_precision: true
```

### Loading Configuration

```python
from nnunetsegmentator import Config

# Load from file
config = Config.from_file("config.yaml")

# Update configuration
config = config.update(use_gpu=False, num_workers=2)
```

## Environment Variables

```bash
export nnUNet_raw="/path/to/nnunet_workspace/raw"
export nnUNet_preprocessed="/path/to/nnunet_workspace/preprocessed"
export nnUNet_results="/path/to/nnunet_workspace/results"
```

## Best Practices

1. **Use config files** for reproducibility
2. **Set nnUNet paths** explicitly
3. **Adjust num_workers** based on CPU cores
4. **Monitor GPU memory** with batch_size
5. **Set appropriate log_level** for debugging
