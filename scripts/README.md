# nnunetsegmentator Scripts

This directory contains utility scripts for managing and using nnunetsegmentator.

## Available Scripts

### 1. download_models.py

Download pretrained model weights for nnunetsegmentator tasks.

**Usage**:
```bash
# List available models
python download_models.py --list

# Download specific model
python download_models.py --task gtrc --output-dir ./models

# Download with variant
python download_models.py --task lion --variant psma --output-dir ./models

# Download all models
python download_models.py --task all --output-dir ./models
```

**Model Sources**:
- GTRC-Net: Git LFS repository
- LION: GitHub releases
- DEEP-PSMA: Git LFS repository
- TotalSegmentator: Automatic download

**Requirements**:
- Git LFS (for GTRC-Net and DEEP-PSMA)
- Internet connection
- Sufficient disk space (~10GB for all models)

---

### 2. batch_process.py

Batch processing script for segmenting multiple images.

**Usage**:
```bash
# Process all images in a directory
python batch_process.py --input-dir /data/images --output-dir /data/output --task total

# Process from file list
python batch_process.py --input-list files.txt --output-dir /data/output --task lion

# Use multiple workers
python batch_process.py --input-dir /data/images --output-dir /data/output --task total --workers 4

# List available tasks
python batch_process.py --list-tasks
```

**Features**:
- Automatic file discovery
- Parallel processing
- Progress tracking
- Metadata saving
- Error handling

**Supported formats**:
- NIfTI (.nii, .nii.gz)
- NRRD (.nrrd, .mha, .mhd)
- DICOM (.dcm)

---

### 3. model_manager.py

Model management and status checking.

**Usage**:
```bash
# List all available models
python model_manager.py --list

# Check download status
python model_manager.py --status

# Download specific model
python model_manager.py --download --task gtrc

# Download with variant
python model_manager.py --download --task lion --variant psma

# Preview migration from workspace results into canonical model root
python model_manager.py --migrate-workspace --dry-run

# Execute migration and overwrite existing canonical targets
python model_manager.py --migrate-workspace --force

# Clean cache (dry run)
python model_manager.py --clean-cache

# Clean cache (force)
python model_manager.py --clean-cache --force
```

**Features**:
- Model status tracking
- Download management
- Cache cleaning
- Variant support
- Workspace-results migration to canonical model layout

---

### 4. migrate_workspace_models.py

Migrate model payloads from nnUNet workspace results into canonical model layout.

**Usage**:
```bash
# Dry run
python migrate_workspace_models.py --dry-run

# Migrate using defaults:
#   source: ~/.nnunetsegmentator/nnunet_workspace/results
#   target: ~/.nnunetsegmentator/models
python migrate_workspace_models.py

# Override source/target and force overwrite
python migrate_workspace_models.py \
  --workspace-results /path/to/nnunet_workspace/results \
  --model-root /path/to/models \
  --force
```

**Notes**:
- Canonical target is `{model_root}/{task_name}/{task_id}`.
- `task_id` uses `Dataset<数字>_<model_name>`.
- Payload is normalized to be directly under `{task_id}`.

---

## Quick Start

### 1. Install nnunetsegmentator

```bash
cd nnunetsegmentator
pip install -e .
```

### 2. Download Models

```bash
# Download TotalSegmentator (recommended for beginners)
python scripts/download_models.py --task total

# Or download all models
python scripts/download_models.py --task all
```

### 3. Process Images

```bash
# Single task
python scripts/batch_process.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --task total

# With multiple workers
python scripts/batch_process.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --task total \
    --workers 4
```

---

## Common Workflows

### Workflow 1: Whole-Body CT Segmentation

```bash
# Download TotalSegmentator
python scripts/download_models.py --task total

# Process CT images
python scripts/batch_process.py \
    --input-dir /data/ct_scans \
    --output-dir /data/segmentations \
    --task total \
    --workers 4
```

### Workflow 2: PET Lesion Segmentation

```bash
# Download LION models
python scripts/download_models.py --task lion

# Process FDG PET
python scripts/batch_process.py \
    --input-dir /data/fdg_pet \
    --output-dir /data/lesions \
    --task lion

# Process PSMA PET
python scripts/batch_process.py \
    --input-dir /data/psma_pet \
    --output-dir /data/lesions \
    --task lion \
    --variant psma
```

### Workflow 3: Multi-Modal PET/CT

```bash
# Download GTRC-Net
python scripts/download_models.py --task gtrc

# Process (requires Python API for multi-modal)
python -c "
from nnunetsegmentator import SegmentationOrchestrator
orchestrator = SegmentationOrchestrator(task_name='gtrc')
result = orchestrator.segment(
    input_data={'pet': 'pet.nii.gz', 'ct': 'ct.nii.gz'},
    output_path='segmentation.nii.gz'
)
"
```

---

## Advanced Usage

### Custom Configuration

Create a configuration file:

```python
# config.py
from nnunetsegmentator import Config

config = Config(
    model_dir="/custom/model/path",
    use_gpu=True,
    gpu_id=0,
    num_workers=4,
    log_level="DEBUG"
)
```

Use with scripts:

```python
from config import config
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name="total", config=config)
```

### Progress Callbacks

```python
def progress_callback(current, total, result):
    print(f"Processed {current}/{total}: {result}")

results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir=output_dir,
    progress_callback=progress_callback
)
```

### Error Handling

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='processing.log'
)

try:
    results = orchestrator.segment_batch(...)
except Exception as e:
    logging.error(f"Processing failed: {e}")
```

---

## Troubleshooting

### Issue: Git LFS not found

**Solution**: Install Git LFS

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install

# macOS
brew install git-lfs
git lfs install
```

### Issue: Out of memory

**Solution**: Reduce batch size or use CPU

```bash
# Use CPU
python batch_process.py ... --gpu -1

# Or reduce workers
python batch_process.py ... --workers 1
```

### Issue: Slow download

**Solution**: Use mirror or manual download

```bash
# Check status
python model_manager.py --status

# Manual download if needed
wget <model_url>
unzip model.zip -d ~/.nnunetsegmentator/models/
```

### Issue: Model not found

**Solution**: Verify model download

```bash
# Check model status
python model_manager.py --status

# Re-download if needed
python download_models.py --task <task> --output-dir ./models
```

---

## Performance Tips

### 1. Use GPU

```bash
python batch_process.py ... --gpu 0
```

### 2. Parallel Processing

```bash
python batch_process.py ... --workers 4
```

### 3. Use Fast Models

- `total_fast` - 3mm resolution (faster than `total`)
- `total_fastest` - 6mm resolution (fastest)
- `ts2d` - 2D projection (sub-second inference)

### 4. Batch Processing

Process multiple images together for better GPU utilization.

---

## Additional Resources

- [Examples](../examples/)
- [Documentation](../docs/)
- [API Reference](../docs/reference/api-reference.md)
