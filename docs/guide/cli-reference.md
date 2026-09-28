# CLI Reference

## Basic Commands

### Segment Single Image

```bash
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc
```

**Options:**
- `-i, --input` - Input image path
- `-o, --output` - Output segmentation path
- `-t, --task` - Task name (gtrc, lion, deep_psma, total, dukeseg)
- `-m, --model` - Model path (optional)
- `-c, --config` - Configuration file path

### Batch Processing

```bash
nnunetsegmentator batch -i input_dir/ -o output_dir/ -t lion --num-workers 4
```

**Options:**
- `-i, --input` - Input directory or file list
- `-o, --output` - Output directory
- `-t, --task` - Task name
- `--num-workers` - Number of parallel workers (default: 1)
- `--multiprocessing` - Use multiprocessing

### List Tasks

```bash
nnunetsegmentator list-tasks
```

### Show Task Information

```bash
nnunetsegmentator info -t gtrc
```

## Complete Command Reference

### segment

Segment a single image.

```bash
nnunetsegmentator segment [OPTIONS]
```

**Arguments:**
- `--input`, `-i` - Input image path (required)
- `--output`, `-o` - Output segmentation path (required)
- `--task`, `-t` - Task name (required)

**Options:**
- `--model` - Model path
- `--config` - Configuration file
- `--no-labels` - Do not save individual labels
- `--compute-metrics` - Compute metrics

### batch

Process multiple images.

```bash
nnunetsegmentator batch [OPTIONS]
```

**Arguments:**
- `--input`, `-i` - Input directory (required)
- `--output`, `-o` - Output directory (required)
- `--task`, `-t` - Task name (required)

**Options:**
- `--num-workers` - Number of workers
- `--multiprocessing` - Use multiprocessing
- `--config` - Configuration file

### list-tasks

List all available tasks.

```bash
nnunetsegmentator list-tasks
```

### info

Show task details.

```bash
nnunetsegmentator info [OPTIONS]
```

**Options:**
- `--task`, `-t` - Task name (required)

### install-models

Install task model files from local archives/directories, either by explicit
canonical key or by auto-discovery under a root directory.

```bash
# Explicit mapping (repeatable)
nnunetsegmentator install-models --task TASK --model DATASETID_MODELNAME=PATH [--model ...] [--force]

# Auto-discover under a root directory
nnunetsegmentator install-models --from ROOT [--task TASK] [--dry-run] [--force]
```

**Options:**
- `--model DATASETID_MODELNAME=PATH` - Explicit mapping (repeatable).
- `--from ROOT` - Recursively search ROOT for model payloads/archives and install them.
- `--dry-run` - With `--from`, only list what would be installed (no writes).
- `--force` - Overwrite existing installed model directories.

**Notes:**
- `DATASETID_MODELNAME` must be canonical task_id `Dataset<数字>_<model_name>`.
- `PATH` can point to a directory (unzipped model) or archive (`.zip`, `.tar`, `.tar.gz`, `.tgz`).
- With `--from`, candidates are payload directories (`dataset.json`/`plans.json` present)
  and weight archives; each is matched to a registered model by a canonical `task_id`
  or `Dataset<数字>` prefix found in the candidate or an ancestor directory name.
  Ambiguous candidates (a shared prefix) are skipped — narrow with `-t/--task`.
- Installed output path follows the standard structure:
  `{NNUNETSEGMENTATOR_MODEL_ROOTPATH}/{task_name}/{task_id}`.

## Examples

### Basic Segmentation

```bash
nnunetsegmentator segment -i patient_ct.nii.gz -o output.nii.gz -t lion
```

### Batch Processing with Progress

```bash
nnunetsegmentator batch -i data/patients/ -o results/ -t total --num-workers 8
```

### Using Custom Model

```bash
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc -m /path/to/model
```

### Install Local Models (canonical key)

```bash
nnunetsegmentator install-models \
  --task total \
  --model Dataset291_total_organs=/local/total_organs.zip \
  --model Dataset292_total_vertebrae=/local/total_vertebrae/
```

### Install Local Models (auto-discovery)

```bash
# Preview matches under a weights root without writing anything
nnunetsegmentator install-models --from /data/weights --dry-run

# Install everything that matches a registered model, overwriting existing ones
nnunetsegmentator install-models --from /data/weights --force
```

### With Configuration File

```bash
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc -c config.yaml
```

## Environment Variables

```bash
export nnUNet_results="/path/to/nnunet_workspace/results"
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc
```
