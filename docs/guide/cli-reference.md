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

Install task model files from local archives or directories.

```bash
nnunetsegmentator install-models --task TASK --model ID_OR_NAME=PATH [--model ID_OR_NAME=PATH ...] [--force]
```

**Notes:**
- `ID_OR_NAME` can be either the task model name or the nnUNet task ID directory name.
- `PATH` can point to a directory (unzipped model) or archive (`.zip`, `.tar`, `.tar.gz`, `.tgz`).
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

### Install Local Models (by model name)

```bash
nnunetsegmentator install-models \
  --task total \
  --model total_organs=/local/total_organs.zip \
  --model total_vertebrae=/local/total_vertebrae/
```

### Install Local Models (by task ID)

```bash
nnunetsegmentator install-models \
  --task total \
  --model Dataset291=/local/total_organs.zip
```

### With Configuration File

```bash
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc -c config.yaml
```

## Environment Variables

```bash
export nnUNet_results="/path/to/models"
nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc
```
