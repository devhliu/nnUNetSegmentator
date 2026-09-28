# How to Download Models and Use Them

This guide explains how to configure the model storage location, download
model weights (automatically, via script, or from local files), and run
segmentation with them.

## 1. Model Storage Location

### Default Location

Models are stored under `~/.nnunetsegmentator/models` by default. Override
with an environment variable:

```bash
export NNUNETSEGMENTATOR_MODEL_ROOTPATH=/data/models
```

### Canonical Layout

```
{model_root}/{task_name}/{task_id}/
├── dataset.json
├── plans.json
├── dataset_fingerprint.json
├── fold_0/                          # or fold_all/ (e.g. LION)
│   ├── checkpoint_final.pth
│   ├── checkpoint_best.pth
│   └── progress.png
└── nnUNetTrainer...__nnUNetPlans__3d_fullres/   # optional nested copy
    ├── dataset.json
    ├── plans.json
    └── fold_0/...
```

- `task_name`: registered task name (e.g. `total`, `moose`, `gtrc`, `lion`)
- `task_id`: canonical key `Dataset<数字>_<model_name>` (e.g.
  `Dataset291_total_organs`), and the nnUNet payload sits **directly**
  under it (no wrapper directory):
  `dataset.json` + `plans.json` + fold directories holding the checkpoints
- The `nnUNetTrainer...` subfolder, when present in an upstream archive, is
  a mirrored training-output directory; normalization keeps the payload at
  `{task_id}` level
- This `{task_id}` path is exactly what `nnUNetPredictor.
  initialize_from_trained_model_folder()` receives at inference time

### Separate nnUNet Workspace

The nnUNet scratch workspace (`nnUNet_raw`, `nnUNet_preprocessed`,
`nnUNet_results`) must live outside the model root:

```bash
export NNUNETSEGMENTATOR_NNUNET_WORKSPACE=/data/nnunet_workspace
```

The configuration validates this and raises an error if the workspace is
inside the model directory.

## 2. Downloading Models

Inference never downloads. Model resolution (`TaskRegistry.get_model_path`)
only checks whether the model payload is present at the canonical location
`{model_root}/{task_name}/{task_id}` and raises `ModelNotFoundError` (with an
install hint) when it is missing. Downloading is always an explicit step —
use one of the options below.

### Option A: Download from Registered URLs

```bash
# All models of a task
nnunetsegmentator download-models -t total

# A subset of models
nnunetsegmentator download-models -t moose --models clin_ct_organs,clin_ct_ribs

# Re-download over an existing installation
nnunetsegmentator download-models -t total --force
```

Python API:

```python
from nnunetsegmentator import TaskRegistry

TaskRegistry.download_model("clin_ct_organs")                    # single model
TaskRegistry.download_task_models("moose", ["clin_ct_organs"])   # subset
TaskRegistry.download_task_models("moose")                       # all models
```

Downloads are checksum-verified, extracted, normalized (a single wrapper
directory is flattened so the payload sits directly under `{task_id}`) and
recorded in `model_index.json`. Models whose URL is a placeholder or a git
repository cannot be downloaded this way — use `install-models` with local
files instead.

### Option B: Download Script

`scripts/download_models.py` downloads models from their upstream sources
(GitHub releases, Git LFS, Zenodo) and organizes them into the canonical
layout:

```bash
# List all available models and variants
python scripts/download_models.py --list

# Download a single task
python scripts/download_models.py --task lion --output-dir ~/.nnunetsegmentator/models

# Download a specific variant
python scripts/download_models.py --task lion --variant psma --output-dir ~/.nnunetsegmentator/models

# Download everything
python scripts/download_models.py --task all --output-dir ~/.nnunetsegmentator/models
```

Notes:

- `gtrc` and `deep_psma` are fetched via **Git LFS** (install `git-lfs`
  first).
- `total` / `total_mr` / `mrsegmentator` use the **PIP** method: the upstream
  package is installed, its nnUNet payload is auto-discovered
  (`~/.totalsegmentator/nnunet/results`, `nnUNet_results`, ...) and copied
  into canonical storage.

### Option C: Install Local Model Files (Offline)

For weights downloaded manually (zip/tar archives or already-unzipped
directories), use `install-models`:

```bash
# Keys must use the canonical format Dataset<数字>_<model_name>
nnunetsegmentator install-models \
  --task total \
  --model Dataset291_total_organs=/local/total_organs.zip \
  --model Dataset292_total_vertebrae=/local/total_vertebrae/ \
  --force   # overwrite existing installations
```

Python API equivalent:

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

Archives are extracted with path-traversal protection; a single top-level
wrapper directory is flattened automatically and the payload is normalized
so it sits directly under `{task_id}`.

#### Auto-Discover from a Weights Root

When weights are scattered under an arbitrary root (a download folder, an
extracted `nnUNet_results` tree, a set of `.zip` files, ...), `--from`
searches the root recursively and installs every candidate it can match to a
registered model:

```bash
# Preview what would be installed, without writing anything
nnunetsegmentator install-models --from /data/weights --dry-run

# Install everything that matches a registered model
nnunetsegmentator install-models --from /data/weights

# Restrict discovery to one task and overwrite existing installations
nnunetsegmentator install-models --from /data/weights -t total --force
```

Python API equivalent:

```python
from nnunetsegmentator import TaskRegistry

result = TaskRegistry.install_models_from_root(
    root="/data/weights",
    task_name=None,   # optional filter
    force=False,
    dry_run=False,
)
print(result["installed"])   # model_name -> installed path
print(result["unmatched"])   # candidates that matched nothing
```

Discovery and matching rules:

- **Candidates** are payload directories — a directory containing
  `dataset.json` or `plans.json`, or a `nnUNetTrainer...` subdirectory — as
  well as `.zip` / `.tar` / `.tar.gz` / `.tgz` weight archives.
- Each candidate is matched **deterministically** to a registered model by
  its canonical `task_id` (`Dataset<数字>_<model_name>`) or `Dataset<数字>`
  prefix, using the candidate name and its ancestor directory names (so a
  nested `nnUNet_results/<task_id>/nnUNetTrainer.../` layout matches).
- A `Dataset` prefix shared by several models (e.g. `Dataset881` for both
  `deep_psma` and `gtrc`) is **ambiguous** and is skipped; narrow it with
  `-t/--task`.
- Candidates that match no model are reported as `unmatched` and skipped —
  loose single files that cannot form a full payload are never installed.
- Installation reuses the same extract/normalize/validate path as Option C;
  an existing installation is kept unless `--force` is given.

### Manage Downloaded Models

```python
from nnunetsegmentator.utils.model_download import ModelDownloader

d = ModelDownloader()
d.list_models()                     # name, path, size, exists
d.remove_model("clin_ct_organs")    # delete one model
```

## 3. Using Models

Models must already be installed (see section 2) — segmentation fails with
`ModelNotFoundError` when a required model is missing, rather than downloading
it on the fly.

### CLI

```bash
# Explicit task
nnunetsegmentator segment -i ct.nii.gz -o output.nii.gz -t moose

# Restrict to a subset of the task's models
nnunetsegmentator segment -i ct.nii.gz -o output.nii.gz \
  -t moose --models clin_ct_organs,clin_ct_ribs

# Automatic model selection from target organs (mutually exclusive with -t)
nnunetsegmentator segment -i ct.nii.gz -o output.nii.gz \
  --organs "liver,left kidney" --modality CT

# Batch processing
nnunetsegmentator batch -i input_dir/ -o output_dir/ -t total --num-workers 4
```

Discovery helpers:

```bash
nnunetsegmentator list-tasks                     # registered tasks
nnunetsegmentator list-organs -t moose           # canonical organs per task
nnunetsegmentator find-models --organs "liver,SCT:10200004" --modality CT
```

### Python API

```python
from nnunetsegmentator import SegmentationOrchestrator, Config

# Reuse one orchestrator for multiple images (model is loaded once)
orchestrator = SegmentationOrchestrator(
    task_name="moose",
    config=Config(use_gpu=True),
    models=["clin_ct_organs"],      # optional model subset
)

result = orchestrator.segment(
    "ct.nii.gz",
    "output.nii.gz",
    return_labels=True,             # also emit per-organ binary masks
    compute_metrics=True,
)
```

Per-model tasks (`output_config.strategy == "per_model"`, e.g. `moose`) run
each selected model independently and write one segmentation plus an
organ-mapping sidecar per model:

```
output/
├── output.nii.gz                    # primary model segmentation
├── output__clin_ct_organs_seg.nii.gz
├── output_labels/                   # per-organ masks (model/label keyed)
└── output_organ_mapping.json        # label id -> name / SNOMED / laterality / RGB
```

DICOM series input is handled automatically: the series is converted to
NIfTI (`<stem>_image.nii.gz`) and the result is exported as a DICOM-SEG
object (`<case_id>.dcm`). Disable either step with `--no-convert-nifti` /
`--no-dicom-seg`.

### Merged labels

Label names come from the central SNOMED mapping
(`src/nnunetsegmentator/mapping/data/moose_snomed_mapping.csv`). Rows sharing
the same `merge_group` value are combined into that single label by the
`merge_labels` step appended to every task pipeline, so both the per-label
NIfTI masks and the DICOM-SEG object carry the merged label. The left lung
lobes (`lung_upper_lobe_left`, `lung_lower_lobe_left`) are therefore stored as
one `Left Lung` label and the right lobes as one `Right Lung` label; both are
coded as SNOMED `Lung` (`SCT:39607008`) plus a `Left`/`Right` modifier.

Each merged group keeps the lowest member label id in the multilabel
segmentation. Groups with only one member, and labels without a `merge_group`
value, are left untouched.

### GPU Memory

```python
config = Config(
    use_gpu=True,
    batch_size=1,   # smaller batch for memory efficiency
    num_workers=2,
)
```

## Best Practices

1. **Point `NNUNETSEGMENTATOR_MODEL_ROOTPATH` at fast, persistent storage**
   and keep it separate from the nnUNet workspace.
2. **Install models explicitly before segmenting** — use
   `download-models` when the upstream URL works, or `install-models` with
   local files on offline machines. Inference never downloads.
3. **Keep canonical keys** (`Dataset<数字>_<model_name>`) for install and
   selection; other identifiers are rejected.
4. **Load once, segment many times** — reuse the orchestrator instead of
   re-creating it per image.
5. **Document model sources** for reproducibility.
