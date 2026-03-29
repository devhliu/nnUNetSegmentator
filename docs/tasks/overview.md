# Supported Tasks

## Overview

nnunetsegmentator supports multiple pre-configured segmentation tasks for medical imaging.

## Available Tasks

### GTRC-Net
- **Task ID:** `gtrc`
- **Modality:** PET/CT
- **Description:** Glioblastoma treatment response segmentation
- **Labels:** tumor_core, tumor_edema, tumor_enhancing
- **Resolution:** 1.5mm isotropic

### LION
- **Task ID:** `lion`
- **Modality:** PET
- **Description:** Lesion identification and oncology network
- **Labels:** liver_lesion, lung_lesion, bone_lesion, lymph_node
- **Resolution:** 1.0mm isotropic

### DEEP-PSMA
- **Task ID:** `deep_psma`
- **Modality:** PET
- **Description:** PSMA PET lesion segmentation
- **Labels:** psma_avid_lesion, prostate, lymph_node, bone_lesion
- **Resolution:** 2.0mm isotropic

### TotalSegmentator
- **Task ID:** `total`
- **Modality:** CT
- **Description:** Comprehensive whole-body CT segmentation
- **Labels:** 117 anatomical structures
- **Resolution:** 1.5mm isotropic

### DukeSeg
- **Task ID:** `dukeseg`
- **Modality:** CT
- **Description:** Comprehensive anatomical structure segmentation
- **Labels:** 140 anatomical structures
- **Resolution:** 1.5mm isotropic

## Task Selection Guide

### Choose GTRC-Net if:
- You need glioblastoma segmentation
- You have co-registered PET/CT images

### Choose LION if:
- You need lesion identification
- You have PET or CT images with metastases

### Choose DEEP-PSMA if:
- You need PSMA PET segmentation
- You have prostate cancer imaging

### Choose TotalSegmentator if:
- You need comprehensive anatomical segmentation
- You have CT images

### Choose DukeSeg if:
- You need 140 anatomical structures
- You have CT images

## Custom Tasks

See [Custom Tasks Guide](../guide/custom-tasks.md) for creating your own tasks.

## Task Registry

```python
from nnunetsegmentator import TaskRegistry

# List all tasks
tasks = TaskRegistry.list_tasks()

# Get task details
task = TaskRegistry.get_task("gtrc")
```
