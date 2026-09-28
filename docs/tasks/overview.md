# Supported Tasks

## Overview

nnunetsegmentator supports multiple pre-configured segmentation tasks for medical imaging.

## Available Tasks

### GTRC-Net
- **Task ID:** `gtrc`
- **Modality:** PET/CT
- **Description:** Total tumor burden segmentation
- **Labels:** tumor_burden
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

### BOA Body Parts
- **Task ID:** `boa_body_parts`
- **Modality:** CT
- **Description:** Body parts segmentation — torso, head, arms, legs
- **Labels:** torso, head, leg_right, leg_left, arm_right, arm_left
- **Resolution:** 5.0mm slice thickness (in-plane spacing preserved)

### BOA Body Regions
- **Task ID:** `boa_body_regions`
- **Modality:** CT
- **Description:** Body regions segmentation — tissue compartments and cavities
- **Labels:** 11 body regions (subcutaneous tissue, muscle, thoracic cavity, mediastinum, pericardium, abdominal cavity, bone, glands, ...)
- **Resolution:** 5.0mm slice thickness (in-plane spacing preserved)

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

### Choose BOA if:
- You need body composition analysis (body parts or tissue regions)
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
