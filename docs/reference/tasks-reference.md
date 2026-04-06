# Tasks Reference

## Overview

This document provides detailed information about each supported segmentation task.

## Task Details

### GTRC-Net

**Task ID:** `gtrc`

**Modality:** PET/CT

**Description:** Glioblastoma treatment response segmentation

**Labels:**
- `tumor_core` (1)
- `tumor_edema` (2)
- `tumor_enhancing` (3)

**Default Resolution:** 1.5mm isotropic

**Input Requirements:**
- Co-registered PET and CT images
- NIfTI or DICOM format

**Pipeline Steps:**
1. Resample to 1.5mm isotropic
2. Clip CT intensities [-1000, 1000] HU
3. Apply SUV threshold 2.5 to PET
4. Stack PET and CT as multi-channel
5. nnUNet inference (5 folds)
6. Keep largest connected component
7. Morphological closing (radius=1)

### LION

**Task ID:** `lion`

**Modality:** PET

**Description:** Lesion identification and oncology network

**Labels:**
- `liver_lesion` (1)
- `lung_lesion` (2)
- `bone_lesion` (3)
- `lymph_node` (4)

**Default Resolution:** 1.0mm isotropic

**Input Requirements:**
- PET or CT image
- NIfTI or DICOM format

**Pipeline Steps:**
1. Resample to 1.0mm isotropic
2. Clip intensities [-1000, 500] HU
3. CT normalization
4. nnUNet inference (5 folds)
5. Morphological opening (radius=1)
6. Keep top 3 largest components

### DEEP-PSMA

**Task ID:** `deep_psma`

**Modality:** PET

**Description:** PSMA PET lesion segmentation

**Labels:**
- `psma_avid_lesion` (1)
- `prostate` (2)
- `lymph_node` (3)
- `bone_lesion` (4)

**Default Resolution:** 2.0mm isotropic

**Input Requirements:**
- PSMA PET image
- NIfTI or DICOM format

**Pipeline Steps:**
1. Resample to 2.0mm isotropic
2. Apply SUV threshold 3.0
3. Min-max normalization
4. nnUNet inference (5 folds)
5. Fill holes
6. Keep top 5 largest components
7. Morphological closing (radius=1)

### TotalSegmentator

**Task ID:** `total`

**Modality:** CT

**Description:** Comprehensive whole-body CT segmentation

**Labels:** 117 anatomical structures

**Default Resolution:** 1.5mm isotropic

**Input Requirements:**
- CT image
- NIfTI or DICOM format

**Pipeline Steps:**
1. Resample to 1.5mm isotropic
2. CT normalization
3. nnUNet inference (5 folds)
4. Postprocessing (varies by structure)

### DukeSeg

**Task ID:** `dukeseg`

**Modality:** CT

**Description:** Comprehensive anatomical structure segmentation

**Labels:** 140 anatomical structures

**Default Resolution:** 1.5mm isotropic

**Input Requirements:**
- CT image
- NIfTI or DICOM format

**Pipeline Steps:**
1. Resample to 1.5mm isotropic
2. CT normalization
3. nnUNet inference (5 folds)
4. Postprocessing (varies by structure)

### MRSegmentator

**Task ID:** `mrsegmentator`

**Modality:** MR, CT

**Description:** Multi-modality segmentation of 40 classes in MRI and CT

**Labels:** 40 anatomical structures
- Organs: spleen, kidneys, gallbladder, liver, stomach, pancreas, adrenal glands, lungs, heart
- Vessels: aorta, inferior vena cava, portal vein, iliac arteries/veins
- GI tract: esophagus, small bowel, duodenum, colon, urinary bladder
- Bones: spine, sacrum, hips, femurs
- Muscles: autochthonous, iliopsoas, gluteus (maximus, medius, minimus)

**Default Resolution:** 1.5mm isotropic

**Input Requirements:**
- MRI or CT image
- NIfTI, DICOM, MHA, or NRRD format
- Works on T1-weighted, T2-weighted, Dixon sequences, and CT

**Pipeline Steps:**
1. Resample to 1.5mm isotropic
2. Z-score normalization
3. nnUNet inference (5 folds)
4. Keep largest connected component for organs

**Citation:** Häntze et al., Radiology: Artificial Intelligence (2024). https://doi.org/10.1148/ryai.240777

## Task Selection

### Choose Based on Modality

- **PET/CT:** GTRC-Net
- **PET:** LION, DEEP-PSMA
- **CT:** TotalSegmentator, DukeSeg
- **MR:** MRSegmentator, TotalSegmentator MR
- **Multi-modality (MR/CT):** MRSegmentator

### Choose Based on Structures

- **Glioblastoma:** GTRC-Net
- **Lesions:** LION, DEEP-PSMA
- **Comprehensive anatomy (CT):** TotalSegmentator, DukeSeg
- **Comprehensive anatomy (MR):** MRSegmentator, TotalSegmentator MR
- **Abdominal/pelvic/thorax (MR):** MRSegmentator

## Custom Tasks

See [Custom Tasks Guide](../guide/custom-tasks.md) for creating your own tasks.
