# nnunetsegmentator Examples

This directory contains example scripts demonstrating how to use nnunetsegmentator for various segmentation tasks.

## Table of Contents

1. [Basic Segmentation](#01_basic_segmentationpy)
2. [Multi-Modal Segmentation](#02_multimodal_segmentationpy)
3. [Custom Pipeline](#03_custom_pipelinepy)
4. [DICOM Input/Output Segmentation](#04_dicom_segmentationpy)
5. [Install `total_mr` from TotalSegmentator](#05_install_total_mr_from_totalsegmentatorpy)

---

## Prerequisites

Before running the examples, ensure you have:

1. **Installed nnunetsegmentator**:
   ```bash
   cd nnunetsegmentator
   pip install -e .
   ```

2. **Downloaded required models** (optional - models download automatically on first use):
   ```bash
   python scripts/download_models.py --task total
   python scripts/download_models.py --task gtrc
   python scripts/download_models.py --task lion
   ```

3. **Installed dependencies**:
   ```bash
   pip install nibabel numpy
   ```

---

## 01_basic_segmentation.py

**Description**: Demonstrates basic whole-body CT segmentation using TotalSegmentator.

**What it does**:
- Creates a synthetic CT image with soft tissue and bone
- Initializes TotalSegmentator (117 anatomical structures)
- Performs segmentation
- Saves individual label masks

**Run**:
```bash
python examples/01_basic_segmentation.py
```

**Output**:
- `example_output/synthetic_ct.nii.gz` - Input CT image
- `example_output/segmentation.nii.gz` - Multi-label segmentation
- `example_output/labels/*.nii.gz` - Individual label masks

**Key concepts**:
- Single-modality segmentation
- Automatic model download
- Label extraction and analysis

---

## 02_multimodal_segmentation.py

**Description**: Demonstrates multi-modal PET/CT segmentation using GTRC-Net.

**What it does**:
- Creates synthetic co-registered PET and CT images
- Simulates tumor with high PET uptake
- Performs multi-modal segmentation with GTRC-Net
- Analyzes tumor subregions

**Run**:
```bash
python examples/02_multimodal_segmentation.py
```

**Requirements**:
- GTRC-Net models (requires Git LFS)
- Download: `python scripts/download_models.py --task gtrc`

**Output**:
- `example_output_multimodal/synthetic_pet.nii.gz` - PET image
- `example_output_multimodal/synthetic_ct.nii.gz` - CT image
- `example_output_multimodal/gtrc_segmentation.nii.gz` - Tumor segmentation

**Key concepts**:
- Multi-modal input handling
- PET/CT co-registration
- Tumor subregion segmentation

---

## 03_custom_pipeline.py

**Description**: Demonstrates how to create custom processing pipelines.

**What it does**:
- Builds custom preprocessing pipeline
- Builds custom postprocessing pipeline
- Combines pipelines using the `|` operator
- Shows how to use custom pipeline with orchestrator

**Run**:
```bash
python examples/03_custom_pipeline.py
```

**Key concepts**:
- Pipeline construction
- Step composition
- Pipeline combination
- Custom preprocessing/postprocessing

---

## 04_dicom_segmentation.py

**Description**: Demonstrates DICOM input and output workflows with DICOM SEG and DICOM RTSTRUCT export.

**What it does**:
- Creates synthetic DICOM series (CT)
- Performs segmentation from DICOM input
- Exports results as DICOM SEG (modern standard)
- Exports results as DICOM RTSTRUCT (radiotherapy format)

**Run**:
```bash
python examples/04_dicom_segmentation.py
```

**Requirements**:
- pydicom >= 2.3.0
- highdicom (for DICOM SEG)
- dicom2nifti

**Install requirements**:
```bash
pip install pydicom>=2.3.0 highdicom dicom2nifti
```

**Output**:
- `example_output_dicom/dicom_input/` - Synthetic DICOM series
- `example_output_dicom/segmentation.nii.gz` - NIfTI segmentation
- `example_output_dicom/segmentation_seg.dcm` - DICOM SEG file
- `example_output_dicom/segmentation_rtstruct.dcm` - DICOM RTSTRUCT file

**Key concepts**:
- DICOM series input
- DICOM SEG export (modern standard)
- DICOM RTSTRUCT export (radiotherapy)
- Clinical workflow integration

**DICOM Format Comparison**:

| Format | Use Case | Advantages |
|--------|----------|------------|
| **DICOM SEG** | Modern applications | Rich metadata, overlapping segments, voxel-based |
| **DICOM RTSTRUCT** | Radiotherapy planning | Wide support, contour-based, TPS compatible |

---

## 05_install_total_mr_from_totalsegmentator.py

**Description**: Finds `total_mr` model folders under your local TotalSegmentator cache and installs them into nnunetsegmentator default model root.

**What it does**:
- Searches for TotalSegmentator results under:
  - `~/.totalsegmetnator/nnunet/results` (requested path spelling)
  - `~/.totalsegmentator/nnunet/results` (common path)
- Detects `Dataset850*` / `Dataset852*` model folders
- Installs them into:
  - `~/.nnunetsegmentator/models/total_mr/850`
  - `~/.nnunetsegmentator/models/total_mr/852`

**Run**:
```bash
python examples/05_install_total_mr_from_totalsegmentator.py
```

**Overwrite existing install**:
```bash
python examples/05_install_total_mr_from_totalsegmentator.py --force
```

**Preview without copying**:
```bash
python examples/05_install_total_mr_from_totalsegmentator.py --dry-run
```

**Key concepts**:
- Local model installation by task ID
- Interoperability with existing TotalSegmentator model caches
- Standardized nnunetsegmentator model root layout

---

## Simulation Data

The examples use synthetic/simulation data for demonstration purposes:

### Synthetic CT
- Background: -1000 HU (air)
- Soft tissue: 40 HU
- Bone: 300 HU
- Shape: 128×128×128 voxels
- Spacing: 2.0mm isotropic

### Synthetic PET
- Background: 0 SUV
- Normal tissue: 1.0 SUV
- Tumor: 5.0 SUV (high uptake)
- Same geometry as CT

### Why Simulation Data?
- No patient data required
- Reproducible results
- Easy to understand
- Quick to generate
- Suitable for testing

---

## Common Patterns

### Pattern 1: Basic Usage
```python
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name="total")
result = orchestrator.segment(
    input_data="patient_ct.nii.gz",
    output_path="segmentation.nii.gz"
)
```

### Pattern 2: Multi-Modal Input
```python
result = orchestrator.segment(
    input_data={
        'pet': "patient_pet.nii.gz",
        'ct': "patient_ct.nii.gz"
    },
    output_path="segmentation.nii.gz"
)
```

### Pattern 3: Batch Processing
```python
results = orchestrator.segment_batch(
    input_list=["patient1.nii.gz", "patient2.nii.gz"],
    output_dir="output/",
    num_workers=4
)
```

### Pattern 4: Custom Pipeline
```python
from nnunetsegmentator.pipeline import Pipeline
from nnunetsegmentator.pipeline.steps.preprocessing import ResampleStep

custom_pipeline = Pipeline()
custom_pipeline.add_step(ResampleStep('resample', {'spacing': (1.0, 1.0, 1.0)}))

orchestrator = SegmentationOrchestrator(
    task_name="total",
    pipeline=custom_pipeline
)
```

---

## Troubleshooting

### Issue: Models not found
**Solution**: Download models first
```bash
python scripts/download_models.py --task <task_name>
```

### Issue: Git LFS not installed
**Solution**: Install Git LFS
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# Download from https://git-lfs.github.com/
```

### Issue: Out of memory
**Solution**: Use lower resolution or batch size
```python
config = Config(
    use_gpu=True,
    batch_size=1  # Reduce batch size
)
```

### Issue: Slow inference
**Solution**: Use faster model variants
```bash
# Use TotalSegmentator fast model
orchestrator = SegmentationOrchestrator(task_name="total_fast")

# Or use 2D projection model
orchestrator = SegmentationOrchestrator(task_name="ts2d")
```

---

## Next Steps

After running these examples, you can:

1. **Try different tasks**:
   - `lion` - PET lesion segmentation
   - `deep_psma` - PSMA PET segmentation
   - `dukeseg` - Comprehensive 140-structure segmentation

2. **Process real data**:
   - Replace synthetic data with patient images
   - Ensure proper format (NIfTI or DICOM)
   - Check image orientation and spacing

3. **Customize pipelines**:
   - Modify preprocessing steps
   - Add custom postprocessing
   - Combine multiple models

4. **Batch processing**:
   - Use `segment_batch()` for multiple patients
   - Enable multiprocessing for speed
   - Monitor progress with callbacks

---

## Additional Resources

- [API Reference](../docs/reference/api-reference.md)
- [Task Documentation](../docs/tasks/overview.md)
- [Architecture Guide](../docs/architecture.md)
- [Installation Guide](../docs/getting-started/installation.md)

---

## Support

For issues and questions:
- Check the [documentation](../docs/)
- Review the [troubleshooting guide](#troubleshooting)
- Open an issue on GitHub
