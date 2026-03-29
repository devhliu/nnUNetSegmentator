# API Reference

## Core Classes

### SegmentationOrchestrator

Main orchestrator for segmentation tasks.

#### Constructor

```python
SegmentationOrchestrator(task_name=None, config=None, model_path=None, pipeline=None)
```

#### Methods

- `segment(input_data, output_path=None, return_labels=True, compute_metrics=False)` - Segment single image
- `segment_batch(input_list, output_dir, num_workers=1, use_multiprocessing=False, progress_callback=None)` - Batch processing

### Config

Configuration container.

#### Methods

- `from_file(filepath)` - Load from YAML file
- `from_dict(data)` - Load from dictionary
- `update(**kwargs)` - Update configuration
- `setup_nnunet_environment()` - Set nnUNet environment variables

### TaskRegistry

Task registry (singleton).

#### Methods

- `register(task)` - Register new task
- `unregister(task_name)` - Unregister task
- `get_task(name)` - Get task definition
- `list_tasks()` - List all tasks

## Pipeline Classes

### Pipeline

Processing pipeline.

#### Methods

- `add_step(step)` - Add processing step
- `execute(context)` - Execute pipeline
- `validate()` - Validate pipeline

### PipelineStep

Base class for pipeline steps.

## Result Classes

### SegmentationResult

Container for segmentation results.

#### Attributes

- `segmentation` - Multi-label segmentation image
- `labels` - Dictionary of individual labels
- `metadata` - Input metadata
- `metrics` - Computed metrics

#### Methods

- `get_array()` - Get as numpy array
- `get_label_array(label_name)` - Get specific label as array
- `save(filepath)` - Save to file

## Utility Functions

### Metrics

- `compute_volume(segmentation, spacing)` - Compute volume
- `compute_dice(pred, gt)` - Compute Dice coefficient
- `compute_hausdorff(pred, gt)` - Compute Hausdorff distance

### Resampling

- `resample_image(image, spacing)` - Resample image
- `resample_to_reference(image, reference)` - Resample to reference

## Complete API Documentation

See [API Reference](reference/api-reference.md) for complete documentation.
