# Batch Processing Guide

## Basic Batch Processing

### Using Python API

```python
from pathlib import Path
from nnunetsegmentator import SegmentationOrchestrator

# Initialize orchestrator
orchestrator = SegmentationOrchestrator(task_name="gtrc")

# Get input files
input_dir = Path("data/patients")
input_files = list(input_dir.glob("*.nii.gz"))

# Process batch
results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=4
)

# Check results
successful = sum(1 for r in results if r is not None)
print(f"Processed: {successful}/{len(input_files)}")
```

### Using CLI

```bash
# Process directory
nnunetsegmentator batch -i data/patients/ -o results/ -t gtrc --num-workers 4

# Process specific files
nnunetsegmentator batch -i file1.nii.gz,file2.nii.gz -o results/ -t lion
```

## Advanced Batch Processing

### With Progress Callback

```python
def progress_callback(current, total, result):
    if result:
        print(f"Processed {current + 1}/{total}: {result.segmentation.GetSize()}")
    else:
        print(f"Failed {current + 1}/{total}")

results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=4,
    progress_callback=progress_callback
)
```

### Using Multiprocessing

```python
results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=4,
    use_multiprocessing=True
)
```

### Custom Output Naming

```python
for input_file in input_files:
    output_name = f"{input_file.stem}_seg.nii.gz"
    result = orchestrator.segment(
        input_data=str(input_file),
        output_path=f"results/{output_name}"
    )
```

## Performance Optimization

### Optimize for Speed

```python
results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=8,  # More workers
    use_multiprocessing=True  # Use multiprocessing
)
```

### Optimize for Memory

```python
results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=2,  # Fewer workers
    use_multiprocessing=False  # Use threading
)
```

## Error Handling

```python
results = orchestrator.segment_batch(
    input_list=input_files,
    output_dir="results/",
    num_workers=4
)

# Check for failures
failed = [f for f, r in zip(input_files, results) if r is None]
if failed:
    print(f"Failed to process: {failed}")
```

## Best Practices

1. **Use appropriate num_workers** - Match to CPU cores
2. **Monitor memory usage** - Reduce workers if OOM
3. **Use progress callbacks** - Track long-running batches
4. **Handle failures** - Check results for None values
5. **Log processing** - Keep track of what was processed
