# Custom Tasks Guide

## Creating a Custom Task

### Step 1: Define Task Configuration

```python
from nnunetsegmentator import TaskRegistry, Config
from nnunetsegmentator.pipeline.base import BasePreprocessor, BasePostprocessor

# Custom preprocessor
class MyPreprocessor(BasePreprocessor):
    def preprocess(self, image):
        # Your preprocessing logic
        return image

# Custom postprocessor
class MyPostprocessor(BasePostprocessor):
    def postprocess(self, segmentation):
        # Your postprocessing logic
        return segmentation

# Register custom task
TaskRegistry.register_task(
    task_name='my_custom_task',
    model_path='/path/to/model',
    config=Config(use_gpu=True, num_workers=4),
    preprocessor=MyPreprocessor(),
    postprocessor=MyPostprocessor(),
    labels={
        'structure1': 1,
        'structure2': 2
    }
)
```

### Step 2: Use Custom Task

```python
from nnunetsegmentator import SegmentationOrchestrator

orchestrator = SegmentationOrchestrator(task_name='my_custom_task')
result = orchestrator.segment('input.nii.gz', 'output.nii.gz')
```

## Advanced Custom Tasks

### Multiple Models

```python
# Register multiple tasks
TaskRegistry.register_task(
    task_name='task_a',
    model_path='/path/to/model_a',
    labels={'structure_a': 1}
)

TaskRegistry.register_task(
    task_name='task_b',
    model_path='/path/to/model_b',
    labels={'structure_b': 1}
)

# Use both tasks
orchestrator_a = SegmentationOrchestrator(task_name='task_a')
orchestrator_b = SegmentationOrchestrator(task_name='task_b')
```

### Custom Pipeline

```python
from nnunetsegmentator.pipeline import Pipeline
from nnunetsegmentator.pipeline.steps.preprocessing import ResampleStep

# Create custom pipeline
pipeline = Pipeline(name='custom')
pipeline.add_step(ResampleStep('resample', {'spacing': [1.0, 1.0, 1.0]}))

# Use with orchestrator
orchestrator = SegmentationOrchestrator(
    task_name='my_custom_task',
    pipeline=pipeline
)
```

## Best Practices

1. **Organize tasks** in configuration files
2. **Version control models** with task definitions
3. **Test custom tasks** before production use
4. **Document task requirements** clearly
