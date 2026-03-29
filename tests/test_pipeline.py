"""Tests for Pipeline and PipelineContext"""

import pytest
import numpy as np
from nnunetsegmentator import image as sitk

from nnunetsegmentator.pipeline.base import (
    Pipeline,
    PipelineStep,
    PipelineContext,
)


class DummyStep(PipelineStep):
    """A dummy pipeline step for testing"""

    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.executed = False
        self.execute_count = 0

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.executed = True
        self.execute_count += 1
        context.metadata['step_executed'] = self.name
        context.metadata['execute_count'] = self.execute_count
        return context


class ArrayModifyingStep(PipelineStep):
    """A step that modifies the array"""

    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)

    def execute(self, context: PipelineContext) -> PipelineContext:
        array = context.input_array.copy()
        array = array * 2
        context.input_array = array
        return context


class TestPipelineContext:
    """Tests for PipelineContext dataclass"""

    def test_creation_with_defaults(self):
        """Test basic creation with defaults"""
        image = sitk.GetImageFromArray(np.zeros((10, 10, 10)))
        context = PipelineContext(
            input_image=image,
            input_array=np.zeros((10, 10, 10)),
            metadata={}
        )
        assert context.input_image is image
        assert context.intermediate_results == {}

    def test_get_array(self):
        """Test get_array method"""
        array = np.zeros((5, 5, 5))
        image = sitk.GetImageFromArray(array)
        context = PipelineContext(
            input_image=image,
            input_array=array,
            metadata={}
        )
        result = context.get_array()
        np.testing.assert_array_equal(result, array)

    def test_set_array(self):
        """Test set_array method"""
        array = np.zeros((5, 5, 5))
        image = sitk.GetImageFromArray(array)
        context = PipelineContext(
            input_image=image,
            input_array=array,
            metadata={}
        )
        new_array = np.ones((5, 5, 5))
        context.set_array(new_array)
        np.testing.assert_array_equal(context.input_array, new_array)

    def test_get_image(self):
        """Test get_image method"""
        array = np.zeros((5, 5, 5))
        image = sitk.GetImageFromArray(array)
        context = PipelineContext(
            input_image=image,
            input_array=array,
            metadata={}
        )
        result = context.get_image()
        assert result is image

    def test_set_image(self):
        """Test set_image method"""
        array = np.zeros((5, 5, 5))
        image = sitk.GetImageFromArray(array)
        context = PipelineContext(
            input_image=image,
            input_array=array,
            metadata={}
        )
        new_image = sitk.GetImageFromArray(np.ones((5, 5, 5)))
        context.set_image(new_image)
        assert context.input_image is new_image


class TestPipelineStep:
    """Tests for PipelineStep base class"""

    def test_initialization(self):
        """Test step initialization"""
        step = DummyStep("test_step", {"key": "value"})
        assert step.name == "test_step"
        assert step.config == {"key": "value"}
        assert not step.executed

    def test_callable(self):
        """Test that step is callable"""
        step = DummyStep("test_step")
        image = sitk.GetImageFromArray(np.zeros((5, 5, 5)))
        context = PipelineContext(
            input_image=image,
            input_array=np.zeros((5, 5, 5)),
            metadata={}
        )
        result = step(context)
        assert result is context
        assert step.executed

    def test_validate_config_default(self):
        """Test default config validation returns True"""
        step = DummyStep("test_step")
        assert step.validate_config() is True


class TestPipeline:
    """Tests for Pipeline class"""

    def test_empty_pipeline(self):
        """Test empty pipeline creation"""
        pipeline = Pipeline()
        assert len(pipeline) == 0
        assert pipeline.name == "pipeline"

    def test_named_pipeline(self):
        """Test pipeline with custom name"""
        pipeline = Pipeline(name="my_pipeline")
        assert pipeline.name == "my_pipeline"

    def test_add_step(self):
        """Test adding steps to pipeline"""
        pipeline = Pipeline()
        step = DummyStep("step1")
        result = pipeline.add_step(step)
        assert result is pipeline
        assert len(pipeline) == 1

    def test_add_step_fluent_interface(self):
        """Test fluent interface for adding steps"""
        pipeline = Pipeline()
        pipeline.add_step(DummyStep("step1")).add_step(DummyStep("step2"))
        assert len(pipeline) == 2

    def test_execute_empty_pipeline(self):
        """Test executing empty pipeline"""
        image = sitk.GetImageFromArray(np.zeros((5, 5, 5)))
        context = PipelineContext(
            input_image=image,
            input_array=np.zeros((5, 5, 5)),
            metadata={}
        )
        result = Pipeline().execute(context)
        assert result is context

    def test_execute_single_step(self):
        """Test executing pipeline with single step"""
        pipeline = Pipeline()
        step = DummyStep("test_step")
        pipeline.add_step(step)

        image = sitk.GetImageFromArray(np.zeros((5, 5, 5)))
        context = PipelineContext(
            input_image=image,
            input_array=np.zeros((5, 5, 5)),
            metadata={}
        )
        result = pipeline.execute(context)
        assert step.executed
        assert context.metadata['step_executed'] == "test_step"

    def test_execute_multiple_steps(self):
        """Test executing pipeline with multiple steps"""
        pipeline = Pipeline()
        step1 = DummyStep("step1")
        step2 = DummyStep("step2")
        pipeline.add_step(step1).add_step(step2)

        image = sitk.GetImageFromArray(np.zeros((5, 5, 5)))
        context = PipelineContext(
            input_image=image,
            input_array=np.zeros((5, 5, 5)),
            metadata={}
        )
        pipeline.execute(context)
        assert step1.executed
        assert step2.executed

    def test_pipeline_calls_execute(self):
        """Test that pipeline calls step.execute() not just step()"""
        pipeline = Pipeline()
        step = DummyStep("test_step")
        pipeline.add_step(step)

        image = sitk.GetImageFromArray(np.zeros((5, 5, 5)))
        context = PipelineContext(
            input_image=image,
            input_array=np.zeros((5, 5, 5)),
            metadata={}
        )
        pipeline.execute(context)
        assert step.executed
        assert step.execute_count == 1

    def test_pipeline_len(self):
        """Test pipeline length"""
        pipeline = Pipeline()
        assert len(pipeline) == 0
        pipeline.add_step(DummyStep("step1"))
        assert len(pipeline) == 1
        pipeline.add_step(DummyStep("step2"))
        assert len(pipeline) == 2

    def test_pipeline_repr(self):
        """Test pipeline string representation"""
        pipeline = Pipeline()
        pipeline.add_step(DummyStep("step1"))
        pipeline.add_step(DummyStep("step2"))
        repr_str = repr(pipeline)
        assert "Pipeline" in repr_str
        assert "step1" in repr_str
        assert "step2" in repr_str

    def test_pipeline_or_operator(self):
        """Test pipeline combination with | operator"""
        pipeline1 = Pipeline()
        pipeline1.add_step(DummyStep("step1"))

        pipeline2 = Pipeline()
        pipeline2.add_step(DummyStep("step2"))

        combined = pipeline1 | pipeline2
        assert len(combined) == 2
        assert combined.name == "pipeline_pipeline"

    def test_pipeline_or_with_step(self):
        """Test combining pipeline with single step"""
        pipeline = Pipeline()
        pipeline.add_step(DummyStep("step1"))

        combined = pipeline | DummyStep("step2")
        assert len(combined) == 2

    def test_validate_empty_pipeline(self):
        """Test validating empty pipeline"""
        pipeline = Pipeline()
        assert pipeline.validate() is True

    def test_validate_with_valid_steps(self):
        """Test validating pipeline with valid steps"""
        pipeline = Pipeline()
        pipeline.add_step(DummyStep("step1"))
        assert pipeline.validate() is True

    def test_hooks_pre_hook(self):
        """Test pre-execution hooks"""
        pipeline = Pipeline()
        pipeline.add_step(DummyStep("step1"))

        hook_executed = []

        def pre_hook(context):
            hook_executed.append('pre')
            return context

        pipeline.add_hook('pre', pre_hook)

        image = sitk.GetImageFromArray(np.zeros((5, 5, 5)))
        context = PipelineContext(
            input_image=image,
            input_array=np.zeros((5, 5, 5)),
            metadata={}
        )
        pipeline.execute(context)
        assert 'pre' in hook_executed

    def test_hooks_post_hook(self):
        """Test post-execution hooks"""
        pipeline = Pipeline()
        pipeline.add_step(DummyStep("step1"))

        hook_executed = []

        def post_hook(context):
            hook_executed.append('post')
            return context

        pipeline.add_hook('post', post_hook)

        image = sitk.GetImageFromArray(np.zeros((5, 5, 5)))
        context = PipelineContext(
            input_image=image,
            input_array=np.zeros((5, 5, 5)),
            metadata={}
        )
        pipeline.execute(context)
        assert 'post' in hook_executed

    def test_invalid_hook_stage_raises(self):
        """Test that invalid hook stage raises ValueError"""
        pipeline = Pipeline()
        with pytest.raises(ValueError):
            pipeline.add_hook('invalid', lambda ctx: ctx)
