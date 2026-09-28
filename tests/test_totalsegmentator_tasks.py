"""
Tests for the TotalSegmentator task definitions (v2.5.0-weights).

Verifies official task ids, pipeline step sequences, crop configurations,
label mappings and postprocessing parameters against the official
TotalSegmentator implementation (map_tasks_config.py / map_to_binary.py /
python_api.py). Statistical models (body_stats_*) are out of scope.
"""

import pytest

import nnunetsegmentator.tasks  # noqa: F401  (registers all tasks on import)
from nnunetsegmentator.core.registry import TaskRegistry
from nnunetsegmentator.labels import LUNG_LOBES
from nnunetsegmentator.tasks import totalsegmentator as v250  # noqa: F401  (all 25 task classes)
from nnunetsegmentator.tasks.totalsegmentator import (
    TOTAL_LABELS,
    MR_LABELS,
    BodyMRTask,
    BodySegmentationTask,
    CerebralBleedTask,
    HeartChambersTask,
    KidneyCystsTask,
    LiverLesionsTask,
    LiverVesselsTask,
    LungNodulesTask,
    LungVesselsTask,
    PleuralPericardEffusionTask,
    TotalSegmentatorMRTask,
    TotalSegmentatorTask,
    TissueTypesMRTask,
    TissueTypesTask,
    VertebraeBodyTask,
)


def _step_types(pipeline):
    return [type(step).__name__ for step in pipeline.steps]


def _get_step(pipeline, cls_name):
    for step in pipeline.steps:
        if type(step).__name__ == cls_name:
            return step
    raise AssertionError(
        f"step {cls_name} not found in {_step_types(pipeline)}"
    )


class TestTotalTask:
    """CT total task: official task ids, multi-model concat, label mappings."""

    def test_official_task_ids(self):
        models = TotalSegmentatorTask.get_definition().models
        expected = {
            "total_organs": "Dataset291_TotalSegmentator_part1_organs_1559subj",
            "total_vertebrae": "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
            "total_cardiac": "Dataset293_TotalSegmentator_part3_cardiac_1559subj",
            "total_muscles": "Dataset294_TotalSegmentator_part4_muscles_1559subj",
            "total_ribs": "Dataset295_TotalSegmentator_part5_ribs_1559subj",
            "total_fast": "Dataset297_TotalSegmentator_total_3mm_1559subj",
            "total_fastest": "Dataset298_TotalSegmentator_total_6mm_1559subj",
        }
        assert {name: m.task_id for name, m in models.items()} == expected

    def test_labels_count(self):
        assert len(TOTAL_LABELS) == 117
        assert TOTAL_LABELS[51] == "heart"
        assert TOTAL_LABELS[117] == "costal_cartilages"

    def test_default_mode_uses_multi_model_concat(self):
        pipeline = TotalSegmentatorTask.get_default_pipeline("default")
        assert _step_types(pipeline) == ["MultiModelConcatStep"]
        step = _get_step(pipeline, "MultiModelConcatStep")
        assert step.config["model_names"] == [
            "total_organs", "total_vertebrae", "total_cardiac",
            "total_muscles", "total_ribs",
        ]
        assert len(step.config["label_mappings"]) == 5

    def test_default_mode_label_mappings(self):
        """Part-model local ids map to global ids by label name."""
        pipeline = TotalSegmentatorTask.get_default_pipeline("default")
        mappings = _get_step(pipeline, "MultiModelConcatStep").config["label_mappings"]

        # organs: identity mapping (part 1 shares the global numbering)
        assert mappings[0][1] == 1       # spleen
        assert mappings[0][24] == 24     # kidney_cyst_right

        # muscles: humerus_left is local 1 but global 69; skull local 23 -> 91
        assert mappings[3][1] == 69
        assert mappings[3][23] == 91

        # ribs: rib_left_1 local 1 -> global 92; costal_cartilages 26 -> 117
        assert mappings[4][1] == 92
        assert mappings[4][26] == 117

        # muscles: humerus_right local 2 -> global 70
        assert mappings[3][2] == 70

    def test_fast_and_fastest_single_inference(self):
        fast = TotalSegmentatorTask.get_default_pipeline("fast")
        assert _step_types(fast) == ["nnUNetInferenceStep"]
        assert _get_step(fast, "nnUNetInferenceStep").config["model_name"] == "total_fast"

        fastest = TotalSegmentatorTask.get_default_pipeline("fastest")
        assert _step_types(fastest) == ["nnUNetInferenceStep"]
        assert _get_step(fastest, "nnUNetInferenceStep").config["model_name"] == "total_fastest"


class TestTotalMRTask:
    """MR total task: official 850-853 task ids and concat label mapping."""

    def test_official_task_ids(self):
        models = TotalSegmentatorMRTask.get_definition().models
        expected = {
            "total_mr": "Dataset850_TotalSegMRI_part1_organs_1088subj",
            "total_mr_part2": "Dataset851_TotalSegMRI_part2_muscles_1088subj",
            "total_mr_fast": "Dataset852_TotalSegMRI_total_3mm_1088subj",
            "total_mr_fastest": "Dataset853_TotalSegMRI_total_6mm_1088subj",
        }
        assert {name: m.task_id for name, m in models.items()} == expected

    def test_mr_labels_count(self):
        assert len(MR_LABELS) == 50
        assert MR_LABELS[29] == "iliac_vena_right"
        assert MR_LABELS[30] == "humerus_left"
        assert MR_LABELS[50] == "brain"

    def test_default_mode_concat_and_mappings(self):
        pipeline = TotalSegmentatorMRTask.get_default_pipeline("default")
        assert _step_types(pipeline) == ["MultiModelConcatStep"]
        step = _get_step(pipeline, "MultiModelConcatStep")
        assert step.config["model_names"] == ["total_mr", "total_mr_part2"]

        mappings = step.config["label_mappings"]
        # organs part: identity
        assert mappings[0][1] == 1
        assert mappings[0][29] == 29
        # muscles part: local 1 -> global 30, local 21 -> global 50
        assert mappings[1][1] == 30
        assert mappings[1][21] == 50

    def test_fast_and_fastest(self):
        fast = TotalSegmentatorMRTask.get_default_pipeline("fast")
        assert _get_step(fast, "nnUNetInferenceStep").config["model_name"] == "total_mr_fast"
        fastest = TotalSegmentatorMRTask.get_default_pipeline("fastest")
        assert _get_step(fastest, "nnUNetInferenceStep").config["model_name"] == "total_mr_fastest"

    def test_no_pre_or_postprocessing(self):
        """Official MR workflow runs no preprocessing and no postprocessing."""
        for mode in ("default", "fast", "fastest"):
            types = _step_types(TotalSegmentatorMRTask.get_default_pipeline(mode))
            assert set(types) <= {"MultiModelConcatStep", "nnUNetInferenceStep"}


class TestSpecializedTasks:
    """Specialized tasks: official task ids, crop configs, postprocessing."""

    def test_official_task_ids(self):
        expected = {
            "lung_vessels": "Dataset117_lung_airways_arteries_veins_282subj",
            "body": "Dataset299_body_1559subj",
            "body_fast": "Dataset300_body_6mm_1559subj",
            "tissue_types": "Dataset481_tissue_1559subj",
            "tissue_4_types": "Dataset485_tissue_4types_1559subj",
            "heartchambers_highres": "Dataset301_heart_highres_1559subj",
            "cerebral_bleed": "Dataset150_icb_v0",
            "liver_vessels": "Dataset008_HepaticVessel",
            "lung_nodules": "Dataset913_lung_nodules",
            "vertebrae_body": "Dataset305_vertebrae_discs_1559subj",
            "liver_lesions": "Dataset591_ct_liver_lesions_842subj",
            "liver_lesions_mr": "Dataset589_ct_mri_liver_lesions_750subj",
            "kidney_cysts": "Dataset789_kidney_cyst_501subj",
            "pleural_pericard_effusion": "Dataset315_thoraxCT",
            "body_mr": "Dataset597_mri_body_139subj",
            "body_mr_fast": "Dataset598_mri_body_6mm_139subj",
            "tissue_types_mr": "Dataset925_MRI_tissue_subset_903subj",
        }
        actual = {}
        for task_cls in (LungVesselsTask, BodySegmentationTask, TissueTypesTask,
                         HeartChambersTask, CerebralBleedTask, LiverVesselsTask,
                         LungNodulesTask, VertebraeBodyTask, LiverLesionsTask,
                         KidneyCystsTask, PleuralPericardEffusionTask, BodyMRTask,
                         TissueTypesMRTask):
            for name, model in task_cls.get_definition().models.items():
                actual[name] = model.task_id
        assert actual == expected

    @pytest.mark.parametrize("mode", ["default", "fast"])
    def test_body_official_postprocessing(self, mode):
        """body task keeps the largest body_trunc and drops small extremities."""
        pipeline = BodySegmentationTask.get_default_pipeline(mode)
        assert _step_types(pipeline) == [
            "nnUNetInferenceStep", "LargestComponentStep", "RemoveSmallObjectsStep",
        ]
        largest = _get_step(pipeline, "LargestComponentStep").config
        assert largest["labels"] == ["body_trunc"]
        assert largest["keep_top_n"] == 1
        small = _get_step(pipeline, "RemoveSmallObjectsStep").config
        assert small["labels"] == ["body_extremities"]
        assert small["min_size_mm3"] == 50000

    def test_body_mr_has_no_postprocessing(self):
        pipeline = BodyMRTask.get_default_pipeline("default")
        assert _step_types(pipeline) == ["nnUNetInferenceStep"]
        fast = BodyMRTask.get_default_pipeline("fast")
        assert _step_types(fast) == ["nnUNetInferenceStep"]

    def test_heartchambers_crop_and_remove_outside(self):
        pipeline = HeartChambersTask.get_default_pipeline("default")
        assert _step_types(pipeline) == ["RegionCropStep", "nnUNetInferenceStep", "RegionRestoreStep"]

        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "total_fast"
        assert crop["crop_labels"] == ["heart"]
        assert crop["crop_addon"] == [5, 5, 5]
        # official crop model runs with tta=False
        assert crop.get("use_mirroring", False) is False

        restore = _get_step(pipeline, "RegionRestoreStep").config
        assert restore["remove_outside"] == {
            "labels": ["heart", "aorta", "inferior_vena_cava"],
            "dilation_mm": 10,
        }

    def test_liver_lesions_ct_workflow(self):
        pipeline = LiverLesionsTask.get_default_pipeline("default")
        assert _step_types(pipeline) == ["RegionCropStep", "nnUNetInferenceStep", "RegionRestoreStep"]
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "total_fast"
        assert crop["crop_labels"] == ["liver"]
        assert crop["crop_addon"] == [10, 10, 10]
        assert _get_step(pipeline, "nnUNetInferenceStep").config["model_name"] == "liver_lesions"
        assert _get_step(pipeline, "RegionRestoreStep").config.get("remove_outside") is None

    def test_liver_lesions_mr_workflow(self):
        """mode="mr" selects the MR crop model and MR target model."""
        pipeline = LiverLesionsTask.get_default_pipeline("mr")
        assert _step_types(pipeline) == ["RegionCropStep", "nnUNetInferenceStep", "RegionRestoreStep"]
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "total_mr_fast"
        assert crop["crop_labels"] == ["liver"]
        assert crop.get("crop_addon") is None  # official default addon
        assert _get_step(pipeline, "nnUNetInferenceStep").config["model_name"] == "liver_lesions_mr"

    def test_liver_lesions_model_steps_in_definition(self):
        config = LiverLesionsTask.get_definition().pipeline_config
        assert "steps" in config
        assert set(config["model_steps"]) == {"liver_lesions_mr"}

    def test_lung_vessels_crop_config(self):
        pipeline = LungVesselsTask.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "total_fast"
        assert crop["crop_labels"] == LUNG_LOBES
        assert crop.get("crop_addon") is None  # official default addon [3, 3, 3]

    def test_cerebral_bleed_crop_config(self):
        pipeline = CerebralBleedTask.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "total_fastest"
        assert crop["crop_labels"] == ["brain"]
        assert crop.get("crop_addon") is None

    def test_liver_vessels_crop_config(self):
        pipeline = LiverVesselsTask.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "total_fastest"
        assert crop["crop_labels"] == ["liver"]
        assert crop["crop_addon"] == [20, 20, 20]

    def test_kidney_cysts_crop_config(self):
        pipeline = KidneyCystsTask.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_labels"] == ["kidney_left", "kidney_right", "liver", "spleen", "colon"]
        assert crop["crop_addon"] == [10, 10, 10]

    def test_pleural_pericard_effusion_crop_config(self):
        pipeline = PleuralPericardEffusionTask.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "total_fastest"
        assert crop["crop_labels"] == LUNG_LOBES
        assert crop["crop_addon"] == [50, 50, 50]

    def test_tissue_types_four_types_mode(self):
        default = TissueTypesTask.get_default_pipeline("default")
        assert _get_step(default, "nnUNetInferenceStep").config["model_name"] == "tissue_types"
        four = TissueTypesTask.get_default_pipeline("four_types")
        assert _get_step(four, "nnUNetInferenceStep").config["model_name"] == "tissue_4_types"

    def test_vertebrae_body_plain_inference(self):
        pipeline = VertebraeBodyTask.get_default_pipeline("default")
        assert _step_types(pipeline) == ["nnUNetInferenceStep"]
        assert _get_step(pipeline, "nnUNetInferenceStep").config["model_name"] == "vertebrae_body"


class TestV250Tasks:
    """New v2.5.0-weights tasks: registration, task ids, labels, workflows."""

    V250_TASK_CLASSES = [
        v250.VertebraeMRTask, v250.BreastsTask, v250.VentriclePartsTask,
        v250.LiverSegmentsTask, v250.LiverSegmentsMRTask, v250.TrunkCavitiesTask,
        v250.BrainAneurysmTask, v250.VertebraePPTask, v250.AbdominalMusclesTask,
        v250.CraniofacialStructuresTask, v250.TeethTask,
    ]

    def test_all_tasks_registered(self):
        for task_cls in self.V250_TASK_CLASSES:
            definition = TaskRegistry.get_task(task_cls.name)
            assert definition.name == task_cls.name

    def test_models_findable(self):
        for task_cls in self.V250_TASK_CLASSES:
            for model_name in task_cls.get_definition().models:
                found = TaskRegistry.find_model(model_name)
                assert found is not None, model_name
                task_def, model_info = found
                assert task_def.name == task_cls.name
                assert model_info.name == model_name

    def test_official_task_ids(self):
        expected = {
            "vertebrae_mr": "Dataset756_mri_vertebrae_1076subj",
            "breasts": "Dataset527_breasts_1559subj",
            "ventricle_parts": "Dataset552_ventricle_parts_38subj",
            "liver_segments": "Dataset570_ct_liver_segments",
            "liver_segments_mr": "Dataset576_mri_liver_segments_120subj",
            "trunk_cavities": "Dataset343_mediastinum_1786subj",
            "brain_aneurysm": "Dataset615_MAXIMUS",
            "vertebrae_pp": "Dataset803_TotalSegmentator_vertebrae_inner_1559subj",
            "abdominal_muscles": "Dataset952_abdominal_muscles_167subj",
            "craniofacial_structures": "Dataset115_mandible",
            "teeth": "Dataset113_ToothFairy3",
        }
        actual = {}
        for task_cls in self.V250_TASK_CLASSES:
            for name, model in task_cls.get_definition().models.items():
                actual[name] = model.task_id
        assert actual == expected

    @pytest.mark.parametrize("task_name,expected_count", [
        ("vertebrae_mr", 25),
        ("breasts", 1),
        ("ventricle_parts", 12),
        ("liver_segments", 8),
        ("liver_segments_mr", 8),
        ("trunk_cavities", 4),
        ("brain_aneurysm", 1),
        ("vertebrae_pp", 24),
        ("abdominal_muscles", 22),
        ("craniofacial_structures", 7),
        ("teeth", 77),
    ])
    def test_label_counts(self, task_name, expected_count):
        definition = TaskRegistry.get_task(task_name)
        model_info = next(iter(definition.models.values()))
        assert len(model_info.labels) == expected_count
        assert definition.output_config["labels"] == model_info.labels

    def test_teeth_crop_config(self):
        """teeth crops with the craniofacial_structures model, 10mm addon."""
        pipeline = v250.TeethTask.get_default_pipeline("default")
        assert _step_types(pipeline) == ["RegionCropStep", "nnUNetInferenceStep", "RegionRestoreStep"]
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "craniofacial_structures"
        assert crop["crop_labels"] == ["teeth_lower", "teeth_upper"]
        assert crop["crop_addon"] == [10, 10, 10]

    def test_abdominal_muscles_crop_config(self):
        """abdominal_muscles crops with the 6mm body model, 5mm addon."""
        pipeline = v250.AbdominalMusclesTask.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "body_fast"
        assert crop["crop_labels"] == ["body_trunc"]
        assert crop["crop_addon"] == [5, 5, 5]

    def test_ventricle_parts_crop_config(self):
        pipeline = v250.VentriclePartsTask.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == "total_fastest"
        assert crop["crop_labels"] == ["brain"]
        assert crop["crop_addon"] == [0, 0, 0]

    @pytest.mark.parametrize("task_cls,crop_model,crop_addon", [
        (v250.LiverSegmentsTask, "total_fastest", [10, 10, 10]),
        (v250.LiverSegmentsMRTask, "total_mr_fast", [10, 10, 10]),
        (v250.CraniofacialStructuresTask, "total_fastest", [20, 20, 20]),
    ])
    def test_liver_and_craniofacial_crop_configs(self, task_cls, crop_model, crop_addon):
        pipeline = task_cls.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_model_name"] == crop_model
        assert crop["crop_addon"] == crop_addon

    def test_craniofacial_crop_labels(self):
        pipeline = v250.CraniofacialStructuresTask.get_default_pipeline("default")
        crop = _get_step(pipeline, "RegionCropStep").config
        assert crop["crop_labels"] == ["skull"]

    @pytest.mark.parametrize("task_cls", [
        v250.VertebraeMRTask, v250.BreastsTask, v250.TrunkCavitiesTask,
        v250.BrainAneurysmTask,
    ])
    def test_plain_inference_tasks(self, task_cls):
        """Tasks without crop run plain inference only."""
        pipeline = task_cls.get_default_pipeline("default")
        assert _step_types(pipeline) == ["nnUNetInferenceStep"]

    def test_vertebrae_pp_postprocessing(self):
        """VertebraePP runs the official vertebrae postprocessing after inference."""
        pipeline = v250.VertebraePPTask.get_default_pipeline("default")
        assert _step_types(pipeline) == [
            "nnUNetInferenceStep", "VertebraePPPostprocessStep",
        ]

    def test_vertebrae_mr_label_order(self):
        labels = v250.VertebraeMRTask.LABELS
        assert labels[1] == "sacrum"
        assert labels[2] == "vertebrae_L5"
        assert labels[25] == "vertebrae_C1"

    def test_vertebrae_pp_label_order(self):
        labels = v250.VertebraePPTask.LABELS
        assert labels[1] == "vertebrae_C1"
        assert labels[24] == "vertebrae_L5"


class TestNoClippingOrResampling:
    """Faithfulness: no task configures intensity clipping or explicit resampling."""

    ALL_CASES = [
        (TotalSegmentatorTask, ["default", "fast", "fastest", "roi"]),
        (TotalSegmentatorMRTask, ["default", "fast", "fastest"]),
        (LungVesselsTask, ["default"]),
        (BodySegmentationTask, ["default", "fast"]),
        (TissueTypesTask, ["default", "four_types"]),
        (HeartChambersTask, ["default"]),
        (CerebralBleedTask, ["default"]),
        (LiverVesselsTask, ["default"]),
        (LungNodulesTask, ["default"]),
        (VertebraeBodyTask, ["default"]),
        (LiverLesionsTask, ["default", "mr"]),
        (KidneyCystsTask, ["default"]),
        (PleuralPericardEffusionTask, ["default"]),
        (BodyMRTask, ["default", "fast"]),
        (TissueTypesMRTask, ["default"]),
    ] + [(cls, ["default"]) for cls in TestV250Tasks.V250_TASK_CLASSES]

    def test_forbidden_steps_absent(self):
        forbidden = {"ClipIntensityStep", "ResampleStep", "ResampleNibabelStep"}
        for task_cls, modes in self.ALL_CASES:
            for mode in modes:
                pipeline = task_cls.get_default_pipeline(mode)
                types = set(_step_types(pipeline))
                assert not types & forbidden, (
                    f"{task_cls.name} (mode={mode}) contains forbidden steps: "
                    f"{types & forbidden}"
                )
