"""
BOA (Body and Organ Analysis) Tasks

Body-composition-analysis models from BOA (University of Duisburg-Essen):

- ``boa_body_parts``: coarse body parts (torso, head, arms, legs)
- ``boa_body_regions``: body regions (subcutaneous tissue, muscle, cavities,
  bone, glands, ...)

Both models are nnUNetv2 trainers without mirroring trained at a 5.0 mm slice
thickness (``resample_only_thickness``). The pipeline resamples the slice
thickness only, keeping the in-plane spacing, runs the full fold ensemble and
applies the ported BOA cleanup postprocessing.
"""

from ._helpers import SingleModelTask
from ._sources import BOA_SMALL_OBJECT_SIZE, boa_url
from ..labels.tables import BODY_PARTS_LABELS, BODY_REGION_LABELS


# BOA expects a 5.0 mm slice thickness while preserving the in-plane spacing.
_BOA_PREPROCESSING = [
    {
        "type": "resample",
        "name": "resample",
        "params": {"spacing": [5.0, 5.0, 5.0], "only_thickness": True},
    }
]

# BOA ships five folds (0-4) and trains with mirroring disabled.
_BOA_INFERENCE = {"folds": ("all",), "use_mirroring": False}


class BoaBodyPartsTask(SingleModelTask):
    """BOA body parts segmentation (torso, head, arms, legs)."""

    name = "boa_body_parts"
    description = "BOA body parts segmentation (torso, head, arms, legs)"
    TASK_ID = "Dataset543_BCA_body_parts"
    URL = boa_url("Dataset543_BCA_body_parts")
    LABELS = BODY_PARTS_LABELS
    PRE = _BOA_PREPROCESSING
    INFERENCE = _BOA_INFERENCE
    POST = [
        {
            "type": "boa_body_parts",
            "name": "boa_body_parts",
            "params": {"threshold": BOA_SMALL_OBJECT_SIZE},
        }
    ]


class BoaBodyRegionsTask(SingleModelTask):
    """BOA body regions segmentation."""

    name = "boa_body_regions"
    description = "BOA body regions segmentation"
    TASK_ID = "Dataset542_BCA_inference"
    URL = boa_url("Dataset542_BCA_inference")
    LABELS = BODY_REGION_LABELS
    PRE = _BOA_PREPROCESSING
    INFERENCE = _BOA_INFERENCE
    POST = [
        {
            "type": "boa_body_regions",
            "name": "boa_body_regions",
            "params": {},
        }
    ]