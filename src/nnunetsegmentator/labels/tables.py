"""
Organ Label Tables for nnUNetSegmentator

Central, per-model label tables for every TotalSegmentator task. Task classes
import their tables from here instead of declaring them inline, so organ
naming is managed in a single module.

Table contents are faithful to the official TotalSegmentator v2.5.0-weights
release (map_to_binary.py class maps / map_tasks_config.py task ids):

- CT "total" family: the 117-class full-resolution map plus the five part
  models (organs / vertebrae / cardiac / muscles / ribs) used by the
  multi-model concat and roi workflows.
- MR "total_mr" family: organs (29) + muscles (21) part maps and the
  remapped 50-class full map.
- Specialized and v2.5.0-weights tasks: one table per task.

The ``TASK_LABEL_TABLES`` registry maps task names to their default output
label table, and :func:`get_labels` looks tables up by task name.
"""

# --------------------------------------------------------------------------- #
# CT total (117 classes) and the 5-part decomposition
# --------------------------------------------------------------------------- #

# Label mappings for TotalSegmentator (117 classes)
TOTAL_LABELS = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages"
}

# 5-part model decomposition for efficient inference
PART_ORGANS_LABELS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
    9: "adrenal_gland_left", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland", 18: "small_bowel",
    19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate",
    23: "kidney_cyst_left", 24: "kidney_cyst_right"
}

PART_VERTEBRAE_LABELS = {
    1: "sacrum", 2: "vertebrae_S1", 3: "vertebrae_L5", 4: "vertebrae_L4",
    5: "vertebrae_L3", 6: "vertebrae_L2", 7: "vertebrae_L1", 8: "vertebrae_T12",
    9: "vertebrae_T11", 10: "vertebrae_T10", 11: "vertebrae_T9", 12: "vertebrae_T8",
    13: "vertebrae_T7", 14: "vertebrae_T6", 15: "vertebrae_T5", 16: "vertebrae_T4",
    17: "vertebrae_T3", 18: "vertebrae_T2", 19: "vertebrae_T1", 20: "vertebrae_C7",
    21: "vertebrae_C6", 22: "vertebrae_C5", 23: "vertebrae_C4", 24: "vertebrae_C3",
    25: "vertebrae_C2", 26: "vertebrae_C1"
}

PART_CARDIAC_LABELS = {
    1: "heart", 2: "aorta", 3: "pulmonary_vein", 4: "brachiocephalic_trunk",
    5: "subclavian_artery_right", 6: "subclavian_artery_left",
    7: "common_carotid_artery_right", 8: "common_carotid_artery_left",
    9: "brachiocephalic_vein_left", 10: "brachiocephalic_vein_right",
    11: "atrial_appendage_left", 12: "superior_vena_cava", 13: "inferior_vena_cava",
    14: "portal_vein_and_splenic_vein", 15: "iliac_artery_left",
    16: "iliac_artery_right", 17: "iliac_vena_left", 18: "iliac_vena_right"
}

PART_MUSCLES_LABELS = {
    1: "humerus_left", 2: "humerus_right", 3: "scapula_left", 4: "scapula_right",
    5: "clavicula_left", 6: "clavicula_right", 7: "femur_left", 8: "femur_right",
    9: "hip_left", 10: "hip_right", 11: "spinal_cord", 12: "gluteus_maximus_left",
    13: "gluteus_maximus_right", 14: "gluteus_medius_left", 15: "gluteus_medius_right",
    16: "gluteus_minimus_left", 17: "gluteus_minimus_right", 18: "autochthon_left",
    19: "autochthon_right", 20: "iliopsoas_left", 21: "iliopsoas_right",
    22: "brain", 23: "skull"
}

PART_RIBS_LABELS = {
    1: "rib_left_1", 2: "rib_left_2", 3: "rib_left_3", 4: "rib_left_4",
    5: "rib_left_5", 6: "rib_left_6", 7: "rib_left_7", 8: "rib_left_8",
    9: "rib_left_9", 10: "rib_left_10", 11: "rib_left_11", 12: "rib_left_12",
    13: "rib_right_1", 14: "rib_right_2", 15: "rib_right_3", 16: "rib_right_4",
    17: "rib_right_5", 18: "rib_right_6", 19: "rib_right_7", 20: "rib_right_8",
    21: "rib_right_9", 22: "rib_right_10", 23: "rib_right_11", 24: "rib_right_12",
    25: "sternum", 26: "costal_cartilages"
}


# --------------------------------------------------------------------------- #
# MR total (organs 29 + muscles 21 = 50 classes)
# --------------------------------------------------------------------------- #

# Official MR label maps (map_to_binary.class_map_parts_mr). The full-resolution
# "total_mr" workflow runs two part models (organs + muscles) whose outputs are
# concatenated by label name; the fast/fastest models predict all 50 classes at once.
MR_ORGANS_LABELS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
    9: "adrenal_gland_left", 10: "lung_left", 11: "lung_right",
    12: "esophagus", 13: "small_bowel", 14: "duodenum", 15: "colon",
    16: "urinary_bladder", 17: "prostate", 18: "sacrum", 19: "vertebrae",
    20: "intervertebral_discs", 21: "spinal_cord", 22: "heart", 23: "aorta",
    24: "inferior_vena_cava", 25: "portal_vein_and_splenic_vein",
    26: "iliac_artery_left", 27: "iliac_artery_right", 28: "iliac_vena_left",
    29: "iliac_vena_right",
}

MR_MUSCLES_LABELS = {
    1: "humerus_left", 2: "humerus_right", 3: "scapula_left", 4: "scapula_right",
    5: "clavicula_left", 6: "clavicula_right", 7: "femur_left", 8: "femur_right",
    9: "hip_left", 10: "hip_right", 11: "gluteus_maximus_left",
    12: "gluteus_maximus_right", 13: "gluteus_medius_left",
    14: "gluteus_medius_right", 15: "gluteus_minimus_left",
    16: "gluteus_minimus_right", 17: "autochthon_left", 18: "autochthon_right",
    19: "iliopsoas_left", 20: "iliopsoas_right", 21: "brain",
}

# MR label mapping (50 classes): organs 1-29 as-is, muscles remapped 30-50.
MR_LABELS = {
    **MR_ORGANS_LABELS,
    **{local_id + 29: name for local_id, name in MR_MUSCLES_LABELS.items()},
}


# --------------------------------------------------------------------------- #
# Specialized tasks (CT unless noted)
# --------------------------------------------------------------------------- #

LUNG_VESSELS_LABELS = {
    1: "lung_airways",
    2: "lung_airways_wall",
    3: "lung_arteries",
    4: "lung_veins",
}

# Shared by the "body" (CT) and "body_mr" (MR) tasks.
BODY_LABELS = {
    1: "body_trunc",
    2: "body_extremities",
}

TISSUE_TYPES_LABELS_3 = {
    1: "subcutaneous_fat",
    2: "torso_fat",
    3: "skeletal_muscle",
}

TISSUE_TYPES_LABELS_4 = {
    1: "subcutaneous_fat",
    2: "torso_fat",
    3: "skeletal_muscle",
    4: "intermuscular_fat",
}

# tissue_types_mr uses the 3-class map.
TISSUE_TYPES_MR_LABELS = TISSUE_TYPES_LABELS_3

HEART_CHAMBERS_LABELS = {
    1: "heart_myocardium",
    2: "heart_atrium_left",
    3: "heart_ventricle_left",
    4: "heart_atrium_right",
    5: "heart_ventricle_right",
    6: "aorta",
    7: "pulmonary_artery",
}

CEREBRAL_BLEED_LABELS = {
    1: "intracerebral_hemorrhage",
}

LIVER_VESSELS_LABELS = {
    1: "liver_vessels",
    2: "liver_tumor",
}

LUNG_NODULES_LABELS = {
    1: "lung",
    2: "lung_nodules",
}

VERTEBRAE_BODY_LABELS = {
    1: "vertebrae_body",
    2: "intervertebral_discs",
}

LIVER_LESIONS_LABELS = {
    1: "liver_lesions",
}

KIDNEY_CYSTS_LABELS = {
    1: "kidney_cyst_left",
    2: "kidney_cyst_right",
}

PLEURAL_PERICARD_EFFUSION_LABELS = {
    1: "lung_pleural",
    2: "pleural_effusion",
    3: "pericardial_effusion",
}


# --------------------------------------------------------------------------- #
# v2.5.0-weights tasks
# --------------------------------------------------------------------------- #

VERTEBRAE_MR_LABELS = {
    1: "sacrum",
    2: "vertebrae_L5",
    3: "vertebrae_L4",
    4: "vertebrae_L3",
    5: "vertebrae_L2",
    6: "vertebrae_L1",
    7: "vertebrae_T12",
    8: "vertebrae_T11",
    9: "vertebrae_T10",
    10: "vertebrae_T9",
    11: "vertebrae_T8",
    12: "vertebrae_T7",
    13: "vertebrae_T6",
    14: "vertebrae_T5",
    15: "vertebrae_T4",
    16: "vertebrae_T3",
    17: "vertebrae_T2",
    18: "vertebrae_T1",
    19: "vertebrae_C7",
    20: "vertebrae_C6",
    21: "vertebrae_C5",
    22: "vertebrae_C4",
    23: "vertebrae_C3",
    24: "vertebrae_C2",
    25: "vertebrae_C1",
}

BREASTS_LABELS = {
    1: "breast",
}

VENTRICLE_PARTS_LABELS = {
    1: "ventricle_frontal_horn_left",
    2: "ventricle_occipital_horn_left",
    3: "ventricle_body_left",
    4: "ventricle_temporal_horn_left",
    5: "ventricle_trigone_left",
    6: "ventricle_frontal_horn_right",
    7: "ventricle_occipital_horn_right",
    8: "ventricle_body_right",
    9: "ventricle_temporal_horn_right",
    10: "ventricle_trigone_right",
    11: "third_ventricle",
    12: "fourth_ventricle",
}

# Shared by "liver_segments" (CT) and "liver_segments_mr" (MR).
LIVER_SEGMENTS_LABELS = {
    1: "liver_segment_1",
    2: "liver_segment_2",
    3: "liver_segment_3",
    4: "liver_segment_4",
    5: "liver_segment_5",
    6: "liver_segment_6",
    7: "liver_segment_7",
    8: "liver_segment_8",
}

TRUNK_CAVITIES_LABELS = {
    1: "abdominal_cavity",
    2: "thoracic_cavity",
    3: "pericardium",
    4: "mediastinum",
}

BRAIN_ANEURYSM_LABELS = {
    1: "brain_aneurysm",
}

VERTEBRAE_PP_LABELS = {
    1: "vertebrae_C1",
    2: "vertebrae_C2",
    3: "vertebrae_C3",
    4: "vertebrae_C4",
    5: "vertebrae_C5",
    6: "vertebrae_C6",
    7: "vertebrae_C7",
    8: "vertebrae_T1",
    9: "vertebrae_T2",
    10: "vertebrae_T3",
    11: "vertebrae_T4",
    12: "vertebrae_T5",
    13: "vertebrae_T6",
    14: "vertebrae_T7",
    15: "vertebrae_T8",
    16: "vertebrae_T9",
    17: "vertebrae_T10",
    18: "vertebrae_T11",
    19: "vertebrae_T12",
    20: "vertebrae_L1",
    21: "vertebrae_L2",
    22: "vertebrae_L3",
    23: "vertebrae_L4",
    24: "vertebrae_L5",
}

ABDOMINAL_MUSCLES_LABELS = {
    1: "pectoralis_major_right",
    2: "pectoralis_major_left",
    3: "rectus_abdominis_right",
    4: "rectus_abdominis_left",
    5: "serratus_anterior_right",
    6: "serratus_anterior_left",
    7: "latissimus_dorsi_right",
    8: "latissimus_dorsi_left",
    9: "trapezius_right",
    10: "trapezius_left",
    11: "external_oblique_right",
    12: "external_oblique_left",
    13: "internal_oblique_right",
    14: "internal_oblique_left",
    15: "erector_spinae_right",
    16: "erector_spinae_left",
    17: "transversospinalis_right",
    18: "transversospinalis_left",
    19: "psoas_major_right",
    20: "psoas_major_left",
    21: "quadratus_lumborum_right",
    22: "quadratus_lumborum_left",
}

CRANIOFACIAL_STRUCTURES_LABELS = {
    1: "mandible",
    2: "teeth_lower",
    3: "skull",
    4: "head",
    5: "sinus_maxillary",
    6: "sinus_frontal",
    7: "teeth_upper",
}

TEETH_LABELS = {
    1: "lower_jawbone",
    2: "upper_jawbone",
    3: "left_inferior_alveolar_canal",
    4: "right_inferior_alveolar_canal",
    5: "left_maxillary_sinus",
    6: "right_maxillary_sinus",
    7: "pharynx",
    8: "bridge",
    9: "crown",
    10: "implant",
    11: "upper_right_central_incisor_fdi11",
    12: "upper_right_lateral_incisor_fdi12",
    13: "upper_right_canine_fdi13",
    14: "upper_right_first_premolar_fdi14",
    15: "upper_right_second_premolar_fdi15",
    16: "upper_right_first_molar_fdi16",
    17: "upper_right_second_molar_fdi17",
    18: "upper_right_third_molar_fdi18",
    19: "upper_left_central_incisor_fdi21",
    20: "upper_left_lateral_incisor_fdi22",
    21: "upper_left_canine_fdi23",
    22: "upper_left_first_premolar_fdi24",
    23: "upper_left_second_premolar_fdi25",
    24: "upper_left_first_molar_fdi26",
    25: "upper_left_second_molar_fdi27",
    26: "upper_left_third_molar_fdi28",
    27: "lower_left_central_incisor_fdi31",
    28: "lower_left_lateral_incisor_fdi32",
    29: "lower_left_canine_fdi33",
    30: "lower_left_first_premolar_fdi34",
    31: "lower_left_second_premolar_fdi35",
    32: "lower_left_first_molar_fdi36",
    33: "lower_left_second_premolar_fdi37",
    34: "lower_left_third_molar_fdi38",
    35: "lower_right_central_incisor_fdi41",
    36: "lower_right_lateral_incisor_fdi42",
    37: "lower_right_canine_fdi43",
    38: "lower_right_first_premolar_fdi44",
    39: "lower_right_second_premolar_fdi45",
    40: "lower_right_first_molar_fdi46",
    41: "lower_right_second_molar_fdi47",
    42: "lower_right_third_molar_fdi48",
    43: "left_mandibular_incisive_canal_fdi103",
    44: "right_mandibular_incisive_canal_fdi104",
    45: "lingual_canal",
    46: "upper_right_central_incisor_pulp_fdi111",
    47: "upper_right_lateral_incisor_pulp_fdi112",
    48: "upper_right_canine_pulp_fdi113",
    49: "upper_right_first_premolar_pulp_fdi114",
    50: "upper_right_second_premolar_pulp_fdi115",
    51: "upper_right_first_molar_pulp_fdi116",
    52: "upper_right_second_molar_pulp_fdi117",
    53: "upper_right_third_molar_pulp_fdi118",
    54: "upper_left_central_incisor_pulp_fdi121",
    55: "upper_left_lateral_incisor_pulp_fdi122",
    56: "upper_left_canine_pulp_fdi123",
    57: "upper_left_first_premolar_pulp_fdi124",
    58: "upper_left_second_premolar_pulp_fdi125",
    59: "upper_left_first_molar_pulp_fdi126",
    60: "upper_left_second_molar_pulp_fdi127",
    61: "upper_left_third_molar_pulp_fdi128",
    62: "lower_left_central_incisor_pulp_fdi131",
    63: "lower_left_lateral_incisor_pulp_fdi132",
    64: "lower_left_canine_pulp_fdi133",
    65: "lower_left_first_premolar_pulp_fdi134",
    66: "lower_left_second_premolar_pulp_fdi135",
    67: "lower_left_first_molar_pulp_fdi136",
    68: "lower_left_second_premolar_pulp_fdi137",
    69: "lower_left_third_molar_pulp_fdi138",
    70: "lower_right_central_incisor_pulp_fdi141",
    71: "lower_right_lateral_incisor_pulp_fdi142",
    72: "lower_right_canine_pulp_fdi143",
    73: "lower_right_first_premolar_pulp_fdi144",
    74: "lower_right_second_premolar_pulp_fdi145",
    75: "lower_right_first_molar_pulp_fdi146",
    76: "lower_right_second_molar_pulp_fdi147",
    77: "lower_right_third_molar_pulp_fdi148",
}


# --------------------------------------------------------------------------- #
# BOA (Body and Organ Analysis) body_composition_analysis tasks
# --------------------------------------------------------------------------- #

# BOA "body_parts" model (body_composition_analysis.body_parts.definition.BodyParts).
BODY_PARTS_LABELS = {
    1: "torso",
    2: "head",
    3: "leg_right",
    4: "leg_left",
    5: "arm_right",
    6: "arm_left",
}

# BOA "body_regions" model (body_composition_analysis.body_regions.definition.BodyRegion).
BODY_REGION_LABELS = {
    1: "subcutaneous_tissue",
    2: "muscle",
    3: "abdominal_cavity",
    4: "thoracic_cavity",
    5: "bone",
    6: "glands",
    7: "pericardium",
    8: "breast_implant",
    9: "mediastinum",
    10: "brain",
    11: "nervous_system",
}


# --------------------------------------------------------------------------- #
# Crop label sets and task registry
# --------------------------------------------------------------------------- #

# Official crop label set for lung-restricted tasks (map_tasks_config.py).
LUNG_LOBES = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
]


# Default output label table per task name. Tasks with multiple label tables
# (e.g. "tissue_types") map to their default-mode table.
TASK_LABEL_TABLES = {
    # CT / MR total
    "total": TOTAL_LABELS,
    "total_mr": MR_LABELS,
    # Specialized
    "lung_vessels": LUNG_VESSELS_LABELS,
    "body": BODY_LABELS,
    "body_mr": BODY_LABELS,
    "tissue_types": TISSUE_TYPES_LABELS_3,
    "tissue_types_mr": TISSUE_TYPES_MR_LABELS,
    "heartchambers_highres": HEART_CHAMBERS_LABELS,
    "cerebral_bleed": CEREBRAL_BLEED_LABELS,
    "liver_vessels": LIVER_VESSELS_LABELS,
    "lung_nodules": LUNG_NODULES_LABELS,
    "vertebrae_body": VERTEBRAE_BODY_LABELS,
    "liver_lesions": LIVER_LESIONS_LABELS,
    "kidney_cysts": KIDNEY_CYSTS_LABELS,
    "pleural_pericard_effusion": PLEURAL_PERICARD_EFFUSION_LABELS,
    # v2.5.0-weights
    "vertebrae_mr": VERTEBRAE_MR_LABELS,
    "breasts": BREASTS_LABELS,
    "ventricle_parts": VENTRICLE_PARTS_LABELS,
    "liver_segments": LIVER_SEGMENTS_LABELS,
    "liver_segments_mr": LIVER_SEGMENTS_LABELS,
    "trunk_cavities": TRUNK_CAVITIES_LABELS,
    "brain_aneurysm": BRAIN_ANEURYSM_LABELS,
    "vertebrae_pp": VERTEBRAE_PP_LABELS,
    "abdominal_muscles": ABDOMINAL_MUSCLES_LABELS,
    "craniofacial_structures": CRANIOFACIAL_STRUCTURES_LABELS,
    "teeth": TEETH_LABELS,
    # BOA body_composition_analysis
    "boa_body_parts": BODY_PARTS_LABELS,
    "boa_body_regions": BODY_REGION_LABELS,
}


def get_labels(task_name: str):
    """
    Return the default output label table for a task name.

    Args:
        task_name: Task name as registered in ``TaskRegistry`` (e.g. "total").

    Returns:
        The ``{label_id: label_name}`` table. The returned dict is shared
        with the task definitions - treat it as read-only.

    Raises:
        KeyError: If the task name is unknown (the error message lists the
            available task names).
    """
    try:
        return TASK_LABEL_TABLES[task_name]
    except KeyError:
        available = ", ".join(sorted(TASK_LABEL_TABLES))
        raise KeyError(
            f"Unknown task name '{task_name}'. Available: {available}"
        ) from None
