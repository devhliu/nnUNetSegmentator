# Task–Model Mapping Reference

This reference captures current runtime task-model mappings and cardinality constraints.

## 1. Relationship constraints

| Relationship | Cardinality | Enforced by code | Notes |
|---|---|---|---|
| Task -> model entries | 1:N | Yes | `TaskDefinition.models` is a dict. |
| `model_name` -> `task_id` (within one task) | 1:1 | Yes | A model name key maps to one `ModelInfo`. |
| `task_id` -> `model_name` (within one task) | 1:1 or 1:N | No | Multiple model names may share the same task ID. |
| Global uniqueness of `model_name` | Not guaranteed | No | Avoid cross-task collisions. |
| Global uniqueness of `task_id` | Not guaranteed | No | Same ID can appear under different tasks. |

## 2. Current task <-> model mapping

| Task | Model name | Task ID (numeric, except `XXX`) | Modality | Comments |
|---|---|---|---|---|
| `body` | `body` | 299 | `CT` | Standard numeric ID from registry. |
| `body` | `body_fast` | 300 | `CT` | Standard numeric ID from registry. |
| `body_composition` | `body_composition` | 1005 | `CT` | Standard numeric ID from registry. |
| `body_mr` | `body_mr` | `XXX` | `MR` | Placeholder value from source registry. |
| `cerebral_bleed` | `cerebral_bleed` | 150 | `CT` | Standard numeric ID from registry. |
| `deep_psma` | `deep_psma` | 881 | `PET` | Normalized from source `Dataset881_PSMA_PET`. |
| `dukeseg` | `dukeseg_skeleton` | 1004 | `CT` | Standard numeric ID from registry. |
| `dukeseg` | `dukeseg_model2` | 1001 | `CT` | Standard numeric ID from registry. |
| `dukeseg` | `dukeseg_model3` | 1002 | `CT` | Standard numeric ID from registry. |
| `gtrc` | `gtrc_psma` | 881 | `PETCT` | Normalized from source `Dataset881_PSMA_PET`. |
| `gtrc` | `gtrc_fdg` | 882 | `PETCT` | Normalized from source `Dataset882_FDG_PET`. |
| `gtrc` | `gtrc_lupsma` | 883 | `SPECTCT` | Normalized from source `Dataset883_LuPSMA_SPECT`. |
| `heartchambers_highres` | `heartchambers_highres` | 301 | `CT` | Standard numeric ID from registry. |
| `kidney_cysts` | `kidney_cysts` | `XXX` | `CT` | Placeholder value from source registry. |
| `lion` | `fdg` | 789 | `PT` | Normalized from source `Dataset789_Tumors`. |
| `lion` | `psma` | 711 | `PT` | Normalized from source `Dataset711_PSMA`. |
| `liver_lesions` | `liver_lesions` | `XXX` | `CT` | Placeholder value from source registry. |
| `liver_lesions` | `liver_lesions_mr` | `XXX` | `MR` | Placeholder value from source registry. |
| `liver_vessels` | `liver_vessels` | `XXX` | `CT` | Placeholder value from source registry. |
| `lung_nodules` | `lung_nodules` | 504 | `CT` | Standard numeric ID from registry. |
| `lung_vessels` | `lung_vessels` | 117 | `CT` | Standard numeric ID from registry. |
| `pleural_pericard_effusion` | `pleural_pericard_effusion` | `XXX` | `CT` | Placeholder value from source registry. |
| `tissue_types` | `tissue_types` | 481 | `CT` | Standard numeric ID from registry. |
| `tissue_types` | `tissue_4_types` | 485 | `CT` | Standard numeric ID from registry. |
| `tissue_types_mr` | `tissue_types_mr` | `XXX` | `MR` | Placeholder value from source registry. |
| `total` | `total_organs` | 291 | `CT` | Standard numeric ID from registry. |
| `total` | `total_vertebrae` | 292 | `CT` | Standard numeric ID from registry. |
| `total` | `total_cardiac` | 293 | `CT` | Standard numeric ID from registry. |
| `total` | `total_muscles` | 294 | `CT` | Standard numeric ID from registry. |
| `total` | `total_ribs` | 295 | `CT` | Standard numeric ID from registry. |
| `total` | `total_fast` | 297 | `CT` | Standard numeric ID from registry. |
| `total` | `total_fastest` | 298 | `CT` | Standard numeric ID from registry. |
| `total_mr` | `total_mr_organs` | 850 | `MR` | Standard numeric ID from registry. |
| `total_mr` | `total_mr_fast` | 852 | `MR` | Standard numeric ID from registry. |
| `ts2d` | `ts2d_v2_organs` | 900 | `CT` | Standard numeric ID from registry. |
| `ts2d` | `ts2d_v2_vertebrae` | 901 | `CT` | Standard numeric ID from registry. |
| `ts2d` | `ts2d_v2_cardiac` | 902 | `CT` | Standard numeric ID from registry. |
| `ts2d` | `ts2d_v2_muscles` | 903 | `CT` | Standard numeric ID from registry. |
| `ts2d` | `ts2d_v2_ribs` | 904 | `CT` | Standard numeric ID from registry. |
| `tsxr` | `tsxr_v2` | 910 | `XR` | Standard numeric ID from registry. |
| `vertebrae_body` | `vertebrae_body` | 257 | `CT` | Standard numeric ID from registry. |
