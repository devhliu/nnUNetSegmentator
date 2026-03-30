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

Canonical schema:

- `task_id` is canonical and must be `Dataset<数字>_<model_name>`.
- `model_name` + `task_id` is 1:1 within a task.
- Local install key uses the same canonical string as `task_id`.

## 2. Current task <-> model mapping

| Task | Model name | Task ID (canonical) | Modality | Comments |
|---|---|---|---|---|
| `body` | `body` | `Dataset299_body` | `CT` | Canonical task_id. |
| `body` | `body_fast` | `Dataset300_body_fast` | `CT` | Canonical task_id. |
| `body_composition` | `body_composition` | `Dataset1005_body_composition` | `CT` | Canonical task_id. |
| `body_mr` | `body_mr` | `Dataset1506_body_mr` | `MR` | Canonical task_id. |
| `cerebral_bleed` | `cerebral_bleed` | `Dataset150_cerebral_bleed` | `CT` | Canonical task_id. |
| `deep_psma` | `deep_psma` | `Dataset881_deep_psma` | `PET` | Canonical task_id. |
| `dukeseg` | `dukeseg_skeleton` | `Dataset1004_dukeseg_skeleton` | `CT` | Canonical task_id. |
| `dukeseg` | `dukeseg_model2` | `Dataset1001_dukeseg_model2` | `CT` | Canonical task_id. |
| `dukeseg` | `dukeseg_model3` | `Dataset1002_dukeseg_model3` | `CT` | Canonical task_id. |
| `gtrc` | `gtrc_psma` | `Dataset881_gtrc_psma` | `PETCT` | Canonical task_id. |
| `gtrc` | `gtrc_fdg` | `Dataset882_gtrc_fdg` | `PETCT` | Canonical task_id. |
| `gtrc` | `gtrc_lupsma` | `Dataset883_gtrc_lupsma` | `SPECTCT` | Canonical task_id. |
| `heartchambers_highres` | `heartchambers_highres` | `Dataset301_heartchambers_highres` | `CT` | Canonical task_id. |
| `kidney_cysts` | `kidney_cysts` | `Dataset1504_kidney_cysts` | `CT` | Canonical task_id. |
| `lion` | `fdg` | `Dataset789_fdg` | `PT` | Canonical task_id. |
| `lion` | `psma` | `Dataset711_psma` | `PT` | Canonical task_id. |
| `liver_lesions` | `liver_lesions` | `Dataset1502_liver_lesions` | `CT` | Canonical task_id. |
| `liver_lesions` | `liver_lesions_mr` | `Dataset1503_liver_lesions_mr` | `MR` | Canonical task_id. |
| `liver_vessels` | `liver_vessels` | `Dataset1501_liver_vessels` | `CT` | Canonical task_id. |
| `lung_nodules` | `lung_nodules` | `Dataset504_lung_nodules` | `CT` | Canonical task_id. |
| `lung_vessels` | `lung_vessels` | `Dataset117_lung_vessels` | `CT` | Canonical task_id. |
| `pleural_pericard_effusion` | `pleural_pericard_effusion` | `Dataset1505_pleural_pericard_effusion` | `CT` | Canonical task_id. |
| `tissue_types` | `tissue_types` | `Dataset481_tissue_types` | `CT` | Canonical task_id. |
| `tissue_types` | `tissue_4_types` | `Dataset485_tissue_4_types` | `CT` | Canonical task_id. |
| `tissue_types_mr` | `tissue_types_mr` | `Dataset1507_tissue_types_mr` | `MR` | Canonical task_id. |
| `total` | `total_organs` | `Dataset291_total_organs` | `CT` | Canonical task_id. |
| `total` | `total_vertebrae` | `Dataset292_total_vertebrae` | `CT` | Canonical task_id. |
| `total` | `total_cardiac` | `Dataset293_total_cardiac` | `CT` | Canonical task_id. |
| `total` | `total_muscles` | `Dataset294_total_muscles` | `CT` | Canonical task_id. |
| `total` | `total_ribs` | `Dataset295_total_ribs` | `CT` | Canonical task_id. |
| `total` | `total_fast` | `Dataset297_total_fast` | `CT` | Canonical task_id. |
| `total` | `total_fastest` | `Dataset298_total_fastest` | `CT` | Canonical task_id. |
| `total_mr` | `total_mr_organs` | `Dataset850_total_mr_organs` | `MR` | Canonical task_id. |
| `total_mr` | `total_mr_fast` | `Dataset852_total_mr_fast` | `MR` | Canonical task_id. |
| `ts2d` | `ts2d_v2_organs` | `Dataset900_ts2d_v2_organs` | `CT` | Canonical task_id. |
| `ts2d` | `ts2d_v2_vertebrae` | `Dataset901_ts2d_v2_vertebrae` | `CT` | Canonical task_id. |
| `ts2d` | `ts2d_v2_cardiac` | `Dataset902_ts2d_v2_cardiac` | `CT` | Canonical task_id. |
| `ts2d` | `ts2d_v2_muscles` | `Dataset903_ts2d_v2_muscles` | `CT` | Canonical task_id. |
| `ts2d` | `ts2d_v2_ribs` | `Dataset904_ts2d_v2_ribs` | `CT` | Canonical task_id. |
| `tsxr` | `tsxr_v2` | `Dataset910_tsxr_v2` | `XR` | Canonical task_id. |
| `vertebrae_body` | `vertebrae_body` | `Dataset257_vertebrae_body` | `CT` | Canonical task_id. |
