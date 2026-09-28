"""Tests for the unified DICOM flow: series -> NIfTI -> segmentation -> DICOM-SEG."""

import numpy as np
import pytest

from nnunetsegmentator import image as sitk
from nnunetsegmentator.io.readers import ImageReader
from nnunetsegmentator.io.dicom_export import DICOMSEGExporter

pydicom = pytest.importorskip("pydicom")
pytest.importorskip("highdicom")

ROWS, COLS, NUM_SLICES = 8, 8, 4
SPACING = (1.0, 1.0, 2.0)


def _write_dicom_series(directory, num_slices=NUM_SLICES, extensionless=False):
    """Write a minimal but highdicom-compatible CT series."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    paths = []
    for i in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.PatientID = "TEST^PATIENT"
        dataset.PatientName = "TEST^PATIENT"
        dataset.PatientBirthDate = ""
        dataset.PatientSex = "O"
        dataset.StudyID = "1"
        dataset.StudyDate = "20260101"
        dataset.StudyTime = "000000"
        dataset.AccessionNumber = ""
        dataset.ReferringPhysicianName = ""
        dataset.Modality = "CT"
        dataset.Rows = ROWS
        dataset.Columns = COLS
        dataset.SamplesPerPixel = 1
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelRepresentation = 0
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.PixelData = bytes([i * 10] * (ROWS * COLS))
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.ImagePositionPatient = [0.0, 0.0, float(i) * 2.0]
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 2.0
        dataset.InstanceNumber = i + 1

        name = f"slice_{i}" if extensionless else f"slice_{i}.dcm"
        path = directory / name
        dataset.save_as(str(path), enforce_file_format=True)
        paths.append(path)
    return paths


def _write_rescaled_series(directory, iop, rows=3, cols=5, slices=4,
                           spacing=(2.0, 3.0), slope=2.0, intercept=-1000.0):
    """Write a non-square series with a modality LUT.

    Returns the ground-truth (slices, rows, cols) HU stack in DICOM order.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    truth = []
    for k in range(slices):
        stored = (np.arange(rows * cols, dtype=np.int16).reshape(rows, cols) + k * 100)
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.ImplementationClassUID = generate_uid()
        ds.preamble = b"\x00" * 128

        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1
        ds.PatientName = "Test^Synthetic"
        ds.PatientID = "SYN001"
        ds.PatientBirthDate = ""
        ds.PatientSex = "O"
        ds.StudyID = "1"
        ds.StudyDate = "20260101"
        ds.StudyTime = "000000"
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.FrameOfReferenceUID = frame_uid

        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
        ds.RescaleType = "HU"

        ds.PixelSpacing = [spacing[0], spacing[1]]
        ds.SliceThickness = 4.0
        ds.ImageOrientationPatient = [float(v) for v in iop]
        ds.ImagePositionPatient = [0.0, 0.0, float(k) * 4.0]
        ds.SliceLocation = float(k) * 4.0
        ds.PixelData = stored.tobytes()

        ds.save_as(str(directory / f"slice_{k:03d}.dcm"))
        truth.append(stored.astype(np.float64) * slope + intercept)
    return np.stack(truth, axis=0)


def _seg_image(seg_xyz, spacing=SPACING):
    """Framework image carrying the reference spacing used by the fixtures."""
    image = sitk.GetImageFromArray(seg_xyz)
    image.SetSpacing(spacing)
    return image


def _label_map_image():
    """Framework label map in (x, y, z) order."""
    seg_xyz = np.zeros((COLS, ROWS, NUM_SLICES), dtype=np.uint8)
    seg_xyz[:, :, 1:3] = 1
    seg_xyz[2:5, 2:5, 2] = 2
    return _seg_image(seg_xyz), seg_xyz


class TestSeriesDiscovery:
    def test_finds_extensionless_dicom_files(self, tmp_path):
        _write_dicom_series(tmp_path, extensionless=True)
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(tmp_path))
        assert len(files) == NUM_SLICES

    def test_find_dcm_files(self, tmp_path):
        _write_dicom_series(tmp_path)
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(tmp_path))
        assert len(files) == NUM_SLICES


class TestDicomReading:
    def test_read_records_series_reference(self, tmp_path):
        _write_dicom_series(tmp_path)
        image, metadata = ImageReader.read(tmp_path)

        assert metadata["format"] == "dicom"
        assert metadata["dicom_dir"] == str(tmp_path)
        assert image.GetSize() == (COLS, ROWS, NUM_SLICES)
        assert image.GetSpacing() == (1.0, 1.0, 2.0)

    def test_read_list_records_files(self, tmp_path):
        paths = _write_dicom_series(tmp_path)
        _, metadata = ImageReader.read([str(p) for p in paths])
        assert metadata["dicom_files"] == [str(p) for p in paths]


class TestDicomToNiftiConversion:
    """The DICOM->NIfTI step now goes through dicom2nifti."""

    def test_applies_modality_lut_on_non_square_grid(self, tmp_path):
        truth = _write_rescaled_series(
            tmp_path, iop=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], rows=3, cols=5, slices=4)
        image, metadata = ImageReader.read(tmp_path)

        # dicom2nifti reorients to LAS, so (i, j, k) = (cols, rows, slices).
        assert image.GetSize() == (5, 3, 4)
        assert metadata["num_slices"] == 4
        assert len(metadata["dicom_files"]) == 4

        # HU values (slope/intercept applied) are preserved up to reorientation.
        values = np.sort(np.asarray(image.array).ravel())
        assert np.allclose(values, np.sort(truth.ravel()))

    def test_select_primary_series_picks_largest(self, tmp_path):
        from nnunetsegmentator.io.dicom_utils import select_primary_series

        main = _write_dicom_series(tmp_path / "main", num_slices=4)
        scout = _write_dicom_series(tmp_path / "scout", num_slices=2)

        paths, datasets = select_primary_series(
            [str(p) for p in main + scout])

        assert len(datasets) == 4
        assert set(paths) == {str(p) for p in main}


class TestDICOMSEGExporter:
    def test_export_end_to_end(self, tmp_path):
        _write_dicom_series(tmp_path)
        seg_image, seg_xyz = _label_map_image()

        output_path = tmp_path / "out" / "case_seg.dcm"
        result = DICOMSEGExporter().export(
            segmentation=seg_image,
            reference_dicom_dir=tmp_path,
            output_path=output_path,
            labels={1: "liver", 2: "lesion"},
        )

        assert result == output_path
        dataset = pydicom.dcmread(str(result))
        assert dataset.Modality == "SEG"
        segment_labels = {
            int(item.SegmentNumber): str(item.SegmentLabel)
            for item in dataset.SegmentSequence
        }
        assert segment_labels == {1: "liver", 2: "lesion"}

        # Every foreground voxel appears exactly once across binary frames.
        expected_foreground = int((seg_xyz > 0).sum())
        assert int(dataset.pixel_array.sum()) == expected_foreground

    def test_export_from_file_list(self, tmp_path):
        paths = _write_dicom_series(tmp_path)
        seg_image, seg_xyz = _label_map_image()

        result = DICOMSEGExporter().export(
            segmentation=seg_image,
            reference_dicom_dir=[str(p) for p in paths],
            output_path=tmp_path / "seg.dcm",
            labels={1: "liver"},
        )
        assert result.exists()

        dataset = pydicom.dcmread(str(result))
        assert int(dataset.pixel_array.sum()) == int((seg_xyz > 0).sum())

    def test_export_rejects_frame_mismatch(self, tmp_path):
        _write_dicom_series(tmp_path)
        seg_xyz = np.zeros((COLS, ROWS, NUM_SLICES + 1), dtype=np.uint8)
        seg_xyz[:, :, 0] = 1

        with pytest.raises(ValueError, match="frames"):
            DICOMSEGExporter().export(
                segmentation=seg_xyz,
                reference_dicom_dir=tmp_path,
                output_path=tmp_path / "seg.dcm",
                labels={1: "liver"},
            )

    def test_export_rejects_empty_label_map(self, tmp_path):
        _write_dicom_series(tmp_path)
        # ndarray inputs follow the NIfTI (i, j, k) order of the source volume
        seg_empty = np.zeros((COLS, ROWS, NUM_SLICES), dtype=np.uint8)

        with pytest.raises(ValueError, match="empty"):
            DICOMSEGExporter().export(
                segmentation=seg_empty,
                reference_dicom_dir=tmp_path,
                output_path=tmp_path / "seg.dcm",
                labels={1: "liver"},
            )

    def test_rejects_spacing_mismatch(self, tmp_path):
        _write_dicom_series(tmp_path)
        seg_xyz = np.zeros((COLS, ROWS, NUM_SLICES), dtype=np.uint8)
        seg_xyz[:, :, 1] = 1
        # Reference series uses ((1, 1, 2)); a different slice spacing must fail.
        bad = _seg_image(seg_xyz, spacing=(1.0, 1.0, 5.0))

        with pytest.raises(ValueError, match="spacing"):
            DICOMSEGExporter().export(
                segmentation=bad,
                reference_dicom_dir=tmp_path,
                output_path=tmp_path / "seg.dcm",
                labels={1: "liver"},
            )

    def test_axial_round_trip_recovers_dicom_orientation(self, tmp_path):
        """A mask written back as DICOM-SEG must reproduce the source pixel grid."""
        truth = _write_rescaled_series(
            tmp_path, iop=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], rows=3, cols=5, slices=4)
        image, _ = ImageReader.read(tmp_path)

        # A non-degenerate pattern that is non-empty on every slice, defined on
        # HU values so it is invariant to the (permutating) reorientation.
        def mask(values):
            return (np.round(values).astype(np.int64) % 4 == 0).astype(np.uint8)

        seg_ijk = mask(np.asarray(image.array))

        result = DICOMSEGExporter().export(
            segmentation=seg_ijk,
            reference_dicom_dir=tmp_path,
            output_path=tmp_path / "seg.dcm",
            labels={1: "liver"},
        )

        dataset = pydicom.dcmread(str(result))
        frames = dataset.pixel_array.astype(bool)
        expected = mask(truth).astype(bool)
        assert frames.shape == expected.shape
        assert np.array_equal(frames, expected)

    def test_skips_empty_segments_and_renumbers(self, tmp_path):
        _write_dicom_series(tmp_path)
        seg_xyz = np.zeros((COLS, ROWS, NUM_SLICES), dtype=np.uint8)
        seg_xyz[:, :, 1] = 1
        seg_xyz[:, :, 2] = 3  # label 2 is absent -> must be dropped/renumbered

        result = DICOMSEGExporter().export(
            segmentation=_seg_image(seg_xyz),
            reference_dicom_dir=tmp_path,
            output_path=tmp_path / "seg.dcm",
            labels={1: "liver", 2: "lesion", 3: "kidney"},
        )

        dataset = pydicom.dcmread(str(result))
        segment_labels = {
            int(item.SegmentNumber): str(item.SegmentLabel)
            for item in dataset.SegmentSequence
        }
        assert segment_labels == {1: "liver", 2: "kidney"}
        assert int(dataset.pixel_array.sum()) == int((seg_xyz > 0).sum())

    def test_sets_cielab_display_color(self, tmp_path):
        from nnunetsegmentator.io.dicom_export import rgb_to_cielab_dicom

        _write_dicom_series(tmp_path)
        seg_xyz = np.zeros((COLS, ROWS, NUM_SLICES), dtype=np.uint8)
        seg_xyz[:, :, 1] = 1

        result = DICOMSEGExporter().export(
            segmentation=_seg_image(seg_xyz),
            reference_dicom_dir=tmp_path,
            output_path=tmp_path / "seg.dcm",
            labels={1: "zzz_custom_organ"},  # unmapped -> default palette colour
        )

        dataset = pydicom.dcmread(str(result))
        cielab = list(dataset.SegmentSequence[0].RecommendedDisplayCIELabValue)
        assert cielab == list(rgb_to_cielab_dicom((255, 0, 0)))

    def test_export_rejects_missing_series(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No DICOM files"):
            DICOMSEGExporter().export(
                segmentation=np.ones((2, 2, 2), dtype=np.uint8),
                reference_dicom_dir=empty_dir,
                output_path=tmp_path / "seg.dcm",
                labels={1: "liver"},
            )


class TestLabelNormalization:
    def test_normalize_both_orientations(self):
        from nnunetsegmentator.core.orchestrator import SegmentationOrchestrator

        id_to_name = SegmentationOrchestrator._normalize_labels_id_to_name(
            {1: "liver", 2: "lesion"})
        assert id_to_name == {1: "liver", 2: "lesion"}

        name_to_id = SegmentationOrchestrator._normalize_labels_id_to_name(
            {"liver": 1, "lesion": 2})
        assert name_to_id == {1: "liver", 2: "lesion"}


def _label_foreground_xyz():
    """Shared label map in (x, y, z) order with two foreground labels."""
    seg_xyz = np.zeros((COLS, ROWS, NUM_SLICES), dtype=np.uint8)
    seg_xyz[:, :, 1:3] = 1
    seg_xyz[2:5, 2:5, 2] = 2
    return seg_xyz


def _fake_task():
    from types import SimpleNamespace

    return SimpleNamespace(
        name="fake_task",
        output_config={"labels": {1: "liver", 2: "lesion"}},
        models={"m1": SimpleNamespace(labels={1: "liver", 2: "lesion"})},
    )


class _FakePipeline:
    """Minimal pipeline mapping the synthetic series to labels {1, 2}.

    The fixture stores slice i with intensity i*10, so dividing by 10 and
    clipping yields label 1 on slice 1 and label 2 on slices 2+.
    """

    def execute(self, context):
        context.intermediate_results["final_prediction"] = np.clip(
            context.input_array // 10, 0, 2).astype(np.uint8)
        return context


def _bare_orchestrator():
    from nnunetsegmentator.core.orchestrator import SegmentationOrchestrator

    return SegmentationOrchestrator()


class TestPrepareDicomInput:
    def test_writes_converted_nifti_named_after_series(self, tmp_path):
        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        image, metadata = ImageReader.read(series_dir)
        metadata["output_dir"] = str(tmp_path / "out")

        reference = _bare_orchestrator()._prepare_dicom_input(image, metadata)

        nifti_path = tmp_path / "out" / "series1_image.nii.gz"
        assert nifti_path.exists()
        assert metadata["nifti_path"] == str(nifti_path)
        assert reference == {"files": [str(p) for p in sorted(series_dir.glob("*.dcm"))]}
        loaded = sitk.ReadImage(str(nifti_path))
        assert loaded.GetSize() == image.GetSize()

    def test_rewrite_is_idempotent(self, tmp_path, monkeypatch):
        import nnunetsegmentator.core.orchestrator as orch_mod

        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        image, metadata = ImageReader.read(series_dir)
        metadata["output_dir"] = str(tmp_path / "out")

        calls = []
        original = orch_mod.ImageWriter.write

        def counting_write(img, path):
            calls.append(path)
            return original(img, path)

        monkeypatch.setattr(orch_mod.ImageWriter, "write", counting_write)
        orch = _bare_orchestrator()
        orch._prepare_dicom_input(image, metadata)
        orch._prepare_dicom_input(image, metadata)

        assert len(calls) == 1

    def test_skip_conversion_when_disabled(self, tmp_path):
        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        image, metadata = ImageReader.read(series_dir)
        metadata["output_dir"] = str(tmp_path / "out")

        reference = _bare_orchestrator()._prepare_dicom_input(
            image, metadata, save_converted_nifti=False)

        assert not (tmp_path / "out" / "series1_image.nii.gz").exists()
        assert "nifti_path" not in metadata
        assert reference == {"files": [str(p) for p in sorted(series_dir.glob("*.dcm"))]}

    def test_file_list_reference(self, tmp_path):
        paths = _write_dicom_series(tmp_path)
        image, metadata = ImageReader.read([str(p) for p in paths])
        metadata["output_dir"] = str(tmp_path / "out")

        reference = _bare_orchestrator()._prepare_dicom_input(image, metadata)

        assert reference == {"files": [str(p) for p in paths]}
        assert (tmp_path / "out" / "dicom_series_image.nii.gz").exists()

    def test_missing_reference_returns_none(self, tmp_path):
        image = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.uint8))
        metadata = {"format": "dicom", "output_dir": str(tmp_path)}

        assert _bare_orchestrator()._prepare_dicom_input(image, metadata) is None


class TestExportDicomSeg:
    def test_writes_seg_next_to_output(self, tmp_path):
        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        _, metadata = ImageReader.read(series_dir)
        metadata["labels"] = {1: "liver", 2: "lesion"}
        metadata["case_id"] = "case_seg"
        seg_image = _seg_image(_label_foreground_xyz())
        output_path = tmp_path / "out" / "case_seg.nii.gz"

        result = _bare_orchestrator()._export_dicom_seg(
            seg_image, metadata, output_path, {"files": str(series_dir)})

        seg_dcm = tmp_path / "out" / "case_seg.dcm"
        assert result == seg_dcm
        dataset = pydicom.dcmread(str(seg_dcm))
        assert dataset.Modality == "SEG"
        assert int(dataset.pixel_array.sum()) == int((_label_foreground_xyz() > 0).sum())

    def test_skips_without_labels(self, tmp_path):
        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        seg_image = _seg_image(_label_foreground_xyz())
        output_path = tmp_path / "case_seg.nii.gz"

        result = _bare_orchestrator()._export_dicom_seg(
            seg_image, {"case_id": "case_seg"}, output_path,
            {"files": str(series_dir)})

        assert result is None
        assert not (tmp_path / "case_seg.dcm").exists()

    def test_degrades_on_export_failure(self, tmp_path):
        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        # Frame count mismatch triggers a ValueError inside the exporter.
        bad_zyx = np.zeros((NUM_SLICES + 2, ROWS, COLS), dtype=np.uint8)
        bad_zyx[1] = 1
        metadata = {"labels": {1: "liver"}, "case_id": "case_seg"}

        result = _bare_orchestrator()._export_dicom_seg(
            sitk.GetImageFromArray(bad_zyx), metadata,
            tmp_path / "case_seg.nii.gz", {"files": str(series_dir)})

        assert result is None

    def test_degrades_when_highdicom_missing(self, tmp_path, monkeypatch):
        import nnunetsegmentator.io.dicom_export as dicom_export_mod

        class BrokenExporter:
            def __init__(self):
                raise ImportError("highdicom is required")

        monkeypatch.setattr(dicom_export_mod, "DICOMSEGExporter", BrokenExporter)
        seg_image = _seg_image(_label_foreground_xyz())
        metadata = {"labels": {1: "liver"}, "case_id": "case_seg"}

        result = _bare_orchestrator()._export_dicom_seg(
            seg_image, metadata, tmp_path / "case_seg.nii.gz",
            {"files": str(tmp_path)})

        assert result is None


class TestSegmentSingleDicomFlow:
    """End-to-end _segment_single runs over a synthetic DICOM series."""

    def _orchestrator(self):
        orch = _bare_orchestrator()
        orch.task = _fake_task()
        orch.pipeline = _FakePipeline()
        return orch

    def test_full_flow_writes_nifti_and_seg(self, tmp_path):
        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        output_path = tmp_path / "out" / "case_seg.nii.gz"

        result = self._orchestrator()._segment_single(str(series_dir), output_path)

        assert (tmp_path / "out" / "case_seg.nii.gz").exists()
        nifti_path = tmp_path / "out" / "series1_image.nii.gz"
        assert nifti_path.exists()
        assert result.metadata["nifti_path"] == str(nifti_path)

        seg_dcm = tmp_path / "out" / "case_seg.dcm"
        assert seg_dcm.exists()
        dataset = pydicom.dcmread(str(seg_dcm))
        assert dataset.Modality == "SEG"
        labels = {int(i.SegmentNumber): str(i.SegmentLabel)
                  for i in dataset.SegmentSequence}
        assert labels == {1: "liver", 2: "lesion"}

    def test_no_convert_nifti_keeps_seg(self, tmp_path):
        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        output_path = tmp_path / "out" / "case_seg.nii.gz"

        self._orchestrator()._segment_single(
            str(series_dir), output_path, save_converted_nifti=False)

        assert not (tmp_path / "out" / "series1_image.nii.gz").exists()
        assert (tmp_path / "out" / "case_seg.dcm").exists()

    def test_no_dicom_seg_keeps_nifti(self, tmp_path):
        series_dir = tmp_path / "series1"
        _write_dicom_series(series_dir)
        output_path = tmp_path / "out" / "case_seg.nii.gz"

        self._orchestrator()._segment_single(
            str(series_dir), output_path, export_dicom_seg=False)

        assert (tmp_path / "out" / "series1_image.nii.gz").exists()
        assert not (tmp_path / "out" / "case_seg.dcm").exists()


class TestCliDicomFlags:
    """--no-convert-nifti / --no-dicom-seg flags reach the orchestrator."""

    @staticmethod
    def _args(**overrides):
        from argparse import Namespace

        base = dict(
            input="in", output="out.nii.gz", task="fake", organs=None,
            modality=None, strategy="min_models", models=None, model=None,
            config=None, no_labels=False, compute_metrics=False,
            no_convert_nifti=False, no_dicom_seg=False,
        )
        base.update(overrides)
        return Namespace(**base)

    @staticmethod
    def _patch_core(monkeypatch, recorded):
        import importlib
        import types

        import nnunetsegmentator.core as core_mod

        # cli/__init__.py re-exports the main() function, shadowing the
        # module name; resolve the module explicitly.
        cli_main = importlib.import_module("nnunetsegmentator.cli.main")

        class FakeOrchestrator:
            def __init__(self, *args, **kwargs):
                pass

            def segment(self, input_data, output_path, **kwargs):
                recorded.update(kwargs)
                return types.SimpleNamespace(metrics={})

            def segment_batch(self, inputs, output_dir, **kwargs):
                recorded.update(kwargs)
                return [types.SimpleNamespace(metrics={}) for _ in inputs]

        class FakeConfig:
            def __init__(self):
                pass

            @classmethod
            def from_file(cls, path):
                return cls()

        monkeypatch.setattr(core_mod, "SegmentationOrchestrator", FakeOrchestrator)
        monkeypatch.setattr(core_mod, "Config", FakeConfig)
        monkeypatch.setattr(
            core_mod.TaskRegistry, "get_task", lambda name: _fake_task())
        return cli_main

    def test_segment_flags_default_enabled(self, monkeypatch):
        recorded = {}
        cli_main = self._patch_core(monkeypatch, recorded)

        cli_main.cmd_segment(self._args())

        assert recorded["save_converted_nifti"] is True
        assert recorded["export_dicom_seg"] is True

    def test_segment_flags_disable_behaviour(self, monkeypatch):
        recorded = {}
        cli_main = self._patch_core(monkeypatch, recorded)

        cli_main.cmd_segment(self._args(no_convert_nifti=True, no_dicom_seg=True))

        assert recorded["save_converted_nifti"] is False
        assert recorded["export_dicom_seg"] is False

    def test_batch_flags_passthrough(self, tmp_path, monkeypatch):
        (tmp_path / "a.nii.gz").touch()
        (tmp_path / "b.nii.gz").touch()
        recorded = {}
        cli_main = self._patch_core(monkeypatch, recorded)

        cli_main.cmd_batch(self._args(
            input=str(tmp_path), output=str(tmp_path / "out"),
            num_workers=1, multiprocessing=False))

        assert recorded["save_converted_nifti"] is True
        assert recorded["export_dicom_seg"] is True
