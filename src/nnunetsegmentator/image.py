"""
Nibabel-backed image compatibility helpers.

This module exposes a small SimpleITK-like API used by the project while keeping
native nibabel array ordering (X, Y, Z) end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        SecondaryCaptureImageStorage,
        PYDICOM_IMPLEMENTATION_UID,
        generate_uid,
    )

    HAS_PYDICOM = True
except ImportError:  # pragma: no cover
    HAS_PYDICOM = False

logger = logging.getLogger(__name__)

sitkNearestNeighbor = 0
sitkLinear = 1
sitkBSpline = 3
sitkFloat32 = np.float32


def _direction_matrix(direction: Sequence[float]) -> np.ndarray:
    return np.asarray(direction, dtype=float).reshape(3, 3)


def _affine_from_geometry(
    spacing_xyz: Sequence[float], origin_xyz: Sequence[float], direction: Sequence[float]
) -> np.ndarray:
    direction_matrix = _direction_matrix(direction)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = direction_matrix @ np.diag(np.asarray(spacing_xyz, dtype=float))
    affine[:3, 3] = np.asarray(origin_xyz, dtype=float)
    return affine


def _geometry_from_affine(
    affine: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, ...]]:
    matrix = np.asarray(affine[:3, :3], dtype=float)
    spacing = np.linalg.norm(matrix, axis=0)
    safe_spacing = np.where(spacing > 0, spacing, 1.0)
    direction = matrix / safe_spacing
    origin = tuple(float(v) for v in affine[:3, 3])
    return (
        tuple(float(v) for v in spacing),
        origin,
        tuple(float(v) for v in direction.reshape(-1)),
    )


@dataclass
class Image:
    """Simple image container with geometry methods."""

    _array_xyz: np.ndarray
    _spacing_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    _origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _direction: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    def __post_init__(self) -> None:
        self._array_xyz = np.asarray(self._array_xyz)
        if self._array_xyz.ndim != 3:
            raise ValueError(f"Image must be 3D, got shape {self._array_xyz.shape}")

    def clone(self) -> "Image":
        return Image(
            self._array_xyz.copy(),
            tuple(self._spacing_xyz),
            tuple(self._origin_xyz),
            tuple(self._direction),
        )

    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice], ...]]) -> "Image":
        sliced = np.asarray(self._array_xyz[key])
        if sliced.ndim == 2:
            sliced = sliced[:, :, np.newaxis]
        if sliced.ndim != 3:
            raise ValueError(f"Unsupported slice shape: {sliced.shape}")
        out = Image(sliced)
        out.CopyInformation(self)
        return out

    def GetSpacing(self) -> Tuple[float, float, float]:
        return tuple(float(v) for v in self._spacing_xyz)

    def SetSpacing(self, spacing: Sequence[float]) -> None:
        if len(spacing) != 3:
            raise ValueError("Spacing must have 3 values (x, y, z)")
        self._spacing_xyz = tuple(float(v) for v in spacing)

    def GetOrigin(self) -> Tuple[float, float, float]:
        return tuple(float(v) for v in self._origin_xyz)

    def SetOrigin(self, origin: Sequence[float]) -> None:
        if len(origin) != 3:
            raise ValueError("Origin must have 3 values (x, y, z)")
        self._origin_xyz = tuple(float(v) for v in origin)

    def GetDirection(self) -> Tuple[float, ...]:
        return tuple(float(v) for v in self._direction)

    def SetDirection(self, direction: Sequence[float]) -> None:
        if len(direction) != 9:
            raise ValueError("Direction must have 9 values (3x3)")
        self._direction = tuple(float(v) for v in direction)

    def GetSize(self) -> Tuple[int, int, int]:
        x, y, z = self._array_xyz.shape
        return int(x), int(y), int(z)

    @property
    def array(self) -> np.ndarray:
        return self._array_xyz

    @array.setter
    def array(self, value: np.ndarray) -> None:
        value = np.asarray(value)
        if value.ndim != 3:
            raise ValueError(f"Image array must be 3D, got {value.ndim}D")
        self._array_xyz = value

    def CopyInformation(self, other: "Image") -> None:
        self._spacing_xyz = tuple(other.GetSpacing())
        self._origin_xyz = tuple(other.GetOrigin())
        self._direction = tuple(other.GetDirection())

    def to_nifti(self) -> nib.Nifti1Image:
        affine = _affine_from_geometry(self._spacing_xyz, self._origin_xyz, self._direction)
        return nib.Nifti1Image(self._array_xyz, affine)

    @classmethod
    def from_nifti(cls, nifti_image: nib.Nifti1Image) -> "Image":
        data_xyz = np.asarray(nifti_image.dataobj)
        if data_xyz.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI, got {data_xyz.ndim}D")
        spacing, origin, direction = _geometry_from_affine(nifti_image.affine)
        return cls(data_xyz, spacing, origin, direction)


class Transform:
    """Placeholder transform class for API compatibility."""


class ResampleImageFilter:
    """Simple resample filter using scipy.ndimage.zoom."""

    def __init__(self) -> None:
        self._output_spacing: Optional[Tuple[float, float, float]] = None
        self._interpolator: int = sitkLinear
        self._output_direction: Optional[Tuple[float, ...]] = None
        self._output_origin: Optional[Tuple[float, float, float]] = None
        self._size_xyz: Optional[Tuple[int, int, int]] = None

    def SetOutputSpacing(self, spacing: Sequence[float]) -> None:
        self._output_spacing = tuple(float(v) for v in spacing)

    def SetInterpolator(self, interpolator: int) -> None:
        self._interpolator = int(interpolator)

    def SetOutputDirection(self, direction: Sequence[float]) -> None:
        self._output_direction = tuple(float(v) for v in direction)

    def SetOutputOrigin(self, origin: Sequence[float]) -> None:
        self._output_origin = tuple(float(v) for v in origin)

    def SetSize(self, size_xyz: Sequence[int]) -> None:
        if len(size_xyz) != 3:
            raise ValueError("Size must have 3 entries (x, y, z)")
        self._size_xyz = tuple(int(v) for v in size_xyz)

    def SetTransform(self, _transform: Transform) -> None:
        return None

    def Execute(self, image: Image) -> Image:
        output_spacing = self._output_spacing or image.GetSpacing()
        output_origin = self._output_origin or image.GetOrigin()
        output_direction = self._output_direction or image.GetDirection()

        input_array = GetArrayFromImage(image)
        input_spacing = image.GetSpacing()

        if self._size_xyz is not None:
            target_shape_xyz = self._size_xyz
            zoom_factors = [target_shape_xyz[i] / input_array.shape[i] for i in range(3)]
        else:
            zoom_factors = [
                input_spacing[0] / output_spacing[0],
                input_spacing[1] / output_spacing[1],
                input_spacing[2] / output_spacing[2],
            ]

        order = 0 if self._interpolator == sitkNearestNeighbor else 1
        resampled = zoom(input_array, zoom=zoom_factors, order=order)

        result = GetImageFromArray(resampled)
        result.SetSpacing(output_spacing)
        result.SetOrigin(output_origin)
        result.SetDirection(output_direction)
        return result


class LabelIntensityStatisticsImageFilter:
    """Minimal compatibility for GetPhysicalSize(label)."""

    def __init__(self) -> None:
        self._sizes: Dict[int, float] = {}

    def Execute(self, label_image: Image, _intensity_image: Optional[Image] = None) -> None:
        arr = GetArrayFromImage(label_image)
        spacing = label_image.GetSpacing()
        voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
        self._sizes = {}
        for v in np.unique(arr):
            label = int(v)
            if label <= 0:
                continue
            self._sizes[label] = float(np.sum(arr == label) * voxel_volume)

    def GetPhysicalSize(self, label: int) -> float:
        return float(self._sizes.get(int(label), 0.0))


class ImageSeriesReader:
    """Minimal DICOM series reader using pydicom."""

    def __init__(self) -> None:
        self._file_names: List[str] = []

    def GetGDCMSeriesFileNames(self, directory: str) -> List[str]:
        directory_path = Path(directory)
        return sorted(str(p) for p in directory_path.glob("*.dcm"))

    def SetFileNames(self, file_names: Sequence[str]) -> None:
        self._file_names = [str(f) for f in file_names]

    def Execute(self) -> Image:
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required for DICOM series reading")
        if not self._file_names:
            raise ValueError("No DICOM filenames set")

        datasets = [pydicom.dcmread(path) for path in self._file_names]

        def _sort_key(ds: Dataset) -> float:
            if "ImagePositionPatient" in ds and len(ds.ImagePositionPatient) >= 3:
                return float(ds.ImagePositionPatient[2])
            if "InstanceNumber" in ds:
                return float(ds.InstanceNumber)
            return 0.0

        datasets.sort(key=_sort_key)

        slices = [np.asarray(ds.pixel_array) for ds in datasets]
        volume = np.stack(slices, axis=2)

        first = datasets[0]
        pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
        spacing_x = float(pixel_spacing[0])
        spacing_y = float(pixel_spacing[1])

        if len(datasets) > 1 and all("ImagePositionPatient" in ds for ds in datasets[:2]):
            spacing_z = abs(
                float(datasets[1].ImagePositionPatient[2])
                - float(datasets[0].ImagePositionPatient[2])
            )
            if spacing_z == 0.0:
                spacing_z = float(getattr(first, "SliceThickness", 1.0))
        else:
            spacing_z = float(getattr(first, "SliceThickness", 1.0))

        origin = getattr(first, "ImagePositionPatient", [0.0, 0.0, 0.0])
        origin_xyz = (float(origin[0]), float(origin[1]), float(origin[2]))

        direction = (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

        orientation = getattr(first, "ImageOrientationPatient", None)
        if orientation is not None and len(orientation) >= 6:
            row = np.asarray(orientation[:3], dtype=float)
            col = np.asarray(orientation[3:6], dtype=float)
            normal = np.cross(row, col)
            matrix = np.column_stack([row, col, normal])
            direction = tuple(float(v) for v in matrix.reshape(-1))

        return Image(volume, (spacing_x, spacing_y, spacing_z), origin_xyz, direction)


class ImageFileWriter:
    """Minimal single-slice DICOM writer."""

    def __init__(self) -> None:
        self._filename: Optional[str] = None

    def SetImageIO(self, _image_io: str) -> None:
        return None

    def SetFileName(self, filename: str) -> None:
        self._filename = filename

    def Execute(self, image: Image) -> None:
        if self._filename is None:
            raise ValueError("Filename not set for ImageFileWriter")
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required for DICOM writing")

        array = np.asarray(GetArrayFromImage(image))
        if array.ndim == 3:
            if array.shape[2] != 1:
                raise ValueError("ImageFileWriter expects a single-slice image")
            array2d = array[:, :, 0]
        elif array.ndim == 2:
            array2d = array
        else:
            raise ValueError(f"Unsupported image shape for DICOM writing: {array.shape}")

        if array2d.dtype != np.uint16:
            if np.issubdtype(array2d.dtype, np.floating):
                array2d = np.clip(array2d, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            else:
                array2d = array2d.astype(np.uint16)

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        ds = FileDataset(self._filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "OT"
        ds.SeriesInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.FrameOfReferenceUID = generate_uid()
        ds.PatientName = "Anonymous"
        ds.PatientID = "000000"

        ds.Rows, ds.Columns = array2d.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.PixelData = array2d.tobytes()

        spacing = image.GetSpacing()
        ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
        ds.SliceThickness = float(spacing[2])

        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(self._filename)


def GetImageFromArray(array: np.ndarray) -> Image:
    arr = np.asarray(array)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if arr.ndim != 3:
        raise ValueError(f"Expected array compatible with 3D image, got shape {arr.shape}")

    return Image(arr)


def GetArrayFromImage(image: Image) -> np.ndarray:
    return np.asarray(image.array)


def ReadImage(path: Union[str, Path]) -> Image:
    path = Path(path)
    if path.is_dir():
        reader = ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(str(path))
        if not file_names:
            raise ValueError(f"No DICOM files found in {path}")
        reader.SetFileNames(file_names)
        return reader.Execute()

    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".nii.gz") or path.suffix in {".nii"}:
        return Image.from_nifti(nib.load(str(path)))

    if path.suffix.lower() == ".dcm":
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required for DICOM reading")
        ds = pydicom.dcmread(str(path))
        arr = np.asarray(ds.pixel_array)
        image = GetImageFromArray(arr[:, :, np.newaxis])
        spacing_x, spacing_y = getattr(ds, "PixelSpacing", [1.0, 1.0])
        spacing_z = getattr(ds, "SliceThickness", 1.0)
        image.SetSpacing((float(spacing_x), float(spacing_y), float(spacing_z)))
        origin = getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])
        image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
        return image

    raise ValueError(f"Unsupported image format for path: {path}")


def WriteImage(image: Union[Image, np.ndarray], path: Union[str, Path]) -> None:
    path = Path(path)

    if isinstance(image, np.ndarray):
        image = GetImageFromArray(image)

    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".nii.gz") or path.suffix == ".nii":
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(image.to_nifti(), str(path))
        return

    if path.suffix.lower() == ".dcm":
        writer = ImageFileWriter()
        writer.SetFileName(str(path))
        writer.Execute(image)
        return

    raise ValueError(f"Unsupported output format for path: {path}")


def Clamp(image: Image, lowerBound: float, upperBound: float) -> Image:
    array = np.clip(GetArrayFromImage(image), lowerBound, upperBound)
    out = GetImageFromArray(array)
    out.CopyInformation(image)
    return out


def Resample(
    image: Image,
    reference_image: Image,
    interpolator: int = sitkLinear,
    _output_pixel_type: Union[type, np.dtype, None] = None,
    output_origin: Optional[Sequence[float]] = None,
    output_spacing: Optional[Sequence[float]] = None,
    output_direction: Optional[Sequence[float]] = None,
) -> Image:
    filt = ResampleImageFilter()
    filt.SetOutputSpacing(output_spacing or reference_image.GetSpacing())
    filt.SetOutputOrigin(output_origin or reference_image.GetOrigin())
    filt.SetOutputDirection(output_direction or reference_image.GetDirection())
    filt.SetSize(reference_image.GetSize())
    filt.SetInterpolator(interpolator)
    return filt.Execute(image)
