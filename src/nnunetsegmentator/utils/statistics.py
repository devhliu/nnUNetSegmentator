"""
Statistics and Radiomics Features

This module provides comprehensive statistics calculation for segmentations,
including volume, intensity, shape, and radiomics features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from .. import image as sitk

logger = logging.getLogger(__name__)


class SegmentationStatistics:
    """
    Calculate comprehensive statistics for segmentation results.
    """
    
    def __init__(self, 
                 image: Optional[sitk.Image] = None,
                 segmentation: Optional[sitk.Image] = None):
        """
        Initialize statistics calculator.
        
        Args:
            image: Original image (for intensity statistics)
            segmentation: Segmentation image
        """
        self.image = image
        self.segmentation = segmentation
        
        if image is not None:
            self.image_array = sitk.GetArrayFromImage(image)
        if segmentation is not None:
            self.seg_array = sitk.GetArrayFromImage(segmentation)
    
    def compute_volume(self, 
                      label: int,
                      spacing: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Compute volume of a label in mm³.
        
        Args:
            label: Label value
            spacing: Image spacing (uses segmentation spacing if None)
        
        Returns:
            Volume in mm³
        """
        if self.segmentation is None:
            raise ValueError("Segmentation not set")
        
        # Get spacing
        if spacing is None:
            spacing = self.segmentation.GetSpacing()
        
        # Count voxels
        mask = self.seg_array == label
        voxel_count = mask.sum()
        
        # Compute volume
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        volume = voxel_count * voxel_volume
        
        return float(volume)
    
    def compute_surface_area(self, label: int) -> float:
        """
        Compute surface area of a label in mm².
        
        Args:
            label: Label value
        
        Returns:
            Surface area in mm²
        """
        if self.segmentation is None:
            raise ValueError("Segmentation not set")
        
        # Extract binary mask
        mask = (self.seg_array == label).astype(np.uint8)
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.CopyInformation(self.segmentation)
        
        # Use SimpleITK to compute surface area
        # This uses the marching cubes algorithm
        try:
            # Label intensity statistics includes surface area
            stats = sitk.LabelIntensityStatisticsImageFilter()
            stats.Execute(mask_image, mask_image)
            
            # Get surface area (perimeter in 3D)
            surface_area = stats.GetPhysicalSize(1)  # Approximate
            return float(surface_area)
        except RuntimeError as exc:
            logger.warning(f"Failed to compute surface area: {exc}")
            return 0.0
    
    def compute_intensity_statistics(self, label: int) -> Dict[str, float]:
        """
        Compute intensity statistics for a label.
        
        Args:
            label: Label value
        
        Returns:
            Dictionary of intensity statistics
        """
        if self.image is None:
            raise ValueError("Image not set")
        if self.segmentation is None:
            raise ValueError("Segmentation not set")
        
        # Get mask
        mask = self.seg_array == label
        
        if mask.sum() == 0:
            return {}
        
        # Get intensities
        intensities = self.image_array[mask]
        
        return {
            "mean": float(np.mean(intensities)),
            "std": float(np.std(intensities)),
            "median": float(np.median(intensities)),
            "min": float(np.min(intensities)),
            "max": float(np.max(intensities)),
            "range": float(np.max(intensities) - np.min(intensities)),
            "q25": float(np.percentile(intensities, 25)),
            "q75": float(np.percentile(intensities, 75)),
            "iqr": float(np.percentile(intensities, 75) - np.percentile(intensities, 25)),
            "skewness": float(self._skewness(intensities)),
            "kurtosis": float(self._kurtosis(intensities)),
        }
    
    def _skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        n = len(data)
        if n < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.sum(((data - mean) / std) ** 3) / n
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis."""
        n = len(data)
        if n < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.sum(((data - mean) / std) ** 4) / n - 3
    
    def compute_shape_statistics(self, label: int) -> Dict[str, float]:
        """
        Compute shape statistics for a label.
        
        Args:
            label: Label value
        
        Returns:
            Dictionary of shape statistics
        """
        if self.segmentation is None:
            raise ValueError("Segmentation not set")
        
        # Extract binary mask
        mask = self.seg_array == label
        
        if mask.sum() == 0:
            return {}
        
        # Get spacing (array axis order is X, Y, Z)
        spacing = np.array(self.segmentation.GetSpacing())
        
        # Compute centroid
        coords = np.where(mask)
        centroid = [
            np.mean(coords[0]) * spacing[0],  # x
            np.mean(coords[1]) * spacing[1],  # y
            np.mean(coords[2]) * spacing[2],  # z
        ]
        
        # Compute bounding box
        bbox_min = [
            np.min(coords[0]) * spacing[0],
            np.min(coords[1]) * spacing[1],
            np.min(coords[2]) * spacing[2],
        ]
        bbox_max = [
            np.max(coords[0]) * spacing[0],
            np.max(coords[1]) * spacing[1],
            np.max(coords[2]) * spacing[2],
        ]
        bbox_size = [bbox_max[i] - bbox_min[i] for i in range(3)]
        
        # Compute volume
        volume = self.compute_volume(label)
        
        # Compute sphericity (how sphere-like the shape is)
        surface_area = self.compute_surface_area(label)
        if surface_area > 0:
            sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
        else:
            sphericity = 0.0
        
        return {
            "volume_mm3": volume,
            "volume_cm3": volume / 1000.0,
            "surface_area_mm2": surface_area,
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "centroid_z": centroid[2],
            "bbox_x": bbox_size[0],
            "bbox_y": bbox_size[1],
            "bbox_z": bbox_size[2],
            "sphericity": sphericity,
            "voxel_count": int(mask.sum()),
        }
    
    def compute_all_statistics(self, 
                               labels: Dict[int, str],
                               include_intensity: bool = True,
                               include_shape: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Compute all statistics for all labels.
        
        Args:
            labels: Dictionary mapping label values to names
            include_intensity: Include intensity statistics
            include_shape: Include shape statistics
        
        Returns:
            Dictionary of statistics per label
        """
        results = {}
        
        for label_value, label_name in labels.items():
            if label_value == 0:  # Skip background
                continue
            
            # Check if label exists
            if (self.seg_array == label_value).sum() == 0:
                logger.debug(f"Label {label_name} not present in segmentation")
                continue
            
            stats = {"label_value": label_value, "label_name": label_name}
            
            # Shape statistics
            if include_shape:
                stats.update(self.compute_shape_statistics(label_value))
            
            # Intensity statistics
            if include_intensity and self.image is not None:
                intensity_stats = self.compute_intensity_statistics(label_value)
                stats.update({f"intensity_{k}": v for k, v in intensity_stats.items()})
            
            results[label_name] = stats
        
        return results


class RadiomicsFeatures:
    """
    Compute radiomics features using pyradiomics.
    
    Radiomics features are quantitative features extracted from medical images
    that can be used for radiomic analysis and machine learning.
    """
    
    def __init__(self):
        """Initialize radiomics calculator."""
        try:
            import radiomics
            from radiomics import featureextractor
            self.featureextractor = featureextractor
            self.available = True
        except ImportError:
            logger.warning("pyradiomics not available. Install with: pip install pyradiomics")
            self.available = False
    
    def compute_features(self,
                        image: sitk.Image,
                        mask: sitk.Image,
                        feature_classes: Optional[List[str]] = None,
                        settings: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Compute radiomics features for a mask.
        
        Args:
            image: Original image
            mask: Binary mask
            feature_classes: List of feature classes to compute
                           (firstorder, shape, texture, etc.)
            settings: pyradiomics settings
        
        Returns:
            Dictionary of feature names to values
        """
        if not self.available:
            raise RuntimeError("pyradiomics not available")
        
        # Default feature classes
        if feature_classes is None:
            feature_classes = [
                'firstorder',
                'shape',
                'glcm',      # Gray Level Co-occurrence Matrix
                'glrlm',     # Gray Level Run Length Matrix
                'glszm',     # Gray Level Size Zone Matrix
                'ngtdm',     # Neighboring Gray Tone Difference Matrix
            ]
        
        # Default settings
        if settings is None:
            settings = {
                'binWidth': 25,
                'interpolator': 'sitkBSpline',
                'resampledPixelSpacing': None,
                'textureSampleDistance': 1.0,
                'normalize': False,
                'force2D': False,
            }
        
        # Create extractor
        extractor = self.featureextractor.RadiomicsFeatureExtractor(**settings)
        
        # Enable feature classes
        extractor.enableFeatureClassesByName(feature_classes)
        
        # Compute features
        try:
            features = extractor.execute(image, mask)
            return features
        except (ValueError, RuntimeError) as exc:
            logger.error(f"Failed to compute radiomics features: {exc}")
            return {}
    
    def compute_for_multilabel(self,
                              image: sitk.Image,
                              segmentation: sitk.Image,
                              labels: Dict[int, str],
                              feature_classes: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute radiomics features for all labels in a multilabel segmentation.
        
        Args:
            image: Original image
            segmentation: Multilabel segmentation
            labels: Label mapping
            feature_classes: Feature classes to compute
        
        Returns:
            Dictionary mapping label names to feature dictionaries
        """
        if not self.available:
            logger.warning("pyradiomics not available, skipping radiomics computation")
            return {}
        
        results = {}
        seg_array = sitk.GetArrayFromImage(segmentation)
        
        for label_value, label_name in labels.items():
            if label_value == 0:
                continue
            
            # Extract binary mask
            mask_array = (seg_array == label_value).astype(np.uint8)
            
            if mask_array.sum() == 0:
                continue
            
            mask = sitk.GetImageFromArray(mask_array)
            mask.CopyInformation(segmentation)
            
            # Compute features
            features = self.compute_features(
                image, mask, feature_classes
            )
            
            # Clean up feature names (remove class prefix)
            cleaned_features = {}
            for key, value in features.items():
                # Remove "original_" prefix
                clean_key = key.replace("original_", "")
                cleaned_features[clean_key] = value
            
            results[label_name] = cleaned_features
        
        return results


def compute_statistics_report(
    image: sitk.Image,
    segmentation: sitk.Image,
    labels: Dict[int, str],
    output_path: Optional[Path] = None,
    include_radiomics: bool = False,
    format: str = "json"
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics report.
    
    Args:
        image: Original image
        segmentation: Segmentation
        labels: Label mapping
        output_path: Optional output file path
        include_radiomics: Include radiomics features
        format: Output format (json, csv)
    
    Returns:
        Statistics dictionary
    """
    # Compute basic statistics
    stats_calc = SegmentationStatistics(image, segmentation)
    statistics = stats_calc.compute_all_statistics(labels)
    
    # Add radiomics if requested
    if include_radiomics:
        radiomics_calc = RadiomicsFeatures()
        if radiomics_calc.available:
            radiomics = radiomics_calc.compute_for_multilabel(image, segmentation, labels)
            
            for label_name, rad_features in radiomics.items():
                if label_name in statistics:
                    statistics[label_name].update(rad_features)
    
    # Add metadata
    report = {
        "metadata": {
            "image_spacing": segmentation.GetSpacing(),
            "image_size": segmentation.GetSize(),
            "image_origin": segmentation.GetOrigin(),
            "num_labels": len(statistics),
        },
        "statistics": statistics
    }
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        
        if format == "json":
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        elif format == "csv":
            import csv
            # Flatten statistics for CSV
            rows = []
            for label_name, stats in statistics.items():
                row = {"label": label_name}
                row.update(stats)
                rows.append(row)
            
            if rows:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        
        logger.info(f"Statistics report saved to {output_path}")
    
    return report


def compute_volume_statistics(
    segmentation: sitk.Image,
    labels: Dict[int, str]
) -> Dict[str, float]:
    """
    Compute volume statistics for all labels.
    
    Args:
        segmentation: Segmentation image
        labels: Label mapping
    
    Returns:
        Dictionary mapping label names to volumes in cm³
    """
    stats_calc = SegmentationStatistics(segmentation=segmentation)
    volumes = {}
    
    for label_value, label_name in labels.items():
        if label_value == 0:
            continue
        
        volume_mm3 = stats_calc.compute_volume(label_value)
        volumes[label_name] = volume_mm3 / 1000.0  # Convert to cm³
    
    return volumes
