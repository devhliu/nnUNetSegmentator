"""
Preprocessing Pipeline Steps

This module provides preprocessing steps for image preparation before inference.
"""

from ... import image as sitk
import numpy as np
from ..base import PipelineStep, PipelineContext
import logging

logger = logging.getLogger(__name__)


class PreserveOriginalStep(PipelineStep):
    """
    Preserve original input array for later use.
    
    This step saves the current input array to context.metadata['original_input_array'],
    which is useful if subsequent steps modify the array (e.g., normalization)
    but postprocessing requires the original intensities (e.g., thresholding).
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute preservation.
        """
        context.metadata['original_input_array'] = context.input_array.copy()
        logger.debug("Preserved original input array in metadata")
        return context


class ResampleStep(PipelineStep):
    """
    Resample image to target spacing.
    
    This step resamples the input image to a specified target spacing,
    which is often required for nnUNet models that expect specific resolutions.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute resampling.
        
        Config options:
            spacing: Target spacing tuple (default: (1.5, 1.5, 1.5))
            interpolator: SimpleITK interpolator (default: sitk.sitkLinear)
        """
        target_spacing = self.config.get('spacing', (1.5, 1.5, 1.5))
        interpolator = self.config.get('interpolator', sitk.sitkLinear)
        
        image = context.input_image
        original_spacing = image.GetSpacing()
        
        logger.debug(f"Resampling from {original_spacing} to {target_spacing}")
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        
        # Calculate new size
        original_size = image.GetSize()
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]
        resampler.SetSize(new_size)
        
        context.input_image = resampler.Execute(image)
        context.metadata['original_spacing'] = original_spacing
        context.metadata['current_spacing'] = target_spacing
        
        # Update array
        context.input_array = sitk.GetArrayFromImage(context.input_image)
        
        return context


class ClipIntensityStep(PipelineStep):
    """
    Clip intensity values.
    
    This step clips image intensities to a specified range, which is useful
    for CT images where extreme values can affect normalization.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute intensity clipping.
        
        Config options:
            lower: Lower bound (default: -1000)
            upper: Upper bound (default: 1000)
        """
        lower = self.config.get('lower', -1000)
        upper = self.config.get('upper', 1000)
        
        logger.debug(f"Clipping intensities to [{lower}, {upper}]")
        
        context.input_image = sitk.Clamp(
            context.input_image,
            lowerBound=lower,
            upperBound=upper
        )
        
        # Update array
        context.input_array = sitk.GetArrayFromImage(context.input_image)
        
        return context


class NormalizeStep(PipelineStep):
    """
    Normalize image intensities.
    
    This step normalizes image intensities using various methods suitable
    for different modalities.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute normalization.
        
        Config options:
            method: Normalization method
                - 'minmax': Scale to [0, 1]
                - 'zscore': Zero mean, unit variance
                - 'ct': CT-specific normalization (clip to [-1000, 1000], scale to [0, 1])
        """
        method = self.config.get('method', 'minmax')
        
        array = sitk.GetArrayFromImage(context.input_image)
        
        logger.debug(f"Normalizing with method: {method}")
        
        if method == 'minmax':
            array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        elif method == 'zscore':
            array = (array - array.mean()) / (array.std() + 1e-8)
        elif method == 'ct':
            # CT normalization: clip to [-1000, 1000] then scale to [0, 1]
            array = np.clip(array, -1000, 1000)
            array = (array + 1000) / 2000
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        original_image = context.input_image
        context.input_array = array

        context.input_image = sitk.GetImageFromArray(array.astype(np.float32))
        context.input_image.CopyInformation(original_image)
        
        return context


class SUVConversionStep(PipelineStep):
    """
    Convert raw PET units (Bq/ml or CNTS) to SUV.
    
    This step requires 'suv_factor' in the context metadata, or it attempts
    to calculate it from provided DICOM metadata.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute SUV conversion.
        """
        logger.debug("Executing SUV Conversion")
        
        factor = context.metadata.get('suv_factor')
        
        if factor is None:
            # Try to calculate from DICOM metadata if available
            dicom_metadata = context.metadata.get('dicom_metadata')
            if dicom_metadata:
                from ...io.dicom_utils import calculate_suv_conversion_factor
                factor = calculate_suv_conversion_factor(dicom_metadata)
                logger.info(f"Calculated SUV conversion factor: {factor}")
        
        if factor is None or factor == 1.0:
            logger.warning("No SUV conversion factor found or factor is 1.0. Skipping conversion.")
            return context
            
        original_image = context.input_image
        array = sitk.GetArrayFromImage(original_image) * factor

        context.input_array = array
        context.metadata['unit'] = 'SUV'

        context.input_image = sitk.GetImageFromArray(array.astype(np.float32))
        context.input_image.CopyInformation(original_image)
        
        return context


class SUVThresholdStep(PipelineStep):
    """
    Apply SUV threshold for PET images.
    
    This step applies a Standard Uptake Value (SUV) threshold to PET images,
    which is commonly used to identify metabolically active regions.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute SUV thresholding.
        
        Config options:
            threshold: SUV threshold value (default: 2.5)
        """
        threshold = self.config.get('threshold', 2.5)
        
        logger.debug(f"Applying SUV threshold: {threshold}")
        
        original_image = context.input_image
        array = sitk.GetArrayFromImage(original_image)
        mask = array > threshold
        array = array * mask

        context.input_array = array
        context.intermediate_results['suv_mask'] = mask

        context.input_image = sitk.GetImageFromArray(array.astype(np.float32))
        context.input_image.CopyInformation(original_image)
        
        return context


class MultiChannelStackStep(PipelineStep):
    """
    Stack multiple modalities into channels.
    
    This step combines multiple modalities (e.g., PET and CT) into a
    multi-channel input suitable for multi-modal nnUNet models.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute multi-channel stacking.
        
        Config options:
            images: List of image keys in intermediate_results
            order: Order of modalities for stacking (default: ['pet', 'ct'])
        """
        images = self.config.get('images', [])
        order = self.config.get('order', ['pet', 'ct'])
        
        logger.debug(f"Stacking modalities: {order}")
        
        stacked = np.stack([
            context.intermediate_results[f'{mod}_array']
            for mod in order
        ], axis=0)
        
        context.input_array = stacked
        context.metadata['modalities'] = order
        context.metadata['num_channels'] = len(order)
        
        return context


class WindowLevelStep(PipelineStep):
    """
    Apply window/level adjustment for CT images.
    
    This step applies a window/level transformation commonly used in
    radiology to enhance specific tissue types.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute window/level adjustment.
        
        Config options:
            window: Window width (default: 400)
            level: Window level/center (default: 40)
        """
        window = self.config.get('window', 400)
        level = self.config.get('level', 40)
        
        logger.debug(f"Applying window/level: {window}/{level}")
        
        original_image = context.input_image
        array = sitk.GetArrayFromImage(original_image)

        min_val = level - window / 2
        max_val = level + window / 2

        array = np.clip(array, min_val, max_val)
        array = (array - min_val) / (max_val - min_val)

        context.input_array = array

        context.input_image = sitk.GetImageFromArray(array.astype(np.float32))
        context.input_image.CopyInformation(original_image)
        
        return context


class CropToContentStep(PipelineStep):
    """
    Crop image to content region.
    
    This step crops the image to remove empty borders, which can reduce
    computation time and memory usage.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute content cropping.
        
        Config options:
            threshold: Threshold for identifying content (default: 0)
            margin: Margin to add around content (default: 10)
        """
        threshold = self.config.get('threshold', 0)
        margin = self.config.get('margin', 10)
        
        logger.debug(f"Cropping to content with threshold {threshold}")
        
        array = sitk.GetArrayFromImage(context.input_image)
        
        # Find content bounds
        content_mask = array > threshold
        
        # Find bounding box
        coords = np.where(content_mask)
        if len(coords[0]) == 0:
            # No content found
            return context
        
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        z_min, z_max = coords[2].min(), coords[2].max()
        
        # Add margin
        x_min = max(0, x_min - margin)
        x_max = min(array.shape[0] - 1, x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(array.shape[1] - 1, y_max + margin)
        z_min = max(0, z_min - margin)
        z_max = min(array.shape[2] - 1, z_max + margin)
        
        # Crop
        cropped_array = array[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        
        # Store crop bounds for later use
        context.metadata['crop_bounds'] = {
            'x': (x_min, x_max),
            'y': (y_min, y_max),
            'z': (z_min, z_max),
        }
        
        context.input_array = cropped_array
        
        # Update image
        context.input_image = sitk.GetImageFromArray(cropped_array)
        
        return context


class ResampleNibabelStep(PipelineStep):
    """
    Resample NIfTI image using nibabel.processing functions.
    
    This step uses nibabel's resampling functions for accurate and efficient
    resampling of NIfTI images. It supports both resampling to a target spacing
    and resampling to match a reference image geometry.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute nibabel-based resampling.
        
        Config options:
            mode: Resampling mode
                - 'to_spacing': Resample to target voxel spacing
                - 'to_reference': Resample to match reference image geometry
            target_spacing: Target voxel spacing (for 'to_spacing' mode)
                - Single float: Isotropic spacing (e.g., 1.5)
                - Tuple: Anisotropic spacing (e.g., (1.0, 1.0, 2.0))
            reference_image: Path to reference image (for 'to_reference' mode)
            interpolation: Interpolation type
                - 'nearest': Nearest neighbor (for segmentations)
                - 'linear': Linear interpolation (default)
                - 'cubic': Cubic interpolation
        """
        import nibabel as nib
        from nibabel.processing import resample_from_to, resample_to_output
        
        mode = self.config.get('mode', 'to_spacing')
        interpolation = self.config.get('interpolation', 'linear')
        
        # Map interpolation to order
        interp_order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
        order = interp_order_map.get(interpolation, 1)
        
        # Convert internal image (X,Y,Z) to nibabel image (X,Y,Z).
        if hasattr(context.input_image, 'to_nifti'):
            nifti_img = context.input_image.to_nifti()
        else:
            # Already a NIfTI image
            nifti_img = context.input_image
        
        original_spacing = nifti_img.header.get_zooms()[:3]
        logger.debug(f"Original spacing: {original_spacing}")
        
        if mode == 'to_spacing':
            # Resample to target spacing
            target_spacing = self.config.get('target_spacing', (1.5, 1.5, 1.5))
            
            logger.debug(f"Resampling to spacing: {target_spacing}")
            
            resampled = resample_to_output(
                nifti_img,
                target_spacing,
                order=order
            )
            
        elif mode == 'to_reference':
            # Resample to match reference image
            reference_path = self.config.get('reference_image')
            if reference_path is None:
                raise ValueError("reference_image required for 'to_reference' mode")
            
            reference = nib.load(reference_path)
            
            logger.debug(f"Resampling to match reference: {reference_path}")
            
            resampled = resample_from_to(
                nifti_img,
                reference,
                order=order
            )
        else:
            raise ValueError(f"Unknown resampling mode: {mode}")
        
        # Store metadata
        new_spacing = resampled.header.get_zooms()[:3]
        context.metadata['original_spacing'] = original_spacing
        context.metadata['current_spacing'] = new_spacing
        context.metadata['resampling_mode'] = mode
        
        # Convert nibabel image (X,Y,Z) back to internal image (X,Y,Z).
        resampled_image = sitk.Image.from_nifti(resampled)
        context.input_image = resampled_image
        context.input_array = sitk.GetArrayFromImage(resampled_image)
        
        logger.info(f"Resampled from {original_spacing} to {new_spacing}")
        
        return context


class ProjectionStep(PipelineStep):
    """
    Create 2D projections from 3D images for 2D-based segmentation.
    
    This step creates coronal, sagittal, or axial projections using
    maximum and/or average intensity projection, suitable for 2D U-Net models.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute projection.
        
        Config options:
            method: Projection method
                - 'coronal': Coronal projection (default, along y-axis)
                - 'sagittal': Sagittal projection (along x-axis)
                - 'axial': Axial projection (along z-axis)
            channels: List of projection types to create
                - ['mip']: Maximum intensity projection only
                - ['aip']: Average intensity projection only
                - ['mip', 'aip']: Both MIP and AIP (default, 2-channel)
            axis: Manual axis specification (overrides method)
        """
        from ...utils.projection import create_coronal_projection, create_multichannel_projection
        
        method = self.config.get('method', 'coronal')
        channels = self.config.get('channels', ['mip', 'aip'])
        axis = self.config.get('axis')
        
        # Map method to axis
        if axis is None:
            method_to_axis = {
                'coronal': 1,
                'sagittal': 0,
                'axial': 2
            }
            axis = method_to_axis.get(method, 1)
        
        logger.debug(f"Creating {method} projection along axis {axis}")
        logger.debug(f"Projection channels: {channels}")
        
        # Get input image
        image = context.input_image
        original_size = image.GetSize()
        
        # Map channel names to projection types
        channel_map = {
            'mip': 'maximum',
            'aip': 'average',
            'minip': 'minimum'
        }
        
        projection_types = [channel_map.get(ch, ch) for ch in channels]
        
        # Create multi-channel projection
        projection_array = create_multichannel_projection(
            image,
            projections=projection_types,
            axis=axis
        )
        
        logger.info(f"Created {len(channels)}-channel projection: {projection_array.shape}")
        
        # Store projection in context
        context.input_array = projection_array
        context.metadata['projection_method'] = method
        context.metadata['projection_axis'] = axis
        context.metadata['projection_channels'] = channels
        context.metadata['original_3d_size'] = original_size
        
        # Store individual projections for potential later use
        for i, ch in enumerate(channels):
            context.intermediate_results[f'projection_{ch}'] = projection_array[i]
        
        return context
