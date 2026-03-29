import numpy as np
from ... import image as sitk
from ..base import PipelineStep, PipelineContext
from scipy.ndimage import rotate
import imageio
import logging

logger = logging.getLogger(__name__)

class MIPGenerationStep(PipelineStep):
    """
    Step for generating Maximum Intensity Projection (MIP) GIFs.
    Rotates the image and segmentation to create a 3D visualization.
    """
    
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.output_spacing = self.config.get('output_spacing', (4, 4, 4))
        self.rotation_step = self.config.get('rotation_step', 40)
        self.frame_duration = self.config.get('frame_duration', 0.4)
        self.output_filename = self.config.get('output_filename', 'mip.gif')
        self.mask_overlay = self.config.get('mask_overlay', True)

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Generate MIP GIF from image and segmentation in context.
        """
        if not context.input_image:
            logger.warning("No input_image found for %s, skipping MIP generation.", self.name)
            return context
            
        pet_img = context.input_image
        # Get segmentation from intermediate results
        mask_array = context.intermediate_results.get('refined_prediction', 
                                                     context.intermediate_results.get('raw_prediction'))
        
        if mask_array is None and self.mask_overlay:
            logger.warning("No segmentation found for %s, skipping overlay.", self.name)
            
        # Resample for faster processing and standard viewing
        pet_resampled = self._resample_image(pet_img, self.output_spacing, is_label=False)
        
        mask_resampled = None
        if mask_array is not None and self.mask_overlay:
            # We need to convert mask_array back to SimpleITK image to resample it with same geometry as pet_img
            # mask_array is likely just array, we need to attach geometry from pet_img (or original if cropped)
            # Assuming mask_array matches pet_img geometry currently in context.
            mask_img = sitk.GetImageFromArray(mask_array)
            mask_img.CopyInformation(pet_img)
            mask_resampled = self._resample_image(mask_img, self.output_spacing, is_label=True)
            
        # Convert to numpy
        pet_array = sitk.GetArrayFromImage(pet_resampled)
        mask_array_res = sitk.GetArrayFromImage(mask_resampled) if mask_resampled else None
        
        # Normalize PET
        pet_array = self._normalize_img(pet_array)
        
        # Create MIPs
        mip_images = self._create_rotational_mip(pet_array, mask_array_res)
        
        # Save GIF
        # We need an output path. PipelineContext might have output_dir or we infer from somewhere.
        # For now, let's assume context.metadata has 'output_path' or we save alongside input.
        # If context has no output path, we might skip or save to current dir.
        
        # Actually, base LION code uses 'output_manager' to save.
        # Here we should probably store the GIF in context or write it to disk if path is known.
        # Let's check context.metadata for output directory.
        output_dir = context.metadata.get('output_dir', '.')
        case_id = context.metadata.get('case_id', 'unknown_case')
        gif_path = f"{output_dir}/{case_id}_{self.output_filename}"
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        imageio.mimsave(gif_path, mip_images, duration=self.frame_duration)
        
        # Store path in metadata for reference
        context.metadata['mip_path'] = gif_path
        
        return context

    def _resample_image(self, image: sitk.Image, output_spacing: tuple, is_label: bool) -> sitk.Image:
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        
        new_size = [
            int(round(osz * osp / nsp))
            for osz, osp, nsp in zip(original_size, original_spacing, output_spacing)
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            
        return resampler.Execute(image)

    def _normalize_img(self, img: np.ndarray) -> np.ndarray:
        # Simple normalization to max
        if np.max(img) > 0:
            img = img / np.max(img)
        return img
        
    def _mip_3d(self, img: np.ndarray, angle: float) -> np.ndarray:
        # Rotate
        rot_img = rotate(img, angle, axes=(1, 2), reshape=False)
        # MIP along axis 1 after rotating in the YZ plane.
        # Arrays in this project use nibabel/native order (X, Y, Z).
        # Rotate on (1, 2) rotates Y and X (axial plane).
        # MIP along axis 1 (Y) creates a coronal view?
        # LION code:
        # rot_img = rotate(img, angle, axes=(1, 2), reshape=False)
        # mip = np.max(rot_img, axis=1)
        # Invert: mip_inverted = np.max(mip) - mip
        # Flip: mip_flipped = np.flip(mip_inverted, axis=0)
        
        mip = np.max(rot_img, axis=1)
        mip_inverted = np.max(mip) - mip
        mip_flipped = np.flip(mip_inverted, axis=0)
        return mip_flipped

    def _create_rotational_mip(self, pet_array: np.ndarray, mask_array: np.ndarray = None) -> list:
        # Create color versions
        pet_color = np.stack((pet_array, pet_array, pet_array), axis=-1)
        mask_color = None
        if mask_array is not None:
            # Purple: (0.5, 0, 0.5)
            mask_color = np.stack((0.5 * mask_array, np.zeros_like(mask_array), 0.5 * mask_array), axis=-1)
            
        angles = list(range(0, 360, self.rotation_step))
        mip_images = []
        
        for angle in angles:
            pet_mip = self._mip_3d(pet_color, angle)
            if mask_color is not None:
                mask_mip = self._mip_3d(mask_color, angle)
                blended = (pet_mip * 0.7) + (mask_mip.astype(pet_mip.dtype) * 0.3)
                mip_images.append(blended)
            else:
                mip_images.append(pet_mip)
                
        # Normalize to 0-255 uint8
        final_images = []
        for im in mip_images:
            if np.max(im) > np.min(im):
                norm = (255 * (im - np.min(im)) / (np.max(im) - np.min(im))).astype(np.uint8)
            else:
                norm = np.zeros_like(im, dtype=np.uint8)
            final_images.append(norm)
            
        return final_images
