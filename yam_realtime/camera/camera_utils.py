"""
Camera utility functions for processing observation data.
"""

from typing import Any, Dict, Optional
import numpy as np
import cv2


def obs_get_rgb(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Recursively search through observation dictionary to find RGB images.
    
    Args:
        obs: Observation dictionary that may contain nested camera data
        
    Returns:
        Dictionary mapping camera names to RGB image arrays
    """
    rgb_dict = {}
    
    for key, value in obs.items():
        if isinstance(value, dict):
            # Check if this dict contains images with rgb data
            if 'images' in value and isinstance(value['images'], dict):
                if 'rgb' in value['images']:
                    rgb_dict[key] = value['images']['rgb']
            else:
                # Recursively search in nested dictionaries
                nested_rgb = obs_get_rgb(value)
                rgb_dict.update(nested_rgb)
    
    return rgb_dict


def obs_get_camera_data(obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract all camera data (not just RGB) from observation dictionary.
    
    Args:
        obs: Observation dictionary that may contain nested camera data
        
    Returns:
        Dictionary mapping camera names to their full camera data
    """
    camera_dict = {}
    
    for key, value in obs.items():
        if isinstance(value, dict):
            # Check if this looks like camera data (has images and timestamp)
            if 'images' in value and 'timestamp' in value:
                camera_dict[key] = value
            else:
                # Recursively search in nested dictionaries
                nested_cameras = obs_get_camera_data(value)
                camera_dict.update(nested_cameras)
    
    return camera_dict


def obs_has_cameras(obs: Dict[str, Any]) -> bool:
    """
    Check if observation dictionary contains any camera data.
    
    Args:
        obs: Observation dictionary
        
    Returns:
        True if cameras are found, False otherwise
    """
    return len(obs_get_camera_data(obs)) > 0


def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resizes an image to a target height and width without distortion by padding with black.
    
    Args:
        images: Input image(s) with shape (h, w, c) or (b, h, w, c)
        height: Target height
        width: Target width  
        interpolation: OpenCV interpolation method (default: cv2.INTER_LINEAR)
        
    Returns:
        Resized and padded image(s) with shape (height, width, c) or (b, height, width, c)
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # Add batch dimension
        
    batch_size, cur_height, cur_width, channels = images.shape
    
    # Calculate scaling ratio to maintain aspect ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    # Process each image in the batch
    resized_images = np.zeros((batch_size, resized_height, resized_width, channels), dtype=images.dtype)
    
    for i in range(batch_size):
        resized_images[i] = cv2.resize(
            images[i], 
            (resized_width, resized_height), 
            interpolation=interpolation
        )
    
    # Calculate padding amounts
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    
    # Determine padding value based on dtype
    if images.dtype == np.uint8:
        pad_value = 0
    elif images.dtype == np.float32:
        pad_value = -1.0
    else:
        pad_value = 0
    
    # Apply padding
    padded_images = np.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode='constant',
        constant_values=pad_value
    )
    
    # Remove batch dimension if it wasn't in the input
    if not has_batch_dim:
        padded_images = padded_images[0]
        
    return padded_images