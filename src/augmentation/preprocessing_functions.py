import random

import cv2
import numpy as np
from PIL import Image
from PIL.ImageEnhance import Contrast
from skimage.util import random_noise


def gaussian_noise(image_arr):
    random_var = random.uniform(0.00, 0.05)
    image_with_noise = random_noise(image_arr.astype(np.uint8), mode='gaussian', var=random_var)

    # Normalize image back from [-1,1] to [0, 255]
    return cv2.normalize(image_with_noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def contrast(image_arr):
    random_contrast = random.uniform(0.35, 1.0)
    image_container = Image.fromarray(image_arr.astype(np.uint8))
    enhanced_image_arr = np.array(Contrast(image_container).enhance(random_contrast))
    return enhanced_image_arr.astype(np.float32)


preprocessing_functions_cfg_mapping = {
    'gaussian_noise': gaussian_noise,
    'contrast': contrast
}


def all_in_once(image):
    previous_image = image
    for function in preprocessing_functions_cfg_mapping.values():
        previous_image = function(previous_image)
    return previous_image


def resolve_preprocessing_function(function_name):
    return preprocessing_functions_cfg_mapping[function_name]
