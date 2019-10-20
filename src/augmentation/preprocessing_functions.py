import random

import cv2
import numpy as np
from PIL import Image
from PIL.ImageEnhance import Contrast
from skimage.util import random_noise


# When using debug function, the image_arr already has the shape (x,y,3), so there is no need
# to perform image_arr.reshape() on provided array.
def gaussian_noise(image_arr):
    random_var = random.uniform(0.00, 0.0)
    image_with_noise = random_noise(image_arr.astype(np.uint8), mode='gaussian', var=random_var)

    # Normalize image back from [-1,1] to [0, 255]
    normalized = cv2.normalize(image_with_noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    xd = normalized.reshape(normalized.shape + (1,))
    return xd


def contrast(image_arr):
    random_contrast = random.uniform(0.35, 1.0)

    # Image.fromarray() handles only (x,y) as grayscale shapes, so flatten the array
    image_arr = image_arr.reshape(image_arr.shape[0], image_arr.shape[1])

    image_container = Image.fromarray(image_arr.astype(np.uint8))
    enhanced_image_arr = np.array(Contrast(image_container).enhance(random_contrast))

    # Reshape array again to Keras grayscale format (x, y, 1)
    enhanced_image_arr = enhanced_image_arr.reshape(enhanced_image_arr.shape + (1,))

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
