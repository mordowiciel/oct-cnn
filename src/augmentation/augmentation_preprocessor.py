import random

import cv2
import numpy as np
from PIL import Image
from PIL.ImageEnhance import Contrast
from skimage.util import random_noise


class AugmentationPreprocessor:

    def __init__(self, augmentation_config, preprocessing_functions):
        self.augmentation_config = augmentation_config
        self.preprocessing_functions = preprocessing_functions
        self.preprocessing_functions_ref_mapping = {
            'gaussian_noise': self.gaussian_noise,
            'contrast': self.contrast
        }

    def resolve_preprocessing_function_ref(self, function_name):
        return self.preprocessing_functions_ref_mapping[function_name]

    # When using debug function, the image_arr already has the shape (x,y,3), so there is no need
    # to perform image_arr.reshape() on provided array.
    def gaussian_noise(self, image_arr):
        random_var = random.uniform(self.augmentation_config.gaussian_noise_var_range[0],
                                    self.augmentation_config.gaussian_noise_var_range[1])
        image_with_noise = random_noise(image_arr.astype(np.uint8), mode='gaussian', var=random_var)

        # Normalize image back from [-1,1] to [0, 255]
        normalized = cv2.normalize(image_with_noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
        xd = normalized.reshape(normalized.shape + (1,))
        return xd

    def contrast(self, image_arr):
        random_contrast = random.uniform(self.augmentation_config.contrast_range[0],
                                         self.augmentation_config.contrast_range[1])

        # Image.fromarray() handles only (x,y) as grayscale shapes, so flatten the array
        image_arr = image_arr.reshape(image_arr.shape[0], image_arr.shape[1])

        image_container = Image.fromarray(image_arr.astype(np.uint8))
        enhanced_image_arr = np.array(Contrast(image_container).enhance(random_contrast))

        # Reshape array again to Keras grayscale format (x, y, 1)
        enhanced_image_arr = enhanced_image_arr.reshape(enhanced_image_arr.shape + (1,))

        return enhanced_image_arr.astype(np.float32)

    def preprocessing_chain(self, image):
        previous_image = image
        for function in self.preprocessing_functions:
            function_ref = self.resolve_preprocessing_function_ref(function)
            previous_image = function_ref(previous_image)
        return previous_image
