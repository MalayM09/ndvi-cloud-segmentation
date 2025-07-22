import numpy as np
from skimage import exposure

class CrossSensorNDVIPreprocessor:
    def clean_invalid_values(self, ndvi_array):
        ndvi_array = np.clip(ndvi_array, -1.0, 1.0)
        ndvi_array[np.isnan(ndvi_array)] = 0.0
        return ndvi_array

    def apply_cross_sensor_augmentation(self, ndvi_array, apply_augmentation=True):
        if apply_augmentation:
            p = np.random.rand()
            if p < 0.25:
                ndvi_array = np.fliplr(ndvi_array)
            elif p < 0.5:
                ndvi_array = np.flipud(ndvi_array)
            elif p < 0.75:
                ndvi_array = np.rot90(ndvi_array)
        return ndvi_array

    def normalize_for_cross_sensor(self, ndvi_array):
        ndvi_min, ndvi_max = -1.0, 1.0
        normalized = (ndvi_array - ndvi_min) / (ndvi_max - ndvi_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized

    def convert_to_sam_format(self, normalized_array):
        sam_input = np.stack([normalized_array] * 3, axis=0).astype(np.float32)
        return sam_input

    def preprocess_single_image_array(self, ndvi_array, apply_augmentation=True):
        cleaned = self.clean_invalid_values(ndvi_array)
        augmented = self.apply_cross_sensor_augmentation(cleaned, apply_augmentation)
        normalized = self.normalize_for_cross_sensor(augmented)
        sam_input = self.convert_to_sam_format(normalized)

        return {
            'cleaned': cleaned,
            'augmented': augmented,
            'normalized': normalized,
            'sam_input': sam_input
        }
