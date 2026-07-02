# Image: lightweight loader that returns a grayscale image as a float32 [0,1] array.
# ImageOperators: transformations that take/return a raw array.

import numpy as np
from PIL import Image as PILImage


class Image:
    file_path: str
    array: np.ndarray

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.array = self._load(file_path)

    @staticmethod
    def _load(file_path: str) -> np.ndarray:
        # Grayscale, normalized to float32 in [0, 1]
        with PILImage.open(file_path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            # ponytail: PIL scales >8-bit to 8-bit, but floats/already-normalized
            # images stay as-is. Normalize only when values are clearly integer-ish.
            return arr / 255.0 if arr.max() > 1.0 else arr


class ImageOperators:
    @staticmethod
    def gaussian_smoothing(array: np.ndarray, sigma: float) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        # ponytail: truncate=3.0 preserves the old hand-rolled kernel radius (~3σ);
        # scipy's default 4.0 changes regularization strength.
        return gaussian_filter(array, sigma=sigma, mode="reflect", truncate=3.0)

    @staticmethod
    def resize(array: np.ndarray, size) -> np.ndarray:
        # Anti-aliased resize (up- or down-sampling) to (height, width),
        # e.g. to move an image across a resolution pyramid as in the solver.
        height, width = size
        if array.shape[:2] == (height, width):
            return array
        img = PILImage.fromarray(array)
        return np.array(img.resize((width, height), resample=PILImage.LANCZOS))

    @staticmethod
    def gradient(array: np.ndarray):
        grad_x, grad_y = np.gradient(array)
        return grad_x, grad_y
