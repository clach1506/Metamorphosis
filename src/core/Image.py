# Image: lightweight loader that returns a grayscale image as a float32 [0,1] array.

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
        with PILImage.open(file_path) as img:
            if img.mode.startswith("I;16"):
                # 16-bit grayscale: PIL's convert("L") clips (not rescales)
                # values above 255, so normalize by the 16-bit range directly.
                return np.asarray(img, dtype=np.float32) / 65535.0
            arr = np.asarray(img.convert("L"), dtype=np.float32)
        # 8-bit images are scaled to [0, 1]. A label mask stored as raw 0/1
        # values is already in [0, 1] and is kept as-is.
        return arr / 255.0 if arr.max() > 1.0 else arr
