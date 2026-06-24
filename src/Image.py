# Image: pure data object that loads a file path into a numpy array
# and can save it back out. Holds no transformation logic.
#
# ImageOperators: transformations that take/return a raw array, so they
# stay decoupled from Image and compose via Image.apply().
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
            return np.asarray(img.convert('L'), dtype=np.float32) / 255.0

    def save(self, output_path: str):
        array_uint8 = np.clip(self.array * 255.0, 0, 255).astype(np.uint8)
        PILImage.fromarray(array_uint8).save(output_path)

    def apply(self, operator, *args, **kwargs) -> "Image":
        self.array = operator(self.array, *args, **kwargs)
        return self


class ImageOperators:
    @staticmethod
    def gaussian_kernel_1d(sigma: float, ksize: int = None) -> np.ndarray:
        # Same construction as the regularization kernel K in the solver
        # normalized 1D Gaussian.
        if ksize is None:
            ksize = int(6 * sigma) | 1
        ax = np.arange(ksize, dtype=np.float32) - ksize // 2
        kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    @staticmethod
    def gaussian_smoothing(array: np.ndarray, sigma: float) -> np.ndarray:
        # Separable convolution along height then width with reflect
        # padding, mirroring the solver's make_smoother.
        from scipy.ndimage import convolve1d
        kernel = ImageOperators.gaussian_kernel_1d(sigma)
        smoothed = convolve1d(array, kernel, axis=1, mode='reflect')
        smoothed = convolve1d(smoothed, kernel, axis=0, mode='reflect')
        return smoothed

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

