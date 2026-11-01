"""Training-only CIFAR corruption operators adapted from Hendrycks & Dietterich.

Apache-2.0 adaptations of make_cifar_c.py at commit
8190fe329f5f072a06e3a2aea02bb5dda69aed9f; see third_party/NOTICE.md and license.
MODIFIED: explicit RNG injection, checked PIL interface, NumPy 2 dtype alias,
uint8 outputs, worker-safe exposure metadata. No benchmark dataset access.
"""
from io import BytesIO
from functools import lru_cache
import random
import numpy as np
from PIL import Image as PILImage, features, __version__ as pillow_version
from torchvision import transforms
from .lofo import CORRUPTION_NAMES
try:
    import cv2
except ImportError:
    cv2 = None

UPSTREAM_COMMIT = "8190fe329f5f072a06e3a2aea02bb5dda69aed9f"


@lru_cache(maxsize=8)
def _disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def _plasma_fractal(mapsize=32, wibbledecay=3, rng=None):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def _gaussian_noise(x, severity=1, rng=None):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255


def _defocus_blur(x, severity=1, rng=None):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    x = np.array(x) / 255.
    kernel = _disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1) * 255


def _fog(x, severity=1, rng=None):
    c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * _plasma_fractal(wibbledecay=c[1], rng=rng)[:32, :32][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def _jpeg_compression(x, severity=1, rng=None):
    c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


OPERATORS = {"gaussian_noise": _gaussian_noise, "defocus_blur": _defocus_blur,
             "fog": _fog, "jpeg_compression": _jpeg_compression}


def require_opencv():
    if cv2 is None:
        raise RuntimeError("Defocus blur requires OpenCV. Install opencv-python-headless "
                           "(requirements-lofo.txt), or use Colab's compatible preinstalled cv2.")
    cv2.setNumThreads(1)  # Avoid worker-by-OpenCV thread oversubscription.


def implementation_metadata():
    return {"source_commit": UPSTREAM_COMMIT, "operator_set": list(OPERATORS),
            "opencv": None if cv2 is None else cv2.__version__,
            "pillow": pillow_version, "jpeg_codec": features.version_codec("jpg"),
            "numpy": np.__version__, "serialization": "clip_then_truncate_uint8_RGB",
            "identity_claim": "CIFAR-reference operators; not released benchmark image identity"}


def apply_corruption(image, name, severity, rng=None):
    if name not in OPERATORS:
        raise ValueError(f"Unsupported training corruption: {name!r}.")
    if isinstance(severity, bool) or not isinstance(severity, (int, np.integer)) or not 1 <= severity <= 5:
        raise ValueError("Severity must be an integer from 1 through 5.")
    if not isinstance(image, PILImage.Image) or image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("Corruptions require a 32×32 RGB PIL image before tensor conversion.")
    if name == "defocus_blur":
        require_opencv()
    rng = np.random if rng is None else rng
    result = OPERATORS[name](image, severity, rng=rng)
    # Same quantization used by the original benchmark's np.uint8 serialization.
    pixels = np.asarray(result)
    if pixels.shape != (32, 32, 3) or not np.isfinite(pixels).all():
        raise ValueError(f"{name} produced invalid image pixels.")
    return PILImage.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def gaussian_noise(image, severity=1, rng=None):
    return apply_corruption(image, "gaussian_noise", severity, rng)


def defocus_blur(image, severity=1, rng=None):
    return apply_corruption(image, "defocus_blur", severity, rng)


def fog(image, severity=1, rng=None):
    return apply_corruption(image, "fog", severity, rng)


def jpeg_compression(image, severity=1, rng=None):
    return apply_corruption(image, "jpeg_compression", severity, rng)


class LOFOTrainingTransform:
    """One spatial augmentation, then zero/one corruption, then tensor+normalize.

    NumPy's worker-local RNG handles sampling. Python seeds a fresh operator RNG
    for each corrupted image; operator draw counts cannot alter future selections.
    Torch handles crop/flip, independently of both corruption streams.
    Main-process RNG and loader worker seeds are already checkpointed by Part 1.
    """

    def __init__(self, config):
        config.validate()
        self.allowed = tuple(config.allowed_corruptions)
        self.probability = config.corruption_probability
        self.severities = tuple(config.allowed_severities)
        if "defocus_blur" in self.allowed:
            require_opencv()
        self.spatial = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        self.tensor = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(config.mean, config.std)])

    def sample(self, rng=None):
        rng = np.random if rng is None else rng
        if rng.random() >= self.probability:
            return "clean", 0
        return str(rng.choice(self.allowed)), int(rng.choice(self.severities))

    def __call__(self, image):
        image = self.spatial(image)
        name, severity = self.sample()
        if name != "clean":
            operator_rng = np.random.RandomState(random.getrandbits(32))
            image = apply_corruption(image, name, severity, rng=operator_rng)
        return self.tensor(image), CORRUPTION_NAMES.index(name), severity
