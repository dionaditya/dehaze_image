import numpy as np
import cv2

def get_haze_density_prior(image):
    """
    Compute haze density prior (HDP) based on the difference between maximum and minimum color channels.
    Args:
        image (numpy.ndarray): Hazy input image.
    Returns:
        f (numpy.ndarray): Haze density prior.
        Ic_min (numpy.ndarray): Minimum channel of the input image.
        Ic_max (numpy.ndarray): Maximum channel of the input image.
    """
    Ic_max = np.max(image, axis=2)  # Maximum channel
    Ic_min = np.min(image, axis=2)  # Minimum channel
    f = Ic_max - Ic_min             # Channel difference as haze density prior
    return f, Ic_min, Ic_max

def estimate_atmospheric_light_veil(Ic_min, f):
    """
    Estimate the atmospheric light veil using the haze density prior.
    Args:
        Ic_min (numpy.ndarray): Minimum channel of the input image.
        f (numpy.ndarray): Haze density prior.
    Returns:
        V (numpy.ndarray): Atmospheric light veil.
    """
    gamma = np.mean(1 - f)          # Adaptive parameter based on haze density prior
    V = Ic_min / gamma              # Atmospheric light veil
    return V

def estimate_mid_channel(Ic_min, Ic_max):
    """
    Estimate the mid-channel of the image for atmospheric light estimation.
    Args:
        Ic_min (numpy.ndarray): Minimum channel of the input image.
        Ic_max (numpy.ndarray): Maximum channel of the input image.
    Returns:
        Imid (numpy.ndarray): Mid-channel.
    """
    Imid = np.sqrt(Ic_max * Ic_min)  # Geometric mean for mid-channel estimation
    return Imid

def bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Apply bilateral filtering for atmospheric light estimation.
    Args:
        image (numpy.ndarray): Input image.
    Returns:
        filtered_image (numpy.ndarray): Bilaterally filtered image.
    """
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def estimate_atmospheric_light(Imid):
    """
    Estimate the atmospheric light using bilateral filtering and mid-channel.
    Args:
        Imid (numpy.ndarray): Mid-channel of the input image.
    Returns:
        A (numpy.ndarray): Atmospheric light.
    """
    # Ensure no NaN or Inf values in Imid
    Imid = np.nan_to_num(Imid, nan=0.0, posinf=255, neginf=0)

    # Check if Imid contains valid values (non-zero and non-empty)
    if np.all(Imid == 0):
        raise ValueError("Imid contains all zero values, cannot proceed with normalization.")
    elif Imid.size == 0:
        raise ValueError("Imid is empty, cannot proceed with normalization.")

    # Ensure Imid is in uint8 format before applying morphology operations
    Imid_normalized = cv2.normalize(Imid, None, 0, 255, cv2.NORM_MINMAX)
    Imid_uint8 = np.uint8(Imid_normalized)

    # Apply the morphological closing operation
    Imid_closed = cv2.morphologyEx(Imid_uint8, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Apply bilateral filtering
    A = bilateral_filter(Imid_closed)
    return A

def dehaze_image(image, V, A, omega=0.95):
    """
    Perform dehazing based on the estimated atmospheric light and veil.
    Args:
        image (numpy.ndarray): Hazy input image.
        V (numpy.ndarray): Atmospheric light veil.
        A (numpy.ndarray): Atmospheric light.
        omega (float): Dehazing parameter for controlling haze removal strength.
    Returns:
        J (numpy.ndarray): Dehazed image.
    """
    J = (A * (image - omega * V[..., np.newaxis])) / (A - omega * V[..., np.newaxis])
    J = np.clip(J, 0, 255).astype(np.uint8)  # Ensure pixel values are valid
    return J

def dehaze(image):
    """
    Full pipeline for dehazing using the Haze Density Prior method.
    Args:
        image (numpy.ndarray): Hazy input image.
    Returns:
        J (numpy.ndarray): Dehazed image.
    """
    f, Ic_min, Ic_max = get_haze_density_prior(image)
    V = estimate_atmospheric_light_veil(Ic_min, f)
    Imid = estimate_mid_channel(Ic_min, Ic_max)
    A = estimate_atmospheric_light(Imid)
    J = dehaze_image(image, V, A)
    return J


image = cv2.imread('./dataset/test/hazy/images.png')
print(image)
dehazed_image = dehaze(image)
cv2.imwrite('dehazed_image.jpg', dehazed_image)