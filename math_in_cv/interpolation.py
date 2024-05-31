import numpy as np
import cv2
from PIL import Image

def nearest_neighbor_interpolation(image, new_width, new_height):
    """
    Perform nearest neighbor interpolation to resize an image.

    Args:
        image (numpy.ndarray): Input image as a numpy array.
        new_width (int): Desired width of the output image.
        new_height (int): Desired height of the output image.

    Returns:
        numpy.ndarray: Resized image.
    """
    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            # Find the nearest neighbor in the original image
            x = int(j * original_width / new_width)
            y = int(i * original_height / new_height)
            resized_image[i, j] = image[y, x]

    return resized_image


def bilinear_interpolation(image, new_width, new_height):
    """
    Perform bilinear interpolation to resize an image.

    Args:
        image (numpy.ndarray): Input image as a numpy array.
        new_width (int): Desired width of the output image.
        new_height (int): Desired height of the output image.

    Returns:
        numpy.ndarray: Resized image.
    """
    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
    # Calculate scale factors
    scale_x = original_width / new_width
    scale_y = original_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            # Find the coordinates of the four surrounding pixels
            x = j * scale_x
            y = i * scale_y

            x0 = int(x)
            x1 = min(x0 + 1, original_width - 1)
            y0 = int(y)
            y1 = min(y0 + 1, original_height - 1)

            # Calculate the differences
            dx = x - x0
            dy = y - y0

            # Get the pixel values of the four surrounding pixels
            top_left = image[y0, x0]
            top_right = image[y0, x1]
            bottom_left = image[y1, x0]
            bottom_right = image[y1, x1]

            # Calculate the interpolated pixel value
            top = top_left * (1 - dx) + top_right * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx
            pixel_value = top * (1 - dy) + bottom * dy

            resized_image[i, j] = pixel_value

    return resized_image


def bicubic_interpolation(image, new_width, new_height):
    """
    Perform bicubic interpolation to resize an image.

    Args:
        image (numpy.ndarray): Input image as a numpy array.
        new_width (int): Desired width of the output image.
        new_height (int): Desired height of the output image.

    Returns:
        numpy.ndarray: Resized image.
    """
    def cubic(x, a=-0.5):
        if abs(x) <= 1:
            return (a + 2) * abs(x)**3 - (a + 3) * abs(x)**2 + 1
        elif 1 < abs(x) < 2:
            return a * abs(x)**3 - 5 * a * abs(x)**2 + 8 * a * abs(x) - 4 * a
        return 0

    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

    scale_x = original_width / new_width
    scale_y = original_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = j * scale_x
            y = i * scale_y

            x0 = int(x)
            y0 = int(y)

            pixel_value = 0
            for m in range(x0 - 1, x0 + 3):
                for n in range(y0 - 1, y0 + 3):
                    if 0 <= m < original_width and 0 <= n < original_height:
                        weight = cubic(x - m) * cubic(y - n)
                        pixel_value += image[n, m] * weight

            resized_image[i, j] = pixel_value

    return resized_image

#Lanczos Interpolation
def lanczos_interpolation(image, new_width, new_height, a=2):
    """
    Perform Lanczos interpolation to resize an image.

    Args:
        image (numpy.ndarray): Input image as a numpy array.
        new_width (int): Desired width of the output image.
        new_height (int): Desired height of the output image.
        a (int): Lanczos interpolation parameter.

    Returns:
        numpy.ndarray: Resized image.
    """
    def sinc(x):
        if x == 0:
            return 1
        return np.sin(np.pi * x) / (np.pi * x)

    def lanczos_filter(x, a):
        if -a <= x <= a:
            return sinc(x) * sinc(x / a)
        return 0

    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

    scale_x = original_width / new_width
    scale_y = original_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = j * scale_x
            y = i * scale_y

            x0 = int(x)
            y0 = int(y)

            pixel_value = 0
            total_weight = 0

            for m in range(x0 - a + 1, x0 + a):
                for n in range(y0 - a + 1, y0 + a):
                    if 0 <= m < original_width and 0 <= n < original_height:
                        weight = lanczos_filter(x - m, a) * lanczos_filter(y - n, a)
                        pixel_value += image[n, m] * weight
                        total_weight += weight

            resized_image[i, j] = pixel_value / total_weight

    return resized_image


def spline_interpolation(image, new_width, new_height, a=2):
    """
    Perform spline interpolation to resize an image.

    Args:
        image (numpy.ndarray): Input image as a numpy array.
        new_width (int): Desired width of the output image.
        new_height (int): Desired height of the output image.
        a (int): Spline interpolation parameter.

    Returns:
        numpy.ndarray: Resized image.
    """
    def spline_filter(x):
        if abs(x) <= 1:
            return 2 / 3 - abs(x)**2 + 0.5 * abs(x)**3
        elif 1 < abs(x) < 2:
            return (2 - abs(x))**3 / 6
        return 0

    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

    scale_x = original_width / new_width
    scale_y = original_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = j * scale_x
            y = i * scale_y

            x0 = int(x)
            y0 = int(y)

            pixel_value = 0
            total_weight = 0

            for m in range(x0 - a + 1, x0 + a):
                for n in range(y0 - a + 1, y0 + a):
                    if 0 <= m < original_width and 0 <= n < original_height:
                        weight = spline_filter(x - m) * spline_filter(y - n)
                        pixel_value += image[n, m] * weight
                        total_weight += weight

            resized_image[i, j] = pixel_value / total_weight

    return resized_image

# Load the image
image_path = './bottle.jpg'
image = np.array(Image.open(image_path))

# Define new dimensions
new_width = 1500
new_height = 1500

# Perform nearest neighbor interpolation
resized_image = nearest_neighbor_interpolation(image, new_width, new_height)

# Save or display the resized image
resized_image_pil = Image.fromarray(resized_image)
resized_image_pil.show()
# or save the resized image
resized_image_pil.save('./resized_image.jpg')
