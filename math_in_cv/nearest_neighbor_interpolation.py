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

# Load the image
image_path = 'path/to/your/image.jpg'
image = np.array(Image.open(image_path))

# Define new dimensions
new_width = 400
new_height = 300

# Perform nearest neighbor interpolation
resized_image = nearest_neighbor_interpolation(image, new_width, new_height)

# Save or display the resized image
resized_image_pil = Image.fromarray(resized_image)
resized_image_pil.show()
# or save the resized image
resized_image_pil.save('path/to/save/resized_image.jpg')
