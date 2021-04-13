import numpy as np
import cv2 as cv

from skimage.feature import hog
from scipy import ndimage

def hogProcess(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1, 1), visualize=True, multichannel=False)
    return hog_image

def sharpen(image):
    kernel_size = (5, 5)
    sigma = 1.0
    amount = 10.0
    threshold = 0
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def process_image(image):
    image = cv.resize(image, (64, 64))

    # Sharpen image
    image = sharpen(image)

    # Convert to Grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Denoise image
    image = ndimage.gaussian_filter(image, 3)

    # Binarize image
    ret, image = cv.threshold(image, 120, 255, cv.THRESH_BINARY)

    # Get HOG features
    image = hogProcess(image)
    cv.imshow("Hog", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Convert image to array
    image = np.array(image).flatten()

    return image