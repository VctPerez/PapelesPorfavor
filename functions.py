import cv2
import numpy as np
import pytesseract as pts
import face_recognition
from scipy import signal
from skimage.exposure import match_histograms

pts.pytesseract.tesseract_cmd = r'C:\Users\vparm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# pts.pytesseract.tesseract_cmd = r'C:\Users\Victor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def load_image(path):
    return cv2.imread(path)


def show_image(w_name, image):
    cv2.imshow(w_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def subImage(image, x, y, w, h):
    return image[y:y + h, x:x + w]


def binarize_image(image, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, image_binarized = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return image_binarized


def matcher_histograms(image, reference_path='images/matcher.png'):
    """Matches the histogram from the reference image and the out image

    :param reference_path: String path of the reference image
    :param image: Input Image
    :return: Reference histogram applied
    """
    # image_BGR = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    reference = load_image(reference_path)
    return match_histograms(image, reference, channel_axis=-1)


def gauss_formula(x, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x * x) / (2 * sigma * sigma))


def lut_chart(image, gamma, verbose=False):
    """ Applies gamma correction to an image and shows the result.

        Args:
            image: Input image
            gamma: Gamma parameter
            verbose: Only show images if this is True

        Returns:
            out_image: Gamma image
    """

    # Transform image to YCrCb color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    out_image = np.copy(image)

    # Define gamma correction LUT
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply LUT to first band of the YCrCb image
    out_image[:, :, 0] = cv2.LUT(out_image[:, :, 0], lut)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_YCrCb2BGR)
    return out_image


def gaussian_filter(image, w_kernel, sigma):
    """ Applies Gaussian filter to an image and display it.

        Args:
            image: Input image
            w_kernel: Kernel aperture size
            sigma: standard deviation of Gaussian distribution

        Returns:
            smoothed_img: smoothed image
    """

    # Create kernel using associative property
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    s = sigma
    w = w_kernel
    kernel_1D = np.float32([gauss_formula(z, s) for z in range(-w, w + 1)])  # Evaluate the gaussian in "expression"
    vertical_kernel = kernel_1D.reshape(2 * w + 1, 1)  # Reshape it as a matrix with just one column
    horizontal_kernel = kernel_1D.reshape(1, 2 * w + 1)  # Reshape it as a matrix with just one row
    kernel = signal.convolve2d(vertical_kernel, horizontal_kernel)  # Get the 2D kernel

    smoothed_img = cv2.filter2D(gray, cv2.CV_16S, kernel)
    # smoothed_img = np.float32(smoothed_img)
    smoothed_img = cv2.cvtColor(smoothed_img, cv2.COLOR_GRAY2BGR)
    return smoothed_img


def passport_image(image, thresh):
    gray_image = binarize_image(image, thresh)
    return matcher_histograms(gray_image)


def extract_text_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = pts.image_to_string(threshold_image)
    return text


def get_name(image):
    return subImage(image, 26, 439, 500, 56)


def get_sex(image):
    return subImage(image, 350, 549, 100, 51)


def get_issuer(image):
    return subImage(image, 350, 594, 120, 40)


def get_expedition(image):
    return subImage(image, 350, 630, 200, 50)


def load_face_image(path):
    return face_recognition.load_image_file(path)


def get_face_locations(image):
    return face_recognition.face_locations(image)


def get_face_from_image(image, coords):
    return subImage(image, coords[0][3], coords[0][0], coords[0][1] - coords[0][3], coords[0][2] - coords[0][0])


def get_face_descriptor(face, locations=None):
    if locations is None:
        return face_recognition.face_encodings(face)
    else:
        return face_recognition.face_encodings(face, locations)


def get_face_difference(encoding_list, face2):
    return face_recognition.face_distance(encoding_list, face2)


def compare_faces(descriptor_list, descriptor2):
    return face_recognition.compare_faces(descriptor_list, descriptor2, 0.6)
