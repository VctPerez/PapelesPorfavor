import cv2
import numpy as np
import pytesseract as pts
import face_recognition
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


def matcher_histograms(image):
    image_BGR = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    reference = load_image("images/matcher.png")
    return match_histograms(image_BGR, reference, channel_axis=-1)


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


def get_face_locations(path):
    image = load_face_image(path)
    return face_recognition.face_locations(image)


def get_face_from_image(image, coords):
    return subImage(image, coords[0][3], coords[0][0], coords[0][1] - coords[0][3], coords[0][2] - coords[0][0])
