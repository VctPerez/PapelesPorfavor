import cv2
import numpy as np
import pytesseract as pts
import matplotlib.pyplot as plt


pts.pytesseract.tesseract_cmd = r'C:\Users\vparm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# pts.pytesseract.tesseract_cmd = r'C:\Users\Victor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def auto_subplotter(size_x, size_y, d):
    i = 1
    for pair in d.items():
        plt.subplot(size_x, size_y, i)
        plt.title(pair[1])
        plt.imshow(pair[0])
        i += 1

    plt.show()


def load_image(path):
    return cv2.imread(path)


def show_image(w_name, image):
    cv2.imshow(w_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def subImage(image, x, y, w, h):
    return image[y:y + h, x:x + w]


def extract_text_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = pts.image_to_string(threshold_image)
    return text


def get_name(image):
    return subImage(image, 26, 439, 344, 56)


def get_sex(image):
    return subImage(image, 339, 548, 186, 34)


def get_issuer(image):
    return subImage(image, 339, 582, 251, 36)


def get_expedition(image):
    return subImage(image, 339, 617, 251, 40)