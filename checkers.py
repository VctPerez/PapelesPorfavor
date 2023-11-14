import cv2

import functions as f
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def check_getters(image):
    name = f.get_name(image)
    sex = f.get_sex(image)
    iss = f.get_issuer(image)
    exp = f.get_expedition(image)

    plt.subplot(221)
    plt.title("Name")
    plt.imshow(name)

    plt.subplot(222)
    plt.title("Sex")
    plt.imshow(sex)

    plt.subplot(223)
    plt.title("Iss")
    plt.imshow(iss)

    plt.subplot(224)
    plt.title("Exp")
    plt.imshow(exp)

    text1 = f.extract_text_from_image(name)
    print(text1)
    text1 = f.extract_text_from_image(sex)
    print(text1)
    text1 = f.extract_text_from_image(iss)
    print(text1)
    text1 = f.extract_text_from_image(exp)
    print(text1)


def check_face_recognition():
    victor1 = f.load_face_image("images/faceRecognition/victor1.jpg")
    obama = f.load_face_image("images/faceRecognition/obama.png")
    # locations_victor1 = f.get_face_locations("images/faceRecognition/victor1.jpg")
    # print(f'Faces: {len(locations_victor1)}\n\t Content: {locations_victor1}')
    # f.show_image("victor", victor1)
    # # face_1 = f.subImage(victor1, locations_victor1[0][3], locations_victor1[0][0],
    #                     locations_victor1[0][1] - locations_victor1[0][3],
    #                     locations_victor1[0][2] - locations_victor1[0][0])

    # face_1 = f.passport_image(face_1, 120)

    # f.show_image("face", face_1)
    descriptor1 = f.get_face_descriptor(victor1)[0]
    descriptor2 = f.get_face_descriptor(obama)[0]
    print(f.compare_faces([descriptor1], descriptor2))

    # print(f'Difference: {f.get_face_difference(descriptor, f)}')

    # print(f'Descriptor: {f.get_face_descriptor(adri)}')


def check_binarize():
    victor = f.load_image("images/faceRecognition/victor1.jpg")
    victor_binarized = f.binarize_image(victor, 100)
    f.show_image("victor", victor_binarized)
    victor_matched = f.matcher_histograms(cv2.cvtColor(victor_binarized, cv2.COLOR_GRAY2BGR))
    f.show_image("victor_matched", victor_matched)


if __name__ == '__main__':
    path = "images/pasaporte_1.jpg"
    image = f.load_image(path)
    # check_getters(image)
    check_face_recognition()
    # check_binarize()
