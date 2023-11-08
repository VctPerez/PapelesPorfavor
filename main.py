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
    adri = f.load_face_image("images/faceRecognition/adri.jpg")
    faces = f.get_face_locations("images/faceRecognition/adri.jpg")
    print(f'Faces: {len(faces)}\n Content: {faces}')

    jose = f.subImage(adri, faces[0][3], faces[0][0], faces[0][1] - faces[0][3], faces[0][2] - faces[0][0])

    jose = f.passport_image(jose, 74)

    f.show_image("jose", jose)


def check_binarize():
    victor = f.load_image("images/faceRecognition/victor1.jpg")
    victor_binarized = f.binarize_image(victor, 100)
    f.show_image("victor", victor_binarized)
    victor_matched = f.matcher_histograms(victor_binarized)
    f.show_image("victor_matched", victor_matched)


if __name__ == '__main__':
    path = "images/pasaporte_1.jpg"
    image = f.load_image(path)
    # check_getters(image)
    check_face_recognition()
    # check_binarize()
