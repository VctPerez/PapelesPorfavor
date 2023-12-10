import time

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
    id = f.get_id(image)

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

    plt.show()
    text1 = f.extract_text_from_image(name)
    print(text1)
    text1 = f.extract_text_from_image(sex)
    print(text1)
    text1 = f.extract_text_from_image(iss)
    print(text1)
    text1 = f.extract_text_from_image(exp)
    print(text1)
    text1 = f.extract_text_from_image(id)
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
    return f.change_face_histogram(victor)


def write_image(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ## ESCRIBIR NOMBRE
    image = cv2.putText(image, "Victor Perez Armenta", (43, 488), font, fontScale=1.2, color=(82, 69, 63), thickness=2,
                         lineType=cv2.LINE_AA)

    ## ESCRIBIR NACIMIENTO
    image= cv2.putText(image, "05/08/2003", (358, 540), font, fontScale=0.8, color=(82, 69, 63), thickness=2,
                       lineType=cv2.LINE_AA)
    ## ESCRIBIR SEXO
    image = cv2.putText(image, "Masc", (358, 580), font, fontScale=0.8, color=(82, 69, 63), thickness=2,
                       lineType=cv2.LINE_AA)
    ## ESCRIBIR ISSUER
    image = cv2.putText(image, "Malaga", (358, 620), font, fontScale=0.8, color=(82, 69, 63), thickness=2,
                       lineType=cv2.LINE_AA)

    ## ESCRIBIR CADUCIDAD
    image = cv2.putText(image, "01/01/2027", (358, 660), font, fontScale=0.8, color=(82, 69, 63), thickness=2,
                        lineType=cv2.LINE_AA)

    ## ESCRIBIR ID
    return cv2.putText(image, "AAAA-BBBB", (43, 800), font, fontScale=1.2, color=(82, 69, 63), thickness=2,
                       lineType=cv2.LINE_AA)


if __name__ == '__main__':
    path = "images/pasaporte_1.jpg"
    image = f.load_image(path)
    f.show_image("imagen", image)
    image = write_image(image)
    f.show_image("image", image)
    check_getters(image)
    # check_face_recognition()
    ##########################################################################
    # face = f.load_image("images/faceRecognition/victor1.jpg")
    # image = f.insert_face_into_passport(image, face)
    # f.show_image("pass", image)
    ###########################################################################
    # main()
    # for i in range(10):
    #     print(i)
    #     time.sleep(1)
    #######################################################################
    # def load_image(self):
    #     self.filename = QtWidgets.QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
    #     self.image = f.load_image(self.filename)
    #     self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    #     self.image = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0]
    #                               , QtGui.QImage.Format_RGB888)
    #     self.label.setPixmap(QtGui.QPixmap.fromImage(self.image))
