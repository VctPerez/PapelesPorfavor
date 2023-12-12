import time
from datetime import datetime

import cv2

import camera
import functions as f
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def check_getters(image):
    name = f.get_name(image)
    dob = f.get_DOB(image)
    #f.show_image("DOB", dob)
    sex = f.get_sex(image)
    iss = f.get_issuer(image)
    exp = f.get_expiration(image)
    id = f.get_id(image)

    plt.subplot(321)
    plt.title("Name")
    plt.imshow(name)

    plt.subplot(322)
    plt.title("Sex")
    plt.imshow(sex)

    plt.subplot(323)
    plt.title("Iss")
    plt.imshow(iss)

    plt.subplot(324)
    plt.title("Exp")
    plt.imshow(exp)

    plt.subplot(325)
    plt.title("ID")
    plt.imshow(id)

    # plt.imshow(326)
    # plt.title("DOB")
    # plt.imshow(dob)

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
    victor1 = f.load_face_image("images/passports/Victor/Victor_face.jpg")
    obama = camera.takePicture()
    # locations_victor1 = f.get_face_locations("images/faceRecognition/victor1.jpg")
    # print(f'Faces: {len(locations_victor1)}\n\t Content: {locations_victor1}')
    # f.show_image("victor", victor1)
    # # face_1 = f.subImage(victor1, locations_victor1[0][3], locations_victor1[0][0],
    #                     locations_victor1[0][1] - locations_victor1[0][3],
    #                     locations_victor1[0][2] - locations_victor1[0][0])

    # face_1 = f.passport_image(face_1, 120)

    # f.show_image("face", face_1)
    # descriptor1 = f.get_face_descriptor(victor1)[0]
    descriptor2 = f.get_face_descriptor(victor1)
    print("Descriptor 2 value: ", descriptor2)
    # print("Son iguales ? ->",f.compare_faces([descriptor1], descriptor2))

    # print(f'Difference: {f.get_face_difference(descriptor, f)}')

    # print(f'Descriptor: {f.get_face_descriptor(adri)}')


def check_binarize():
    victor = f.load_image("images/faceRecognition/victor1.jpg")
    return f.change_face_histogram(victor)


def write_image(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ## ESCRIBIR NOMBRE
    image = f.set_name(image, " Victor Perez")

    ## ESCRIBIR NACIMIENTO
    image= f.set_DOB(image, "05/08/2003")
    ## ESCRIBIR SEXO
    image=  f.set_sex(image, "Masc")
    ## ESCRIBIR ISSUER
    image = f.set_issuer(image, "Malaga")

    ## ESCRIBIR CADUCIDAD
    image = f.set_expiration(image, "01/01/2027")

    ## ESCRIBIR ID
    return f.set_id(image, "AAAA-BBBB")


if __name__ == '__main__':
    # path = "images/passports/Victor/Victor_passport.jpg"
    # image = f.load_image(path)
    # fecha_image = f.get_DOB(image)
    # fecha_str = f.extract_text_from_image(fecha_image)
    # print(fecha_str)
    # fecha_datetime = datetime.strptime(fecha_str, "%d/%m/%Y")
    # print(fecha_datetime)

    check_face_recognition()
    ##########################################################################
    # face = f.load_image("images/faceRecognition/victor1.jpg")
    # image = f.insert_face_into_passport(image, face)
    # f.show_image("pass", image)
    # check_getters(image)
    ###########################################################################
    # Uso de sleep para animaciones
    # for i in range(10):
    #     print(i)
    #     time.sleep(1)
    #######################################################################
    # Funcion para cargar imagenes en una etiqueta
    # def load_image(self):
    #     self.filename = QtWidgets.QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
    #     self.image = f.load_image(self.filename)
    #     self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    #     self.image = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0]
    #                               , QtGui.QImage.Format_RGB888)
    #     self.label.setPixmap(QtGui.QPixmap.fromImage(self.image))
