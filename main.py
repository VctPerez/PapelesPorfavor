import functions as f
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "images/pasaporte prueba2.png"
    image = f.load_image(path)
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
    # f.show_image("passport", image)
    # image_cropped = f.subImage(image,26,439,344,56)
    text1 = f.extract_text_from_image(name)
    print(text1)
    text1 = f.extract_text_from_image(sex)
    print(text1)
    text1 = f.extract_text_from_image(iss)
    print(text1)
    text1 = f.extract_text_from_image(exp)
    print(text1)
    # f.show_image("cropped", image_cropped)
