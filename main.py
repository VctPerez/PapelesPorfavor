import functions as f
if __name__ == '__main__':
    path = "images/pasaporte prueba.png"
    image = f.load_image(path)
    print(len(image))
    f.show_image("passport", image)
    image_cropped = f.subImage(image,26,439,344,56)
    text = f.extract_text_from_image(image_cropped)
    print(text)
    f.show_image("cropped", image_cropped)


