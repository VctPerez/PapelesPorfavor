import functions as f
if __name__ == '__main__':
    path = "images/example_01.png"
    image = f.load_image(path)

    text = f.extract_text_from_image(image)
    #f.show_image("meme", image)
    print(text)
