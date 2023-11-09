import tkinter
import tkinter.messagebox
import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title('PapelesPorfavor')
        self.geometry(f'{1280}x{720}')


if __name__ == '__main__':
    app = App()
    app.mainloop()
