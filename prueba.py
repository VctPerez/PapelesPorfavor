from PyQt5.QtWidgets import (QWidget, QApplication, QPushButton,
                             QLabel, QHBoxLayout, QSizePolicy, QFrame)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QPropertyAnimation, pyqtProperty, QRect
import sys

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.button = QPushButton("Start", self)
        self.button.clicked.connect(self.doAnim)
        self.button.move(30, 30)

        self.frame = QFrame(self)
        self.frame.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        self.frame.setGeometry(150, 30, 100, 100)

        self.setGeometry(300, 300, 380, 300)
        self.setWindowTitle('Animation')
        self.show()

    def doAnim(self):

        self.anim = QPropertyAnimation(self.frame, b"geometry")
        self.anim.setDuration(1666)
        self.anim.setStartValue(QRect(150, 30, 100, 100))
        self.anim.setEndValue(QRect(150, 200, 100, 100))
        self.anim.start()


def main():

    app = QApplication([])
    ex = Example()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()