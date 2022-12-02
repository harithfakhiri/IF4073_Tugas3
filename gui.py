from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
import sys


class UI(QMainWindow):
	def __init__(self):
		super(UI, self).__init__()

		uic.loadUi("gui.ui", self)

		self.button = self.findChild(QPushButton, "pushButton")
		self.inputImg = self.findChild(QLabel, "label__inputImg")
		self.outputImgCNN = self.findChild(QLabel, "label__outputImgCNN")
		self.outputImgManual = self.findChild(QLabel, "label__outputImgManual")
		self.directory = self.findChild(QLabel, "label__directory")

		self.button.clicked.connect(self.clicker)

		self.show()

	def clicker(self):
		fname = QFileDialog.getOpenFileName(
			self, "Open File", "c:\\gui\\images", "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")

		if fname:
			self.pixmap = QPixmap(fname[0])
			self.directory.setText(fname[0])
			self.inputImg.setPixmap(self.pixmap)
            
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
