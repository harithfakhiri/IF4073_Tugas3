from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys
from non_deep_learning import plate_recognition
from deep_learning import plate_recognitionCNN

import numpy as np


class UI(QMainWindow):
	def __init__(self):
		super(UI, self).__init__()

		uic.loadUi("gui.ui", self)

		self.button = self.findChild(QPushButton, "pushButton")
		self.inputImg = self.findChild(QLabel, "label__inputImg")
		self.outputImgCNN = self.findChild(QLabel, "label__outputImgCNN")
		self.outputImgManual = self.findChild(QLabel, "label__outputImgManual")
		self.cnnOCR = self.findChild(QLabel, "label__cnnOCR")
		self.manualOCR = self.findChild(QLabel, "label__manualOCR")
		self.directory = self.findChild(QLabel, "label__directory")

		self.button.clicked.connect(self.clicker)

		self.show()

	def clicker(self):
		fname = QFileDialog.getOpenFileName(
			self, "Open File", "c:\\gui\\images", "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")

		if fname:
			self.pixmap = QPixmap(fname[0])
			self.pixmap = self.pixmap.scaled(300, 300, Qt.KeepAspectRatio)
			self.directory.setText(fname[0])
			self.inputImg.setPixmap(self.pixmap)

			# Manual Plate Detection
			Cropped, result = plate_recognition(fname[0])

			height, width, channel = Cropped.shape
			bytesPerLine = 3 * width

			qImg = QImage(Cropped.data.tobytes(), width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
			self.pixmapManual = QPixmap(qImg)
			self.outputImgManual.setPixmap(self.pixmapManual)
			self.manualOCR.setText(result)

			# CNN Plate Detection
			CroppedCNN, resultCNN = plate_recognitionCNN(fname[0])
			h, w, c = CroppedCNN.shape
			bytesPerLineCNN = 3 * w

			qImgCNN = QImage(CroppedCNN.data.tobytes(), w, h, bytesPerLineCNN, QImage.Format_RGB888).rgbSwapped()
			self.pixmapCNN = QPixmap(qImgCNN)
			self.outputImgCNN.setPixmap(self.pixmapCNN)
			self.cnnOCR.setText(resultCNN)

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
