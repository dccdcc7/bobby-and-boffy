import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # 创建按钮
        btn = QPushButton('打开图片文件', self)
        btn.clicked.connect(self.showDialog)

        # 创建标签用于显示图片
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # 设置窗口布局
        vbox = QVBoxLayout()
        vbox.addWidget(btn)
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        # 设置窗口的基本属性
        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('PyQt5 打开图片文件示例')
        self.show()

    def showDialog(self):
        # 打开文件对话框，选择图片文件
        fname = QFileDialog.getOpenFileName(self, '打开图片文件', '/home', 'Image files (*.jpg *.png *.bmp)')

        # 获取文件名
        filename = fname[0]
        #return filename
        self.showImage(filename)
    def showImage(self, filename):
        # 显示选择的图片
        pixmap = QPixmap(filename)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 将文件名传递给后面的函数，这里只是打印文件名
        print(filename)
        return filename


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    #print(window.showDialog())
    sys.exit(app.exec_())
