import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout
from subprocess import run

class ProgramSelector(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # 创建标签
        label = QLabel('请选择一个程序:', self)

        # 创建下拉框
        self.program_combobox = QComboBox(self)
        self.program_combobox.addItem('图片检测')
        self.program_combobox.addItem('视频检测')
        self.program_combobox.addItem('退出')

        # 创建按钮
        btn_select = QPushButton('运行', self)
        btn_select.clicked.connect(self.runSelectedProgram)

        # 设置窗口布局
        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addWidget(self.program_combobox)
        vbox.addWidget(btn_select)
        self.setLayout(vbox)

        # 设置窗口的基本属性
        self.setGeometry(500, 500, 500, 200)
        self.setWindowTitle('选择并运行程序')

    def runSelectedProgram(self):
        selected_program = self.program_combobox.currentText()

        if selected_program == '图片检测':
            # 运行程序A的命令
            run(['python', 'F:\pycharmproject\yolov5\image.py'])
        elif selected_program == '视频检测':
            # 运行程序B的命令
            run(['python', 'F:\pycharmproject\yolov5\\video11.py'])
        elif selected_program == '退出':
            # 运行程序C的命令
            app.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ProgramSelector()
    window.show()
    sys.exit(app.exec_())