#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
import sys
from yolo import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,QComboBox
from PyQt5.QtCore import Qt
#import keyboard

yolo = YOLO()
#----------------------------------------------------------------------------------------------------------#
#   mode用于指定测试的模式：
#   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
#   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
#   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
#   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
#----------------------------------------------------------------------------------------------------------#
mode = "predict"
#-------------------------------------------------------------------------#
#   crop指定了是否在单张图片预测后对目标进行截取
#   crop仅在mode='predict'时有效
#-------------------------------------------------------------------------#
crop            = False
#----------------------------------------------------------------------------------------------------------#
#   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
#   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
#   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
#   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
#   video_fps用于保存的视频的fps
#   video_path、video_save_path和video_fps仅在mode='video'时有效
#   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
#----------------------------------------------------------------------------------------------------------#
video_path      = "6.mp4"
video_save_path = "6out.mp4"
video_fps       = 24.0
#-------------------------------------------------------------------------#
#   test_interval用于指定测量fps的时候，图片检测的次数
#   理论上test_interval越大，fps越准确。
#-------------------------------------------------------------------------#
test_interval   = 2000
#-------------------------------------------------------------------------#
#   dir_origin_path指定了用于检测的图片的文件夹路径
#   dir_save_path指定了检测完图片的保存路径
#   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
#-------------------------------------------------------------------------#
dir_origin_path = "img/"
dir_save_path   = "img_out/"

def image():
    '''
    1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。
    2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
    3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
    在原图上利用矩阵的方式进行截取。
    4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
    比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
    '''
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
            try:
                image = Image.open(filename)
            except:
                print('Open Error! Try again!')
                # continue
            else:
                r_image = yolo.detect_image(image, crop=crop)
                # r_image_cv2 = cv2.UMat(r_image)
                r_image.show()
                r_image.save("img1.jpg")

    app1 = QApplication(sys.argv)
    window1 = MyWindow()
    window1.showDialog()
    sys.exit(app1.exec_())

# def video1(video_path):


def video():
    class VideoFileReader(QWidget):
        def __init__(self):
            super().__init__()

            self.initUI()

        def initUI(self):
            # 创建标签用于显示文件路径
            self.label_filepath = QLabel('文件路径：', self)

            # 创建按钮
            btn_browse = QPushButton('浏览视频文件', self)
            btn_browse.clicked.connect(self.browseVideoFile)

            # 设置窗口布局
            vbox = QVBoxLayout()
            vbox.addWidget(self.label_filepath)
            vbox.addWidget(btn_browse)
            self.setLayout(vbox)

            # 设置窗口的基本属性
            self.setGeometry(300, 300, 400, 150)
            self.setWindowTitle('视频文件读取器')

        def browseVideoFile(self):
            # 打开文件对话框，选择视频文件
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("Video files (*.mp4 *.avi *.mkv)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setViewMode(QFileDialog.Detail)

            if file_dialog.exec_():
                # 获取用户选择的文件路径
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    video_filepath = selected_files[0]
                    capture = cv2.VideoCapture(video_filepath)
                    if video_save_path != "":
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

                    fps = 0.0
                    while True:
                        t1 = time.time()
                        # 读取某一帧
                        ref, frame = capture.read()
                        if not ref:
                            break
                        # 格式转变，BGRtoRGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 转变成Image
                        frame = Image.fromarray(np.uint8(frame))
                        # 进行检测
                        frame = np.array(yolo.detect_image(frame))
                        # RGBtoBGR满足opencv显示格式
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        fps = (fps + (1. / (time.time() - t1))) / 2
                        print("fps= %.2f" % (fps))
                        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

                        cv2.imshow("video", frame)
                        c = cv2.waitKey(1) & 0xff
                        if video_save_path != "":
                            out.write(frame)

                        if c == 27:
                            capture.release()
                            break

                    print("Video Detection Done!")
                    capture.release()
                    if video_save_path != "":
                        print("Save processed video to the path :" + video_save_path)
                        out.release()
                    cv2.destroyAllWindows()

    app2 = QApplication(sys.argv)
    window2 = VideoFileReader()
    window2.show()
    sys.exit(app2.exec_())

def fps():
    img = Image.open('img/street.jpg')
    tact_time = yolo.get_FPS(img, test_interval)
    print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

def dir_predict():
    import os

    from tqdm import tqdm

    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(dir_origin_path, img_name)
            image       = Image.open(image_path)
            r_image     = yolo.detect_image(image)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

# else:
#     raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
class ProgramSelector(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # 创建标签
        label = QLabel('请选择一个程序:', self)

        # 创建下拉框
        self.program_combobox = QComboBox(self)
        self.program_combobox.addItem('程序A')
        self.program_combobox.addItem('程序B')
        self.program_combobox.addItem('程序C')

        # 创建按钮
        btn_select = QPushButton('选择', self)
        btn_select.clicked.connect(self.on_select)

        # 设置窗口布局
        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addWidget(self.program_combobox)
        vbox.addWidget(btn_select)
        self.setLayout(vbox)

        # 设置窗口的基本属性
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('选择程序')

    def on_select(self):
        selected_program = self.program_combobox.currentText()
        if (selected_program == "程序A"):
            image()
        elif (selected_program == "程序B"):
            video()


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = ProgramSelector()
    window.show()
    sys.exit(app.exec_())