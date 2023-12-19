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
image()