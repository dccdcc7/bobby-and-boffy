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

video()