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
                    capture = cv2.VideoCapture(video_path)
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

if __name__ == '__main__':
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
            if (selected_program == "程序B"):
                video()

    app = QApplication(sys.argv)
    window = ProgramSelector()
    window.show()
    sys.exit(app.exec_())