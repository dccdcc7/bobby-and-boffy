import xml.dom.minidom
import cv2
import os
import xml.etree.ElementTree as ET

"""
该脚本用于目标框可视化
IMAGE_INPUT_PATH：输入图片路径
XML_INPUT_PATH：输入标记框路径
IMAGE_OUTPUT_PATH：生成可视化图片路径
"""
IMAGE_INPUT_PATH = 'F:\pycharmproject\yolov5\VOCdevkit\VOC2007\JPEGImages'
XML_INPUT_PATH = 'F:\pycharmproject\yolov5\VOCdevkit\VOC2007\Annotations'
IMAGE_OUTPUT_PATH = 'F:\pycharmproject\yolov5\VOCdevkit\VOC2007'

imglist = os.listdir(IMAGE_INPUT_PATH)
xmllist = os.listdir(XML_INPUT_PATH)

for i in range(len(imglist)):
    # 每个图像全路径
    image_input_fullname = IMAGE_INPUT_PATH + '/' + imglist[i]
    xml_input_fullname = XML_INPUT_PATH + '/' + xmllist[i]
    image_output_fullname = IMAGE_OUTPUT_PATH + '/' + imglist[i]

    image = cv2.imread(image_input_fullname)

    # 解析XML文件
    tree = ET.parse(xml_input_fullname)
    root = tree.getroot()

    # 获取图像尺寸
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    # 遍历所有目标
    for obj in root.findall('object'):
        # 获取目标类别和边界框坐标
        label = obj.find('name').text
        xmin = float(obj.find('bndbox/xmin').text)
        ymin = float(obj.find('bndbox/ymin').text)
        xmax = float(obj.find('bndbox/xmax').text)
        ymax = float(obj.find('bndbox/ymax').text)

        # 根据类别选择颜色
        if label == 'Car':
            color = (255, 0, 0)  # 蓝色
        elif label == 'Van':
            color = (0, 255, 0)  # 绿色
        else:
            color = (255, 192, 203)  # 不同的颜色

        # 在图像上画出边界框和类别标签
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(image, label, (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        #直接查看生成结果图
        cv2.imshow('show', image)
        cv2.waitKey(0)


    #cv2.imwrite(image_output_fullname, image)

