import os,glob,json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

#获得中心点的坐标
def get_cx_cy(points):
    cx = (points[0][0]+points[1][0])/2.
    cy = (points[0][1]+points[1][1])/2.
    points = np.array(points)
    xmin = min(points[:, 0])
    xmax = max(points[:, 0])
    ymin = min(points[:, 1])
    ymax = max(points[:, 1])
    w = max(points[:,0])-min(points[:,0])
    h = max(points[:,1])-min(points[:,1])
    return xmin, xmax, ymin, ymax

#美化内容（+换行）
def indent(elem,level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

#写入xml文件
def write_xml(data,imgpath,savepath):
    root = ET.Element('annotation') #创建节点
    tree = ET.ElementTree(root) #创建文档
    #图片文件上一级目录
    folder = ET.Element("folder")
    img_folder = imgpath.split(os.sep)[-2]
    folder.text = img_folder
    root.append(folder)
    #文件名
    imgname = Path(imgpath).stem
    #imgname = os.path.basename(imgpath)
    filename = ET.Element("filename")
    filename.text = imgname + ".png"
    root.append(filename)

    #路径
    path = ET.Element("path")
    path.text = imgpath
    root.append(path)

    #source
    source = ET.Element("source")
    root.append(source)
    database = ET.Element("database")
    database.text = "Unknown"
    source.append(database)

    #size
    size = ET.Element("size")
    root.append(size)
    width = ET.Element("width") #宽
    width.text = str(data["imageWidth"])
    size.append(width)
    height = ET.Element("height")#高
    height.text = str(data["imageHeight"])
    size.append(height)
    depth = ET.Element("depth") #深度
    depth.text = str(3)
    size.append(depth)

    #segmented
    segmented = ET.Element("segmented")
    segmented.text = str(0)
    root.append(segmented)

    # 目标
    for shape in data["shapes"]:
        object_ = ET.Element("object")
        root.append(object_)
        #标注框类型
        # type_ = ET.Element("type")
        # type_.text = "bndbox"
        # object_.append(type_)
        #目标类别
        name = ET.Element("name")
        name.text = shape["label"]
        object_.append(name)
        #pose
        pose = ET.Element("pose")
        pose.text = "Unspecified"
        object_.append(pose)
        #截断情况
        truncated = ET.Element("truncated")
        truncated.text = str(0) #默认为0，表示未截断
        object_.append(truncated)
        #样本困难度
        # difficult = ET.Element("difficult")
        # difficult.text = str(0) #默认为0，表示非困难样本
        # object_.append(difficult)
        #四个端点
        bndbox = ET.Element("bndbox")
        object_.append(bndbox)
        # 获得中心点的坐标和宽高
        xmin, xmax, ymin, ymax = get_cx_cy(shape["points"])

        # 看你的xml文件中需要保存的内容
        xmin_ = ET.Element("xmin")
        xmin_.text = str(xmin)
        bndbox.append(xmin_)
        #cy
        ymin_ = ET.Element("ymin")
        ymin_.text = str(ymin)
        bndbox.append(ymin_)
        #w
        xmax_ = ET.Element("xmax")
        xmax_.text = str(xmax)
        bndbox.append(xmax_)
        #h
        ymax_ = ET.Element("ymax")
        ymax_.text = str(ymax)
        bndbox.append(ymax_)
        # #angle
        # angle = ET.Element("angle")
        # angle.text = str(0.0)
        # bndbox.append(angle)

    indent(root,0)
    tree.write(savepath+os.sep+imgname+".xml","UTF-8",xml_declaration=True)

#解析json文件
def load_json(jsonpath):
    data = json.load(open(jsonpath,"r"))
    del data["version"]
    try:
        del data["flags"]
    except Exception as e:
        del data["flag"]
    del data["imagePath"]
    del data["imageData"]
    return data

if __name__ == '__main__':
    json_dir = r"F:\pycharmproject\yolov5\VOCdevkit\VOC2007\SegmentationClass"    #原json文件的路径
    imgpath = r"F:\pycharmproject\yolov5\VOCdevkit\VOC2007\JPEGImages"
    save_dir = r"F:\pycharmproject\yolov5\VOCdevkit\VOC2007\Annotations"   #保存文件夹路径
    os.makedirs(save_dir,exist_ok=True)
    json_file = os.listdir(json_dir)
    for i in range(len(json_file)):
        jsonpath = os.path.join(json_dir,str(json_file[i]))
        imgpath = jsonpath.replace(".json", ".png")
        # 加载json文件中的数据，获得data
        data = load_json(jsonpath)
        write_xml(data, imgpath, save_dir)

