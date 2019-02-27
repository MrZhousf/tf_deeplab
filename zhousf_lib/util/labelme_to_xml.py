# -*- coding:utf-8 -*-

import os
import json
from xml.dom import minidom
import numpy as np
import file_util
from PIL import Image

def _create_info(jpg_path,dom,annotation):
    folder = dom.createElement('folder')
    folder.appendChild(dom.createTextNode(jpg_path))
    annotation.appendChild(folder)
    filename = dom.createElement('filename')
    filename.appendChild(dom.createTextNode(os.path.basename(jpg_path)))
    annotation.appendChild(filename)
    path = dom.createElement('path')
    path.appendChild(dom.createTextNode(jpg_path))
    annotation.appendChild(path)

def _create_size(width,height,dom,annotation):
    size = dom.createElement('size')
    width_n = dom.createElement('width')
    width_n.appendChild(dom.createTextNode(str(width)))
    size.appendChild(width_n)
    height_n = dom.createElement('height')
    height_n.appendChild(dom.createTextNode(str(height)))
    size.appendChild(height_n)
    annotation.appendChild(size)

def _create_object(xmin,ymin,xmax,ymax,label,dom,annotation):
    object = dom.createElement('object')
    name = dom.createElement('name')
    name.appendChild(dom.createTextNode(label))
    object.appendChild(name)
    bndbox = dom.createElement('bndbox')
    xmin_n = dom.createElement('xmin')
    xmin_n.appendChild(dom.createTextNode(str(xmin)))
    bndbox.appendChild(xmin_n)
    ymin_n = dom.createElement('ymin')
    ymin_n.appendChild(dom.createTextNode(str(ymin)))
    bndbox.appendChild(ymin_n)
    xmax_n = dom.createElement('xmax')
    xmax_n.appendChild(dom.createTextNode(str(xmax)))
    bndbox.appendChild(xmax_n)
    ymax_n = dom.createElement('ymax')
    ymax_n.appendChild(dom.createTextNode(str(ymax)))
    bndbox.appendChild(ymax_n)
    object.appendChild(bndbox)
    annotation.appendChild(object)

'''
json转成box_xml
'''
def json_to_box_xml(src_dir):
    json_list = os.listdir(src_dir)
    for i in range(0, len(json_list)):
        json_file = json_list[i]
        if json_file.endswith('.json'):
            with open(os.path.join(src_dir, json_file), 'r') as load_f:
                filename,ext=json_file.split('.')
                filename += '.xml'
                xml_path=os.path.join(src_dir,filename)
                load_dict = json.load(load_f)
                try:
                    imagePath = load_dict['imagePath']
                    shapes = load_dict['shapes']
                    width = load_dict['imgWidth']
                    height = load_dict['imgHeight']
                    print('w=' + str(width) + ',h=' + str(height))
                    dom = minidom.Document()
                    annotation = dom.createElement('annotation')
                    dom.appendChild(annotation)
                    _create_info(imagePath, dom, annotation)
                    _create_size(width, height, dom, annotation)
                    for i in range(0, len(shapes)):
                        label = shapes[i]['label']
                        points = shapes[i]['points']
                        p = np.array(points)
                        xmin = int(min(p[:, 0]))
                        ymin = int(min(p[:, 1]))
                        xmax = int(max(p[:, 0]))
                        ymax = int(max(p[:, 1]))
                        _create_object(xmin, ymin, xmax, ymax, label, dom, annotation)
                    f = open(xml_path, 'w')
                    dom.writexml(f, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
                    f.close()
                except Exception as ex:
                    print('imgPath=None',ex)

'''
图片转成box_xml
'''
def img_to_box_xml(src_dir):
    img_list = os.listdir(src_dir)
    for i in range(0, len(img_list)):
        img = img_list[i]
        ext = file_util.file_extension(img)
        filename = file_util.file_basename(img)
        path = os.path.join(src_dir, img)
        if ext in {'.jpg','.JPG'}:
            img = Image.open(path)
            width = img.size[0]
            height = img.size[1]
            filename += '.xml'
            label,ot = filename.split('-')
            xml_path = os.path.join(src_dir, filename)
            dom = minidom.Document()
            annotation = dom.createElement('annotation')
            dom.appendChild(annotation)
            _create_info(path, dom, annotation)
            _create_size(width, height, dom, annotation)
            _create_object(0, 0, width, height, label, dom, annotation)
            f = open(xml_path, 'w')
            dom.writexml(f, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
            f.close()

def generate_plate_char(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for root, dirs, files in os.walk(src_dir):
        for dir in dirs:
            path = os.path.join(root, dir)
            lis = os.listdir(path)
            for i in range(0, len(lis)):
                ext = file_util.file_extension(lis[i])
                if ext in {'.jpg', '.JPG'}:
                    name = dir+'-'+str(i)+ext
                    print name
                    from_path = os.path.join(root,dir,lis[i])
                    to_path= os.path.join(dst_dir,name)
                    os.rename(from_path,to_path)



if __name__ == '__main__':
    root_dir = '/home/ubuntu/zsf/dl/plate/'
    src_dir = root_dir + 'data-train'
    # json_to_box_xml(src_dir)
    # src_dir = '/home/ubuntu/zsf/zhousf/data/'
    # dst_dir = '/home/ubuntu/zsf/zhousf/plate/'
    # generate_plate_char(src_dir,dst_dir)
    # src_dir = '/home/ubuntu/zsf/zhousf/plate/'
    # img_to_box_xml(src_dir)
