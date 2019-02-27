# -*- coding:utf-8 -*-
import xml.etree.ElementTree as ET
import os
import json
import file_util
import base64
import numpy as np


def dict_json(shapes, imagepath='', imgpath='', imgwidth='', imgHeight='', fillcolor=None, linecolor=None,
              labelUserName='' ,qualityUserName='' ,secondQualityUserName='', companyName='', title=''):
    flags = {}
    return {'shapes': shapes, 'lineColor': linecolor, 'imagePath': imagepath,
            'flags': flags, 'fillColor': fillcolor,
            'imgPath': imgpath, 'imgWidth': imgwidth, 'imgHeight': imgHeight,
            'labelUserName': labelUserName, 'qualityUserName': qualityUserName, 'secondQualityUserName': secondQualityUserName,
            'companyName': companyName, 'title': title}


def dict_shapes(points, label='', fill_color=None, line_color=None):
    return {'line_color': line_color, 'points': points, 'fill_color': fill_color, 'label': label}


def get_shapes_box(root):
    shapes = []
    for obj in root.findall('object'):
        points = []
        try:
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            points.append([xmin, ymin])
            points.append([xmax, ymin])
            points.append([xmax, ymax])
            points.append([xmin, ymax])
            # box的points最少是4个点
            if not len(points) >= 4:
                return None
            shapes.append(dict_shapes(points, label))
        except Exception as err:
            print('error ', err)
            return None
    return shapes


def box_xml_to_json(xml_dir, dest_dir, abort_dir):
    fillcolor = [255, 0, 0, 128]
    linecolor = [0, 255, 0, 128]
    total = 0
    num = 0
    for root, dirs, files in os.walk(xml_dir):
        for xml_file in files:
            if xml_file.endswith('.xml'):
                file_path = os.path.join(root, xml_file)
                tree = ET.parse(file_path)
                root_xml = tree.getroot()
                filename = root_xml.find('filename').text
                jpg_path = root_xml.find('path').text
                total += 1
                if filename.endswith('.jpg'):
                    filename = filename[:len(filename) - 4] + '.JPG'
                jpg_file = os.path.join(root, filename)
                if not os.path.exists(jpg_file):
                    if filename.endswith('.JPG'):
                        filename = filename[:len(filename) - 4] + '.jpg'
                    jpg_file = os.path.join(root, filename)
                    if not os.path.exists(jpg_file):
                        file_util.copy_file(file_path, abort_dir)
                        continue
                shapes = get_shapes_box(root_xml)
                if shapes is None:
                    file_util.copy_file(jpg_file, abort_dir)
                    file_util.copy_file(file_path, abort_dir)
                    continue
                num += 1
                with open(jpg_file, 'rb') as f:
                    # imagedata = base64.b64encode(f.read())
                    imgwidth = root_xml.find('size').find('width').text
                    imgHeight = root_xml.find('size').find('height').text
                    # print('w=' + imgwidth + ',h=' + imgHeight)
                    data = dict_json(shapes=shapes, imagepath=filename, imgpath=jpg_path, imgwidth=imgwidth,
                                     imgHeight=imgHeight, fillcolor=fillcolor, linecolor=linecolor)
                    file_basename = filename.split('.')
                    json_file = '%s/%s.json' % (root, file_basename[0])
                    print(json_file)
                    json.dump(data, open(json_file, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
                    if xml_dir != dest_dir:
                        file_util.copy_file(jpg_file, dest_dir)
    print('共:' + str(total) + '项,有效项：' + str(num))


def get_shapes_segment(root):
    shapes = []
    for obj in root.findall('object'):
        points = []
        try:
            label = obj.find('name').text
            ps = obj.find('points')
            for p in ps.findall('point'):
                x = p.find('x').text
                y = p.find('y').text
                points.append([int(x), int(y)])
            # segment的points最少是3个点
            if not len(points) >= 3:
                return None
            shapes.append(dict_shapes(points, label))
        except Exception as err:
            print('error ', err)
            return None
    return shapes


def segment_xml_to_json(src_dir, dest_dir, abort_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    fillcolor = [255, 0, 0, 128]
    linecolor = [0, 255, 0, 128]
    total = 0
    abort_num = 0
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if not f.endswith('.xml'):
                continue
            total += 1
            file_path = os.path.join(src_dir, f)
            tree = ET.parse(file_path)
            root = tree.getroot()
            labelUserName = ''
            qualityUserName = ''
            secondQualityUserName = ''
            companyName = ''
            title = ''
            labelInfo = root.find('labelInfo')
            if labelInfo is not None:
                if labelInfo.find('labelUserName') is not None:
                    labelUserName = labelInfo.find('labelUserName').text
                if labelInfo.find('qualityUserName') is not None:
                    qualityUserName = labelInfo.find('qualityUserName').text
                if labelInfo.find('secondQualityUserName') is not None:
                    secondQualityUserName = labelInfo.find('secondQualityUserName').text
                if labelInfo.find('companyName') is not None:
                    companyName = labelInfo.find('companyName').text
                if labelInfo.find('title') is not None:
                    title = labelInfo.find('title').text
            filename = root.find('filename').text
            jpg_path = root.find('path').text
            # if filename.endswith('.JPG'):
            #     filename = filename[:len(filename) - 4] + '.jpg'
            jpg_file = os.path.basename(filename)
            jpg_file = os.path.join(src_dir, jpg_file)
            shapes = get_shapes_segment(root)
            if shapes is None:
                file_util.copy_file(jpg_file, abort_dir)
                file_util.copy_file(file_path, abort_dir)
                abort_num += 1
                continue
            if jpg_file.endswith('.JPG'):
                if not os.path.exists(jpg_file):
                    jpg_file = jpg_file.replace('.JPG','.jpg')
                    jpg_path = jpg_path.replace('.JPG','.jpg')
                    if not os.path.exists(jpg_file):
                        abort_num += 1
                        file_util.copy_file(file_path, abort_dir)
                        continue
            if jpg_file.endswith('.jpg'):
                if not os.path.exists(jpg_file):
                    jpg_file = jpg_file.replace('.jpg','.JPG')
                    jpg_path = jpg_path.replace('.jpg','.JPG')
                    if not os.path.exists(jpg_file):
                        abort_num += 1
                        file_util.copy_file(file_path, abort_dir)
                        continue
            with open(jpg_file, 'rb') as f:
                # imagedata = base64.b64encode(f.read())
                imgwidth = root.find('size').find('width').text
                imgHeight = root.find('size').find('height').text
                filebasename = os.path.basename(jpg_path)
                data = dict_json(shapes=shapes, imagepath=filebasename, imgpath=jpg_path, imgwidth=imgwidth,
                                 imgHeight=imgHeight, fillcolor=fillcolor, linecolor=linecolor,
                                 labelUserName=labelUserName, qualityUserName=qualityUserName,
                                 secondQualityUserName=secondQualityUserName, companyName=companyName, title=title)
                file_basename = filebasename.split('.')
                json_file = '%s/%s.json' % (dest_dir, file_basename[0])
                json.dump(data, open(json_file, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
                if src_dir != dest_dir:
                    file_util.copy_file(jpg_file, dest_dir)
            print(str(total))
    print('共:' + str(total) + '项，有效项：' + str(total - abort_num))


def _create_object(xmin, ymin, xmax, ymax, label, dom, annotation):
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


if __name__ == '__main__':
    # root_dir = '/home/ubuntu/zsf/zhousf/'
    # src_dir = root_dir + 'plate'
    # dst_dir = root_dir + 'plate'
    # abort_dir = root_dir + 'del'
    # box_xml_to_json(src_dir,dst_dir,root_dir)

    src_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-gxh/20190211-whole-yunju/data'
    dst_dir = src_dir
    abort_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-gxh/20190211-whole-yunju/abort'
    # box_xml_to_json(src_dir, dst_dir, abort_dir)
    segment_xml_to_json(src_dir, dst_dir, abort_dir)
