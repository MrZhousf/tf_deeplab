# -*- coding: utf-8 -*-
import os, shutil
import cv2
import file_util
import sys
import numpy as np
from PIL import Image
import time
import colorsys

'''
获取img_dir目录中图片最大的宽和高的文件
img_dir：图片目录
ext_list：图片扩展名
'''


def fetch_max_width_height(img_dir, ext_list={'.JPG', '.jpg', '.png'}):
    width_max = [0, 0, 0]
    width_max_file = ''
    height_max = [0, 0, 0]
    height_max_file = ''
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            ext = file_util.file_extension(os.path.join(root, f))
            if ext in ext_list:
                img = cv2.imread(os.path.join(root, f))
                sp = img.shape
                height = sp[0]
                width = sp[1]
                channel = sp[2]
                if width > width_max[0]:
                    width_max[0] = width
                    width_max[1] = height
                    width_max[2] = channel
                    width_max_file = os.path.join(root, f)
                if height > height_max[1]:
                    height_max[0] = width
                    height_max[1] = height
                    height_max[2] = channel
                    height_max_file = os.path.join(root, f)
                print ('width: %d height: %d number: %d' % (width, height, channel))
    print('宽度最大的图片：' + str(width_max) + ',' + width_max_file)
    print('高度最大的图片：' + str(height_max) + ',' + height_max_file)


'''
将img_dir中所有宽或高大于限制宽或高的图片移动到out_limit_dir目录中
'''


def move_limit_width_height(img_dir, out_limit_dir, limit_width, limit_height, ext_list={'.JPG', '.jpg', '.png'}):
    if not os.path.exists(out_limit_dir):
        os.makedirs(out_limit_dir)
    total = 0
    limit = 0
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            ext = file_util.file_extension(os.path.join(root, f))
            if ext in ext_list:
                total += 1
                img = cv2.imread(os.path.join(root, f))
                sp = img.shape
                height = sp[0]
                width = sp[1]
                channel = sp[2]
                if width > limit_width:
                    file_util.move_file(os.path.join(root, f), out_limit_dir)
                    json_f = file_util.file_basename(f)
                    json_f += '.json'
                    file_util.move_file(os.path.join(root, json_f), out_limit_dir)
                    print ('width: %d height: %d channel: %d' % (width, height, channel))
                    limit += 1
                    continue
                if height > limit_height:
                    file_util.move_file(os.path.join(root, f), out_limit_dir)
                    json_f = file_util.file_basename(f)
                    json_f += '.json'
                    file_util.move_file(os.path.join(root, json_f), out_limit_dir)
                    print ('width: %d height: %d channel: %d' % (width, height, channel))
                    limit += 1
                    continue
    print('total=' + str(total) + ',limit=' + str(limit))


def img2gray(imagePath):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(imagePath, gray)


def gray2rgb_single(img_path):
    src = cv2.imread(img_path, 0)
    src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(img_path, src_RGB)


def gray2rgb(src_dir):
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            ext = file_util.file_extension(f)
            if ext in {'.jpg', '.JPG'}:
                path = os.path.join(root, f)
                print path


def show(img_path):
    new = cv2.imread(img_path)
    cv2.imshow(img_path, new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/damage'
    out_limit_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/damage-limit'
    # fetch_max_width_height(img_dir)
    move_limit_width_height(img_dir, out_limit_dir, 1920, 1920)
    # img_dir = '/home/ubuntu/zsf/dl/sample'
    # out_limit_dir = '/home/ubuntu/zsf/dl/sample-limit'
    # fetch_max_width_height(img_dir)
    # move_limit_width_height(img_dir,out_limit_dir,1024,768)
    image_path = '/home/ubuntu/zsf/1.jpg'
    # img2gray(image_path)
    # gray2rgb_single(image_path)
    # show('/home/ubuntu/zsf/zhousf/213.JPG')
    # dir = '/home/ubuntu/zsf/zhousf/plate'
    # gray2rgb(dir)
