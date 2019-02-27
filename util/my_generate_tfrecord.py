"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf


from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

classMember1 = [
"boliposun",
"boliliewen",
"huahen",
"guaca",
"aoxian",
"zhezhou",
"silie",
"jichuan",
"queshi",
"Qian",
"Zuo",
"Hou",
"You",
"ZuoQian",
"ZuoHou",
"YouHou",
"YouQian",
"QianBaoXianGangPi",
"FaDongJiZhao",
"QianYiZiBan-Z",
"A-Zhu-Z",
"QianMen-Z",
"HouMen-Z",
"HouYeZiBan-Z",
"DiDaBian-Z",
"HouBaoXianGangPi",
"XingLiXiangGai",
"HouYeZiBan-Y",
"DiDaBain-Y",
"HouMen-Y",
"QianMen-Y",
"QianYiZiBan-Y",
"A-Zhu-Y",
"CheDing",
"QianDaDeng-Z",
"QianWuDeng-Z",
"QianDaDeng-Y",
"QianWuDeng-Y",
"WaiWeiDeng-Z",
"NeiWeiDeng-Z",
"WaiWeiDeng-Y",
"NeiWeiDeng-Y",
"DaoCheJingZongCheng-Z",
"DaoCheJingZongCheng-Y",
"QianWuDengHuZhao-Z",
"QianWuDengHuZhao-Y",
"ZhongWang",
"QianBaoXianGangXiaGeShan",
"GangQuan",
"HouFengDangBoLiJuShengMenBoLi",
"YouQianYeZiBanZhuanXiangDeng",
"ZuoQianYeZiBanZhuanXiangDeng",
"ZuoQianYeZiBanLunMei",
"QianMenBoLiYou",
"HouMenBoLiYou",
"QianFenDangBoLi",
"QianBaoXiaDuan",
"ZuoQianMenShiTiao",
"ZuoHouMenShiTiao",
"ZuoQianMenShiBen",
"HouMenSanJiaoBoLiYou",
"YouQianMenShiTiao",
"YouHouMenShiTiao",
"ZuoCeDiDaBianShiTiaoShiBen",
"HouCeWeiBoLiYou",
"HouCeWeiBoLiZuo",
"YouQianYeZiBanLunMei",
"QianMenBoLiZuo",
"YouHouYeZiBanLunMei",
"HouMenBoLiZuo",
"HouMenSanJiaoBoLiZuo",
"QianMenSanJiaoBoLiYou",
"YouQianMenShiBan",
"ZuoHouMenShiBan",
"YouHouMenShiBan",
"ZuoHouYeZiBanLunMei",
"YouCeDiDaBianShiTiaoShiBan",
"HouBaoXiaDuan",
"QianMenSanJiaoBoLiZuo",
"BeiTaiZhao",
"YouQianYeZiBanShiBan",
"ZuoQianYeZiBanShiBan"
]

classMember = [
"boliposun",
"huahen",
"guaca",
"aoxian"
]


# TO-DO replace this with label map
def class_text_to_int(row_label):
    # idx = 0
    # for classname in classMember:
    #     idx += 1
    #     if row_label == classname:
    #         return idx
    # return None
    if row_label == 'boliposun' or  row_label == 'boliliewen' :
        return 1
    elif row_label == 'huahen':
        return 2
    elif row_label == 'guaca':
        return 3
    elif row_label == 'aoxian':
        return 4
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        class_id = class_text_to_int(row['class'])
        if class_id != None:
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_id)
            print(class_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate(data_path,csv_path,output_tf_record_path):
    writer = tf.python_io.TFRecordWriter(output_tf_record_path)
    examples = pd.read_csv(csv_path)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, data_path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_tf_record_path))


