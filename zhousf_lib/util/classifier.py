# -*- coding:utf-8 -*-
import os
import json
import file_util
from zhousf_lib.util import string_util

damage_calsses = [
    'boliposun',
    'boliliewen',
    "huahen",
    "guaca",
    "aoxian",
    "zhezhou",
    "silie",
    "jichuan",
    "queshi",
]
parts_classes = [
    'XingLiXiangGai',
    'HouBaoXianGangPi',
    'FaDongJiZhao',
    'ZhongWang',
    'NeiWeiDeng-Y',
    'HouYeZiBan-Z',
    'QianDaDeng-Y',
    'HouMen-Y',
    'NeiWeiDeng-Z',
    'WaiWeiDeng-Y',
    'HouMen-Z',
    'QianBaoXianGangPi',
    'QianDaDeng-Z',
    'QianYiZiBan-Y',
    'QianMen-Z',
    'QianMen-Y',
    'HouYeZiBan-Y',
    'QianYiZiBan-Z',
    'WaiWeiDeng-Z',
    'QianMenBoLiZuo',
    'QianFenDangBoLi',
    'HouFengDangBoLiJuShengMenBoLi',
    'HouMenBoLiYou',
    'HouMenBoLiZuo',
    'QianMenBoLiYou',
    'DaoCheJingZongCheng-Z',
    'DaoCheJingZongCheng-Y',
    'GangQuan'
]

'''
部件分类器
dst_dir：部件数据存放目录
'''


def parts_classifiler(src_dir, dst_dir):
    _classifier(src_dir, dst_dir, parts_classes)
    # _classifier_mod_direction(src_dir, dst_dir, parts_classes)


'''
损伤分类器
dst_dir：损伤数据存放目录
'''


def damage_classifiler(src_dir, dst_dir):
    _classifier(src_dir, dst_dir, damage_calsses)


'''
将包含classes标签的数据移动到dst_dir中
dst_dir：包含的数据存放目录
'''


def contain_parts(src_dir, dst_dir, classes):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(src_dir):
        print('目录不存在：' + src_dir)
        return
    total = 0
    num = 0
    for root, dirs, files in os.walk(src_dir):
        for json_file in files:
            if not json_file.endswith('.json'):
                continue
            total += 1
            with open(os.path.join(src_dir, json_file), 'r') as load_f:
                load_dict = json.load(load_f)
                shapes = load_dict['shapes']
                imagePath = load_dict['imagePath']
                for i in range(0, len(shapes)):
                    label = shapes[i]['label']
                    if label in classes:
                        num += 1
                        print(imagePath)
                        file_util.move_file(os.path.join(root, json_file), dst_dir)
                        file_util.move_file(os.path.join(root, imagePath), dst_dir)
                        break
    print('total=' + str(total) + ',num=' + str(num))


def compute_each_label_num(src_dir):
    """
    统计json文件中每个标签的个数
    :param src_dir:
    :return:
    """
    if not os.path.exists(src_dir):
        print('目录不存在：' + src_dir)
        return
    cls = {}
    for root, dirs, files in os.walk(src_dir):
        for json_file in files:
            if not json_file.endswith('.json'):
                continue
            with open(os.path.join(root, json_file), 'r') as load_f:
                load_dict = json.load(load_f)
                shapes = load_dict['shapes']
                imagePath = load_dict['imagePath']
                print(imagePath)
                for i in range(0, len(shapes)):
                    cls_name = shapes[i]['label']
                    if cls.has_key(cls_name):
                        n = cls[cls_name]
                    else:
                        n = 0
                    cls[cls_name] = 1 + n
    print(cls)
    return cls


def _classifier(src_dir, dst_dir, classes):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(src_dir):
        print('目录不存在：' + src_dir)
        return
    total = 0
    num = 0
    cls_num_map = {}
    for root, dirs, files in os.walk(src_dir):
        for json_file in files:
            if not json_file.endswith('.json'):
                continue
            total += 1
            with open(os.path.join(src_dir, json_file), 'r') as load_f:
                load_dict = json.load(load_f)
                shapes = load_dict['shapes']
                imagePath = load_dict['imagePath']
                parts_shape = []
                for i in range(0, len(shapes)):
                    label = shapes[i]['label']
                    if label in classes:
                        parts_shape.append(shapes[i])
                        if cls_num_map.has_key(label):
                            cls_num_map[label] += 1
                        else:
                            cls_num_map[label] = 1
                if len(parts_shape) == 0:
                    continue
                num += 1
                print(imagePath)
                load_dict['shapes'] = parts_shape
                file_util.copy_file(os.path.join(root, json_file), dst_dir)
                file_util.copy_file(os.path.join(root, imagePath), dst_dir)
                json.dump(load_dict, open(os.path.join(dst_dir, json_file), 'w'),
                          sort_keys=True, indent=4, separators=(',', ': '))
    print('total=' + str(total) + ',num=' + str(num))
    print(cls_num_map)


'''
修改部件方向:将所有右方向部件修改成左方向部件
'''


def _classifier_mod_direction(src_dir, dst_dir, classes):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(src_dir):
        print('目录不存在：' + src_dir)
        return
    total = 0
    num = 0
    cls_num_map = {}
    for root, dirs, files in os.walk(src_dir):
        for json_file in files:
            if not json_file.endswith('.json'):
                continue
            total += 1
            with open(os.path.join(src_dir, json_file), 'r') as load_f:
                load_dict = json.load(load_f)
                shapes = load_dict['shapes']
                imagePath = load_dict['imagePath']
                parts_shape = []
                for i in range(0, len(shapes)):
                    label = shapes[i]['label']
                    if label in classes:
                        label = label.replace('-Y', '-Z')
                        label = label.replace('You', 'Zuo')
                        shapes[i]['label'] = label
                        parts_shape.append(shapes[i])
                        if cls_num_map.has_key(label):
                            cls_num_map[label] += 1
                        else:
                            cls_num_map[label] = 1
                if len(parts_shape) == 0:
                    continue
                num += 1
                print(imagePath)
                load_dict['shapes'] = parts_shape
                file_util.copy_file(os.path.join(root, json_file), dst_dir)
                file_util.copy_file(os.path.join(root, imagePath), dst_dir)
                json.dump(load_dict, open(os.path.join(dst_dir, json_file), 'w'),
                          sort_keys=True, indent=4, separators=(',', ': '))
    print('total=' + str(total) + ',num=' + str(num))
    print(cls_num_map)


def delete_appointed_label_json(src_dir, dst_dir, labels, delete_num):
    """
    删除指定的包含labels标签的标注文件
    :param src_dir: 目录
    :param dst_dir: 删除文件目录
    :param labels: 指定标签，例如['guaca','huahen']
    :param delete_num: 删除个数
    :return:
    """
    if not os.path.exists(src_dir):
        print('目录不存在：' + src_dir)
        return
    if delete_num <= 0:
        print('delete_num must be > 0')
        return
    total = delete_num
    try:
        for root, dirs, files in os.walk(src_dir):
            for json_file in files:
                if delete_num == 0:
                    return
                if not json_file.endswith('.json'):
                    continue
                with open(os.path.join(src_dir, json_file), 'r') as load_f:
                    load_dict = json.load(load_f)
                    shapes = load_dict['shapes']
                    imagePath = load_dict['imagePath']
                    shapes_total = []
                    for i in range(0, len(shapes)):
                        if shapes[i]['label'] not in shapes_total:
                            shapes_total.append(str(shapes[i]['label']))
                    if len(labels) != len(shapes_total):
                        continue
                    labels = sorted(labels)
                    shapes_total = sorted(shapes_total)
                    if labels == shapes_total:
                        print(os.path.join(root, json_file))
                        delete_num -= 1
                        file_util.move_file(os.path.join(root, json_file), dst_dir)
                        file_util.move_file(os.path.join(root, imagePath), dst_dir)
    finally:
        print('共删除包含'+str(labels)+'的标注文件'+str(total - delete_num)+'个')


def generate_classes_txt(src_dir, output_classes_txt):
    if not os.path.exists(src_dir):
        print('目录不存在：' + src_dir)
        return False
    classes_txt_file = open(output_classes_txt, 'w')
    for root, dirs, files in os.walk(src_dir):
        for json_file in files:
            if not json_file.endswith('.json'):
                continue
            cls = {}
            with open(os.path.join(root, json_file), 'r') as load_f:
                load_dict = json.load(load_f)
                shapes = load_dict['shapes']
                imagePath = load_dict['imagePath']
                print(imagePath)
                for i in range(0, len(shapes)):
                    cls_name = shapes[i]['label']
                    if cls.has_key(cls_name):
                        n = cls[cls_name]
                    else:
                        n = 0
                    cls[cls_name] = 1 + n
                name, _ = json_file.split('.')
                classes_txt_file.write(name + '-@-'+str(cls)+'\n')
    classes_txt_file.close()
    return True


def split_train_val_data(src_dir, dst_dir, output_classes_txt, radio=10):
    """
    将数据按照radio百分比等份分为train和val数据集
    :param src_dir:
    :param dst_dir:
    :param output_classes_txt:
    :param radio: 100等份，radio=10表示val占总数据10%
    :return:
    """
    train_dir = os.path.join(dst_dir, 'train')
    val_dir = os.path.join(dst_dir, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(output_classes_txt):
        print('output_classes_txt不存在：' + output_classes_txt)
        return
    lines = []
    with open(output_classes_txt, 'r') as load_f:
        for line in load_f.readlines():
            line = line.strip('\n')
            lines.append(line)
    class_map = compute_each_label_num(src_dir=src_dir)
    cls = {}
    result = {}
    for key in class_map.keys():
        if key not in result:
            result[key] = 0
        cls[key] = int(class_map[key] * (float(radio)/100.0))
    cls = sorted(cls.items(), key=lambda x: x[1], reverse=False)
    val_files = []
    train_files = []
    for i in range(0, len(cls)):
        cls_name = str(cls[i][0])
        cls_num = cls[i][1]
        if cls_num == 0:
            continue
        for j in range(0,len(lines)):
            line = lines[j]
            file_name, f_cls = line.split('-@-')
            if result[cls_name] - cls_num >= 0:
                train_files.append(file_name)
                break
            f_cls = eval(f_cls)
            if cls_name in f_cls:
                lines.pop(j)
                for f_c in f_cls:
                    result[f_c] += 1
                val_files.append(file_name)
            else:
                train_files.append(file_name)
    print('-------------')
    print(class_map)
    print(cls)
    class_map = sorted(class_map.items(), key=lambda x: x[1], reverse=False)
    result = sorted(result.items(), key=lambda x: x[1], reverse=False)
    print(result)
    num_train = 0
    num_val = 0
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            f_name, ext = f.split('.')
            if f_name in val_files:
                num_val += 1
                file_util.copy_file(os.path.join(root,f), val_dir)
            else:
                num_train += 1
                file_util.copy_file(os.path.join(root, f), train_dir)
    num_train = num_train / 2
    num_val = num_val / 2
    log_txt = open(os.path.join(dst_dir, 'log.txt'), 'w')
    log_txt.write('原数据类别数：' + str(class_map) + '\n')
    log_txt.write('val数据占比：' + str(radio) + '%\n')
    log_txt.write('train：' + str(num_train) + '个\n')
    log_txt.write('val：' + str(num_val) + '个\n')
    log_txt.write('train_val：' + str(num_train + num_val) + '个\n')
    print('train：' + str(num_train) + '个')
    print('val：' + str(num_val) + '个')
    print('train_val：' + str(num_train + num_val) + '个')
    log_txt.close()


if __name__ == '__main__':
    src_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/data'
    dst_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/damage'
    # parts_classifiler(src_dir,dst_dir)
    # damage_classifiler(src_dir,dst_dir)
    # damage_calsses = [
    #     'wuxulabel',
    # ]
    # contain_parts(src_dir, dst_dir, damage_calsses)
    # compute_each_label_num('/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/damage/val')
    # delete_appointed_label_json(src_dir, dst_dir, ['guaca'], 80000)
    src_dir = '/Users/zhousf/tensorflow/zhousf/data/car/car'
    dst_dir = '/Users/zhousf/tensorflow/zhousf/data/car/part'
    output_classes_txt = '/Users/zhousf/tensorflow/zhousf/data/car/part.txt'
    # generate_classes_txt(src_dir=src_dir, output_classes_txt=output_classes_txt)
    split_train_val_data(src_dir=src_dir, dst_dir=dst_dir, output_classes_txt=output_classes_txt, radio=20)
