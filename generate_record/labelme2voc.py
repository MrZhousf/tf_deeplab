# -*- coding: utf-8 -*-
from __future__ import print_function
import glob
import json
import os
import os.path as osp
import numpy as np
import PIL.Image
from util import labelme_shape
from util import labelme_draw


def generate_png(labels_file, in_dir, out_dir):
    """
    生成voc图片
    :param labels_file: 类别文件
    :param in_dir:
    :param out_dir:
    :return:
    """
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels_file).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(out_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    # colormap = labelme.utils.label_colormap(255)
    colormap = labelme_draw.label_colormap(255)
    # for label_file in glob.glob(osp.join(in_dir, '*.json')):
    for root, dirs, files in os.walk(in_dir):
        for label_file in files:
            if not label_file.endswith('.json'):
                continue
            label_file = os.path.join(root, label_file)
            print('Generating dataset from:', label_file)
            with open(label_file) as f:
                base = osp.splitext(osp.basename(label_file))[0]
                out_img_file = osp.join(
                    out_dir, 'JPEGImages', base + '.jpg')
                out_lbl_file = osp.join(
                    out_dir, 'SegmentationClass', base + '.png')
                out_viz_file = osp.join(
                    out_dir, 'SegmentationClassVisualization', base + '.jpg')
                data = json.load(f)
                img_file = osp.join(osp.dirname(label_file), data['imagePath'])
                img = np.asarray(PIL.Image.open(img_file))
                PIL.Image.fromarray(img).save(out_img_file)
                # lbl = labelme.utils.shapes_to_label(
                #     img_shape=img.shape,
                #     shapes=data['shapes'],
                #     label_name_to_value=class_name_to_id,
                # )
                try:
                    # 对标注的label面积由大到小排序，防止因标注顺序问题导致大的遮盖了小的
                    lbl = labelme_shape.shapes_to_label_sorted(
                        img_shape=img.shape,
                        shapes=data['shapes'],
                        label_name_to_value=class_name_to_id,
                    )
                    lbl_pil = PIL.Image.fromarray(lbl)
                    # Only works with uint8 label
                    # lbl_pil = PIL.Image.fromarray(lbl, mode='P')
                    # lbl_pil.putpalette((colormap * 255).flatten())
                    lbl_pil.save(out_lbl_file)
                    # 生成验证图片-训练不需要，可以屏蔽
                    # label_names = ['%d: %s' % (cls_id, cls_name)
                    #                for cls_id, cls_name in enumerate(class_names)]
                    # viz = labelme_draw.draw_label(
                    #     lbl, img, label_names, colormap=colormap)
                    # PIL.Image.fromarray(viz).save(out_viz_file)
                except Exception as ex:
                    print (ex.message)
                    print (label_file)


def generate_txt(output_dir, val_ratio=1.0/5.0):
    """
    创建txt文件
    train.txt
    val.txt
    trainval.txt
    val_ratio：验证数据集的比例：1.0/5.0
    当val_ratio=0时，则全为train
    当val_ratio=1时，则全为val
    :param output_dir:
    :param val_ratio:
    :return:
    """
    image_sets = output_dir + '/ImageSets'
    segmentation = image_sets + '/Segmentation'
    if not os.path.exists(segmentation):
        os.makedirs(segmentation)
    train_txt = segmentation + '/train.txt'
    trainval_txt = segmentation + '/trainval.txt'
    val_txt = segmentation + '/val.txt'
    if os.path.exists(train_txt):
        os.remove(train_txt)
    if os.path.exists(trainval_txt):
        os.remove(trainval_txt)
    if os.path.exists(val_txt):
        os.remove(val_txt)
    file_map = []
    png_dir = output_dir + '/SegmentationClass'
    for root, dirs, files in os.walk(png_dir):
        for f in files:
            if f.endswith('.png'):
                name,ext = f.split('.png')
                file_map.append(name)
    train_txt_file = open(train_txt, 'w')
    val_txt_file = open(val_txt, 'w')
    trainval_txt_file = open(trainval_txt, 'w')
    for i in range(0,len(file_map)):
        fname = file_map[i]
        if 0 < val_ratio < 1:
            if i % (1 / val_ratio) == 0:
                val_txt_file.write(fname + '\n')
            else:
                train_txt_file.write(fname + '\n')
        elif val_ratio == 0:
            train_txt_file.write(fname + '\n')
        elif val_ratio == 1:
            val_txt_file.write(fname + '\n')
        trainval_txt_file.write(fname+'\n')
    train_txt_file.close()
    val_txt_file.close()
    trainval_txt_file.close()


def generate_segmentation(data_dir, output_dir):
    """
    生成segmentation中的train.txt val.txt trainval.txt
    :param data_dir: 数据目录中必须包含train目录和val目录
    :param output_dir: 输出txt目录
    :return:
    """
    image_sets = output_dir + '/ImageSets'
    segmentation = image_sets + '/Segmentation'
    if not os.path.exists(segmentation):
        os.makedirs(segmentation)
    train_txt = segmentation + '/train.txt'
    trainval_txt = segmentation + '/trainval.txt'
    val_txt = segmentation + '/val.txt'
    if os.path.exists(train_txt):
        os.remove(train_txt)
    if os.path.exists(trainval_txt):
        os.remove(trainval_txt)
    if os.path.exists(val_txt):
        os.remove(val_txt)
    train_txt_file = open(train_txt, 'w')
    val_txt_file = open(val_txt, 'w')
    trainval_txt_file = open(trainval_txt, 'w')
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.JPG'):
                name, ext = f.split('.')
                trainval_txt_file.write(name + '\n')
                if root.endswith('train'):
                    train_txt_file.write(name + '\n')
                if root.endswith('val'):
                    val_txt_file.write(name + '\n')
    train_txt_file.close()
    val_txt_file.close()
    trainval_txt_file.close()


def generate_record(out_dir):
    """
    生成record文件
    :param out_dir:
    :return:
    """
    from datasets import build_voc2012_data as build_voc
    OUTPUT_DIR = out_dir + "/record"
    IMAGE_FOLDER = out_dir + "/JPEGImages"
    SEMANTIC_SEG_FOLDER = out_dir + "/SegmentationClassRaw"
    LIST_FOLDER = out_dir + "/ImageSets/Segmentation"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    record = 'python %s \
                --image_folder=%s \
                --semantic_segmentation_folder=%s \
                --list_folder=%s \
                --image_format=jpg \
                --output_dir=%s' % (os.path.abspath(build_voc.__file__),
                                    IMAGE_FOLDER,
                                    SEMANTIC_SEG_FOLDER,
                                    LIST_FOLDER,
                                    OUTPUT_DIR
                                    )
    os.system(record)


def remove_gt_colormap(out_dir):
    """
    voc图片处理
    :param out_dir:
    :return:
    """
    from datasets import remove_gt_colormap as gt
    SEG_FOLDER = out_dir+"/SegmentationClass"
    SEMANTIC_SEG_FOLDER = out_dir+"/SegmentationClassRaw"
    cc = 'python %s \
            --original_gt_folder=%s \
            --output_dir=%s'%(os.path.abspath(gt.__file__), SEG_FOLDER,SEMANTIC_SEG_FOLDER)
    os.system(cc)


def verify(image_path):
    img = PIL.Image.open(image_path)
    img = np.array(img)
    print(np.unique(img))


def create_voc_dirs(out_dir):
    if osp.exists(out_dir):
        print('Output directory already exists:', out_dir)
        quit(1)
    os.makedirs(out_dir)
    os.makedirs(osp.join(out_dir, 'JPEGImages'))
    os.makedirs(osp.join(out_dir, 'SegmentationClass'))
    os.makedirs(osp.join(out_dir, 'SegmentationClassVisualization'))
    image_sets = out_dir + '/ImageSets'
    segmentation = image_sets + '/Segmentation'
    if not os.path.exists(segmentation):
        os.makedirs(segmentation)


if __name__ == '__main__':
    """
    1. 通过xml匹配jpg： file_util.py
    2. 剔除尺寸和符合规范的样本：img_util->move_limit_width_height
    3. 样本均衡处理：classifier.py->delete_appointed_label_json
    4. 生成数据分布文件：classifier.py->generate_classes_txt
    5. 划分训练数据与评估数据：classifier.py->split_train_val_data
    6. 生成voc目录：create_voc_dirs
    7. 生成segmentation txt文件：generate_segmentation
    8. 生成voc图片：generate_png
    9. voc图片处理：remove_gt_colormap
    10. 生成record文件：generate_record
    """

    # 注意路径中不能带有[]符号
    base_dir = '/Users/zhousf/tensorflow/zhousf/data/car/'
    model = 'part'
    labels_file = base_dir+'labels-'+model+'.txt'
    in_dir = base_dir+model
    out_dir = base_dir+'dataset-'+model
    # create_voc_dirs(out_dir=out_dir)
    # generate_segmentation(data_dir=in_dir,output_dir=out_dir)
    # generate_png(labels_file,in_dir,out_dir)
    # remove_gt_colormap(out_dir)
    generate_record(out_dir)

    # verify('/home/ubuntu/zsf/dl/aoxian/dataset-aoxian/SegmentationClassRaw/19.png')
    # verify('/home/ubuntu/zsf/dl/plate/dataset-plate/SegmentationClassRaw/37-SAM_4956.png')
    # verify('/home/ubuntu/zsf/dl/dataset-sample/SegmentationClassRaw/123107-BigBull_SDC15175.png')


    # remove_gt_colormap('/home/ubuntu/zsf/tf/models-new/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012')
    # generate_record('/home/ubuntu/zsf/tf/models-new/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012')images_per_person
    # remove_gt_colormap('/home/ubuntu/zsf/dl/dataset-damage')
    # generate_record('/home/ubuntu/zsf/dl/dataset-damage')
