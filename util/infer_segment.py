# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    def __init__(self, input_size, model_path, per_process_gpu_memory_fraction=1):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        self.input_size = input_size
        with open(model_path) as fd:
            graph_def = tf.GraphDef.FromString(fd.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        import time
        start = time.time()
        resize_ratio = 1.0 * self.input_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size,
                                                    Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={
                self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]
            })
        seg_map = batch_seg_map[0]

        cost = int((time.time() - start) * 1000)
        log_txt = '总耗时：' + str(cost) + 'ms' + '\n'
        # print log_txt
        return resized_image, seg_map


def vis_segmentation(class_names, image, seg_map):
    FULL_LABEL_MAP = np.arange(len(class_names)).reshape(len(class_names), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
    plt.figure()
    plt.subplot(221)
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')
    plt.subplot(222)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    # seg_image = label_to_color_image(
    #     seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(223)
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    # print(unique_labels)
    # print(class_names[unique_labels])
    # print(FULL_COLOR_MAP[unique_labels].astype(np.uint8))
    ax = plt.subplot(224)
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8),
        interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), class_names[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0)
    plt.show()


class InferSegment(object):

    def __init__(self, input_size,label_file_path, pb_model_path, per_process_gpu_memory_fraction=1):
        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.label_file_path = label_file_path
        self.model = DeepLabModel(input_size,pb_model_path,per_process_gpu_memory_fraction)
        self.full_color_map = None
        self.class_names = None

    def infer(self, image_path):
        if self.class_names is None:
            with open(self.label_file_path, 'r') as load_f:
                file_context = load_f.read().splitlines()
                self.class_names = np.asarray(file_context)
        orignal_im = Image.open(image_path)
        resized_im, seg_map = self.model.run(orignal_im)
        # seg_image = label_to_color_image(seg_map).astype(np.uint8)
        class_label_map = {}
        for i in range(0, len(self.class_names)):
            class_label_map[i] = self.class_names[i]
        return class_label_map, seg_map, resized_im

    def eval(self, image_path, show=True, crop=0):
        """
        :param image_path:
        :param show:
        :param crop: 数字越小放大比例越大，建议设置为8
        :return:
        """
        if self.class_names is None:
            with open(self.label_file_path, 'r') as load_f:
                file_context = load_f.read().splitlines()
                self.class_names = np.asarray(file_context)
        orignal_im = Image.open(image_path)
        if crop != 0:
            w, h = orignal_im.size
            # 从(x,y)的1/8处开始裁剪
            orignal_im = orignal_im.crop([w / crop, h / crop, w, h])
        resized_im, seg_map = self.model.run(orignal_im)
        if show:
            return vis_segmentation(self.class_names, resized_im, seg_map)
        else:
            class_label_map = {}
            for i in range(0, len(self.class_names)):
                class_label_map[i] = self.class_names[i]
            return class_label_map, seg_map

    # def eval(self, image_path, show=True):
    #     with open(self.label_file_path, 'r') as load_f:
    #         file_context = load_f.read().splitlines()
    #         class_names = np.asarray(file_context)
    #         orignal_im = Image.open(image_path)
    #         resized_im, seg_map = self.model.run(orignal_im)
    #         if show:
    #             return vis_segmentation(class_names, resized_im, seg_map)
    #         else:
    #             labels = np.unique(seg_map)
    #             if self.full_color_map is None:
    #                 FULL_LABEL_MAP = np.arange(len(class_names)).reshape(len(class_names), 1)
    #                 self.full_color_map = FULL_LABEL_MAP.astype(np.uint8)
    #             unique_labels = self.full_color_map[labels].astype(np.uint8)
    #             class_label_map = {}
    #             for i in range(0, len(unique_labels)):
    #                 class_label_map[unique_labels[i][0]] = class_names[labels][i]
    #             return class_label_map, seg_map


if __name__ == '__main__':
    class_names_file = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_pascal_train_aug/main_damage/class_names.txt'
    pb_model_path = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_pascal_train_aug/main_damage/export/1265783/frozen_inference_graph.pb'
    # class_names_file = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_pascal_train_aug/main_part/class_names.txt'
    # pb_model_path = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_pascal_train_aug/main_part/export/1215726/frozen_inference_graph.pb'
    # class_names_file = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_pascal_train_aug/new_part/class_names.txt'
    # pb_model_path = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_pascal_train_aug/new_part/export/549595/frozen_inference_graph.pb'
    # class_names_file = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_pascal_train_aug/main_damage/class_names.txt'
    # pb_model_path = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_pascal_train_aug/main_damage/export/1000/frozen_inference_graph.pb'
    image_path1 = '/home/ubuntu/zsf/tf/5.jpg'
    image_path1 = '/home/ubuntu/zsf/tf/img/5.jpg'
    image_path1 = '/home/ubuntu/zsf/tf/img/error/right_back-1.JPG'
    image_path1 = '/home/ubuntu/zsf/tf/7.jpg'
    image_path1 = '/home/ubuntu/zsf/zhousf/download/2.jpg'
    image_path1 = '/home/ubuntu/zsf/tf/53.JPG'
    image_path1 = '/home/ubuntu/zsf/zhousf/auto_test/1.jpg'
    image_path1 = '/home/ubuntu/zsf/zhousf/n1.jpg'
    image_path1 = '/home/ubuntu/zsf/zhousf/download_image/1/4.JPG'
    image_path1 = '/home/ubuntu/zsf/zhousf/213.JPG'
    image_path1 = '/home/ubuntu/zsf/zhousf/auto_test/data1/287.JPG'
    # image_path1 = '/home/ubuntu/zsf/zhousf/download/20180918/business_123/47096/near.jpg'
    # image_path1 = '/home/ubuntu/zsf/tf/img/error/right_back-1.JPG'
    # infer = InferSegment(513, class_names_file, pb_model_path)
    infer = InferSegment(1025, class_names_file, pb_model_path)
    infer.eval(image_path1,show=True,crop=0)

    # img = Image.open(image_path1)
    # img_c = img.crop([img.size[0] / 8, img.size[1] / 8, img.size[0], img.size[1]])
    # plt.imshow(img_c)
    # plt.show()



