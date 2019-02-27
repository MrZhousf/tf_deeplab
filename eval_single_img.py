# -*- coding:utf-8 -*-
from train_config import config


if __name__ == '__main__':
    # image_path = '/home/ubuntu/zsf/dl/plate/data/2293-C610100VEH16017888_15.jpg'
    image_path = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/dogs/dogs/n02085620-Chihuahua/n02085620_368.jpg'
    # image_path = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/cards/cards/yinghangka/1222_1.jpg'
    # image_path = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/flowers/flower_photos/roses/1469726748_f359f4a8c5.jpg'
    # image_path = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/cards/cards/shengfenzheng/1224_3.jpg'
    image_path = '/home/ubuntu/zsf/zhousf/213.JPG'
    config.TRAIN_MODEL.vis_single_img(image_path)