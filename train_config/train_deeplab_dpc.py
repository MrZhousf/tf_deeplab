# -*- coding: utf-8 -*-
import os
from util import infer_segment
import warnings
import shutil
import numpy as np

'''
deeplab训练基类
'''


class DeepLabDPC(object):

    def __init__(self,
                 train_name='',
                 dataset_dir='',
                 pretrain_checkpoint=None,
                 gpu_with_train='0',
                 gpu_with_eval='1',
                 model_name='deeplabv3_dpc_cityscapes_trainfine',
                 dataset='pascal_voc_seg',
                 train_num=10000000,
                 batch_size=1,
                 eval_vis_crop_size=None,
                 eval_scales=1.0,
                 train_crop_size=None):
        """
        :param train_name:
        :param dataset_dir:
        :param pretrain_checkpoint:预训练ckpt
        :param gpu_with_train: 多GPU训练时，batch_size等于GPU个数
        :param gpu_with_eval:
        :param model_name:
        :param dataset:
        :param train_num:
        :param batch_size:
        :param eval_vis_crop_size: 评估和可视化图片剪裁尺寸.
        该值的设置应该为评估样本中最大高和宽的图片的高、宽值
        :param eval_scales:
        :param train_crop_size:
        """
        if train_crop_size is None:
            train_crop_size = [513, 513]
        self.eval_scales = eval_scales
        if eval_vis_crop_size is None:
            eval_vis_crop_size = [513, 513]
        self.infer_segment = None
        self.train_name = train_name
        self.model_name = model_name
        self.gpu_with_train = gpu_with_train
        self.gpu_with_eval = gpu_with_eval
        # TIME = time.strftime("%Y-%m-%d", time.localtime())
        self.model_dir = os.getcwd() + '/models-master/research/deeplab_dpc'
        self.dpc_json_file = self.model_dir + '/core/dense_prediction_cell_branch5_top1_cityscapes.json'
        self.mymodels_dir = os.getcwd() + '/my_models'
        if pretrain_checkpoint is None:
            self.initial_checkpoint = self.mymodels_dir + '/' + model_name + '/model.ckpt'
        else:
            self.initial_checkpoint = pretrain_checkpoint
        self.dataset = dataset
        self.train_num = train_num
        self.dataset_dir = dataset_dir
        dpath = dataset_dir.strip('/').split('/')
        record = dpath[len(dpath) - 1]
        self.class_names_file, ext = dataset_dir.split(record)
        # 模型类别名称
        self.class_names_file += 'class_names.txt'
        train_model = self.mymodels_dir + "/" + self.model_name + "/" + self.train_name
        class_names_file = train_model + '/class_names.txt'
        if not os.path.exists(class_names_file):
            if not os.path.exists(self.class_names_file):
                warnings.warn(self.class_names_file + '不存在')
            if not os.path.exists(train_model):
                os.makedirs(train_model)
            shutil.copy(self.class_names_file, train_model)
        self.class_names_file = class_names_file
        with open(self.class_names_file, 'r') as load_f:
            file_context = load_f.read().splitlines()
            class_names = np.asarray(file_context)
            num_classes = len(class_names)
            self.num_classes = num_classes
            self.batch_size = batch_size
            self.eval_vis_crop_size = eval_vis_crop_size
            self.train_crop_size = train_crop_size
            # 日志目录
            self.log_dir = train_model + "/log"
            # 可视化目录
            self.vis_dir = train_model + "/vis"
            # 训练目录
            self.train_dir = train_model + "/train"
            # 评估目录
            self.eval_dir = train_model + "/eval"
            # 保存模型目录
            self.save_model_dir = train_model + "/export"
            # 保存训练模型文件
            self.trained_checkpoint = self.train_dir + "/model.ckpt"
            if not os.path.exists(self.train_dir):
                os.makedirs(self.train_dir)
            if not os.path.exists(self.eval_dir):
                os.makedirs(self.eval_dir)
            if not os.path.exists(self.save_model_dir):
                os.makedirs(self.save_model_dir)
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

    def train(self):
        gpu_num = len(self.gpu_with_train.split(','))
        # 多GPU训练时，batch_size等于GPU个数
        if gpu_num > 1:
            self.batch_size = gpu_num
        train = 'python %s/train.py \
            --logtostderr=%s \
            --train_split="train" \
            --model_variant="xception_71" \
            --atrous_rates=6 \
            --atrous_rates=12 \
            --atrous_rates=18 \
            --output_stride=16 \
            --decoder_output_stride=4 \
            --train_crop_size=%s \
            --train_crop_size=%d \
            --log_steps=1 \
            --save_interval_secs=600 \
            --save_summaries_images=true \
            --train_batch_size=%d \
            --training_number_of_steps=%d \
            --fine_tune_batch_norm=false \
            --initialize_last_layer=false \
            --dataset=%s \
            --tf_initial_checkpoint=%s \
            --train_logdir=%s \
            --dense_prediction_cell_json=%s \
            --dataset_dir=%s \
            --num_clones=%d '% (self.model_dir,
                                self.log_dir,
                                self.train_crop_size[0],
                                self.train_crop_size[1],
                                self.batch_size,
                                self.train_num,
                                self.dataset,
                                self.initial_checkpoint,
                                self.train_dir,
                                self.dpc_json_file,
                                self.dataset_dir,
                                gpu_num)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_with_train
        os.system(train)

    def eval(self):
        # max_number_of_evaluations=0时则线程循环等待评估，1则只评估1次
        eval = 'python %s/my_eval.py \
            --logtostderr=%s \
            --eval_split="val" \
            --eval_scales=%d \
            --model_variant="xception_71" \
            --atrous_rates=6 \
            --atrous_rates=12 \
            --atrous_rates=18 \
            --output_stride=16 \
            --decoder_output_stride=4 \
            --max_number_of_evaluations=0 \
            --eval_crop_size=%d \
            --eval_crop_size=%d \
            --eval_batch_size=%d \
            --dataset=%s \
            --checkpoint_dir=%s \
            --eval_logdir=%s \
            --class_names_file=%s \
            --eval_interval_secs=%d \
            --dense_prediction_cell_json=%s \
            --dataset_dir=%s' % (self.model_dir,
                                 self.log_dir,
                                 self.eval_scales,
                                 self.eval_vis_crop_size[0],
                                 self.eval_vis_crop_size[1],
                                 self.batch_size,
                                 self.dataset,
                                 self.train_dir,
                                 self.eval_dir,
                                 self.class_names_file,
                                 7200,
                                 self.dpc_json_file,
                                 self.dataset_dir)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_with_eval
        os.system(eval)

    def export(self):
        print(self.dpc_json_file)
        file_list = os.listdir(self.train_dir)
        check_file = []
        # 找出最大的ckpt进行保存
        for i in range(0, len(file_list)):
            path = os.path.join(self.train_dir, file_list[i])
            if path.endswith(".index"):
                name, index = path.split("-")
                num, ext = index.split(".")
                check_file.append(int(num))
        point = max(check_file)
        if point > 0:
            checkpoint = self.trained_checkpoint + "-" + str(point)
            save_dir = self.save_model_dir + "/" + str(point)
        else:
            checkpoint = self.trained_checkpoint
            save_dir = self.save_model_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_pb = save_dir + '/frozen_inference_graph.pb'
        shutil.copy(checkpoint+'.meta', save_dir)
        shutil.copy(checkpoint+'.index', save_dir)
        shutil.copy(checkpoint+'.data-00000-of-00001', save_dir)
        print(checkpoint)
        print(save_pb)
        export = 'python %s/export_model.py \
            --logtostderr=%s \
            --checkpoint_path=%s \
            --export_path=%s \
            --model_variant="xception_71" \
            --atrous_rates=6 \
            --atrous_rates=12 \
            --atrous_rates=18 \
            --output_stride=16 \
            --decoder_output_stride=4 \
            --num_classes=%d \
            --crop_size=%s \
            --crop_size=%s \
            --dense_prediction_cell_json=%s \
            --inference_scales=1.0' % (self.model_dir,
                                       self.log_dir,
                                       checkpoint,
                                       save_pb,
                                       self.num_classes,
                                       self.train_crop_size[0],
                                       self.train_crop_size[1],
                                       self.dpc_json_file)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_with_eval
        os.system(export)

    def vis(self):
        vis = 'python %s/vis.py \
            --logtostderr=%s \
            --vis_split="val" \
            --model_variant="xception_71" \
            --atrous_rates=6 \
            --atrous_rates=12 \
            --atrous_rates=18 \
            --output_stride=16 \
            --decoder_output_stride=4 \
            --max_number_of_evaluations=1 \
            --also_save_raw_predictions=false \
            --vis_crop_size=%d \
            --vis_crop_size=%d \
            --dataset=%s \
            --checkpoint_dir=%s \
            --vis_logdir=%s \
            --dense_prediction_cell_json=%s \
            --dataset_dir=%s' % (self.model_dir,
                                 self.log_dir,
                                 self.eval_vis_crop_size[0],
                                 self.eval_vis_crop_size[1],
                                 self.dataset,
                                 self.train_dir,
                                 self.vis_dir,
                                 self.dpc_json_file,
                                 self.dataset_dir)
        os.system(vis)

    def show_eval(self, port=6008):
        tensor_board = 'tensorboard --logdir %s/ --port %d' % (self.eval_dir, port)
        os.system(tensor_board)

    def show_train(self, port=6006):
        tensor_board = 'tensorboard --logdir %s/ --port %d' % (self.train_dir, port)
        os.system(tensor_board)

    def vis_single_img(self, image_path, class_names_file=None, pb_model_path=None, show=True):
        if class_names_file is None:
            class_names_file = self.class_names_file
        if pb_model_path is None:
            file_list = os.listdir(self.save_model_dir)
            check_file = []
            for i in range(0, len(file_list)):
                if os.path.isdir(os.path.join(self.save_model_dir, file_list[i])):
                    check_file.append(int(file_list[i]))
            if len(check_file) == 0:
                warnings.warn('frozen_inference_graph.pb不存在')
                return
            max_num = str(max(check_file))
            pb_model_path = self.save_model_dir + '/' + max_num + '/frozen_inference_graph.pb'
            if not os.path.exists(pb_model_path):
                warnings.warn(pb_model_path + '不存在')
                return
        if self.infer_segment is None:
            self.infer_segment = infer_segment.InferSegment(self.train_crop_size[0], class_names_file, pb_model_path)
        return self.infer_segment.eval(image_path, show)


class TrainPart(DeepLabDPC):
    def __init__(self):
        train_name = 'part'
        dataset = 'part_pascal_voc_seg'
        dataset_dir = '/Users/zhousf/tensorflow/zhousf/data/car/dataset-car/record'
        eval_vis_crop_size = [1920, 1920]
        train_num = 10
        batch_size = 1
        train_crop_size = [513, 513]
        gpu_with_train = ''
        gpu_with_eval = ''
        pretrain_checkpoint = None
        DeepLabDPC.__init__(self,
                            train_name,
                            dataset=dataset,
                            dataset_dir=dataset_dir,
                            pretrain_checkpoint=pretrain_checkpoint,
                            eval_vis_crop_size=eval_vis_crop_size,
                            batch_size=batch_size,
                            train_crop_size=train_crop_size,
                            gpu_with_train=gpu_with_train,
                            gpu_with_eval=gpu_with_eval,
                            train_num=train_num)


class TrainDamage(DeepLabDPC):
    def __init__(self):
        train_name = 'damage'
        dataset = 'damage_pascal_voc_seg'
        dataset_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/dataset-damage/record'
        eval_vis_crop_size = [1920, 1920]
        train_num = 1000000
        batch_size = 2
        train_crop_size = [513, 513]
        gpu_with_train = '0,1'
        gpu_with_eval = '0,1'
        pretrain_checkpoint = '/home/ubuntu/zsf/zhousf/tf_project/my_models/deeplabv3_dpc_cityscapes_trainfine/damage/train/model.ckpt-389185'
        DeepLabDPC.__init__(self,
                            train_name,
                            dataset=dataset,
                            dataset_dir=dataset_dir,
                            pretrain_checkpoint=pretrain_checkpoint,
                            eval_vis_crop_size=eval_vis_crop_size,
                            batch_size=batch_size,
                            train_crop_size=train_crop_size,
                            gpu_with_train=gpu_with_train,
                            gpu_with_eval=gpu_with_eval,
                            train_num=train_num)
