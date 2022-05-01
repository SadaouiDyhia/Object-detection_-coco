from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.python.saved_model import tag_constants
import colorsys
import random
import time
from multiprocessing import Process, Queue, Pipe
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


######################### yolov3.configs ############################
# YOLO options
YOLO_TYPE = "yolov3"  # yolov4 or yolov3
YOLO_FRAMEWORK = "tf"  # "tf" or "trt"
YOLO_V3_WEIGHTS = "model_data/yolov3.weights"
YOLO_V4_WEIGHTS = "model_data/yolov4.weights"
YOLO_V3_TINY_WEIGHTS = "model_data/yolov3-tiny.weights"
YOLO_V4_TINY_WEIGHTS = "model_data/yolov4-tiny.weights"
YOLO_TRT_QUANTIZE_MODE = "INT8"  # INT8, FP16, FP32
# "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
YOLO_CUSTOM_WEIGHTS = False
# YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_COCO_CLASSES = "model_data/coco/coco.names"
YOLO_STRIDES = [8, 16, 32]
YOLO_IOU_LOSS_THRESH = 0.5
YOLO_ANCHOR_PER_SCALE = 3
YOLO_MAX_BBOX_PER_SCALE = 100
YOLO_INPUT_SIZE = 416
if YOLO_TYPE == "yolov4":
    YOLO_ANCHORS = [[[12,  16], [19,   36], [40,   28]],
                    [[36,  75], [76,   55], [72,  146]],
                    [[142, 110], [192, 243], [459, 401]]]
if YOLO_TYPE == "yolov3":
    YOLO_ANCHORS = [[[10,  13], [16,   30], [33,   23]],
                    [[30,  61], [62,   45], [59,  119]],
                    [[116, 90], [156, 198], [373, 326]]]
# Train options
TRAIN_YOLO_TINY = True
# saves only best model according validation loss (True recommended)
TRAIN_SAVE_BEST_ONLY = True
# saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_SAVE_CHECKPOINT = False
TRAIN_CLASSES = "mnist/mnist.names"
TRAIN_ANNOT_PATH = "mnist/mnist_train.txt"
TRAIN_LOGDIR = "log"
TRAIN_CHECKPOINTS_FOLDER = "checkpoints"
TRAIN_MODEL_NAME = f"{YOLO_TYPE}_custom"
TRAIN_LOAD_IMAGES_TO_RAM = True  # With True faster training, but need more RAM
TRAIN_BATCH_SIZE = 4
TRAIN_INPUT_SIZE = 416
TRAIN_DATA_AUG = True
TRAIN_TRANSFER = True
TRAIN_FROM_CHECKPOINT = False  # "checkpoints/yolov3_custom"
TRAIN_LR_INIT = 1e-4
TRAIN_LR_END = 1e-6
TRAIN_WARMUP_EPOCHS = 2
TRAIN_EPOCHS = 100

# TEST options
TEST_ANNOT_PATH = "mnist/mnist_test.txt"
TEST_BATCH_SIZE = 4
TEST_INPUT_SIZE = 416
TEST_DATA_AUG = False
TEST_DECTECTED_IMAGE_PATH = ""
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.45

if TRAIN_YOLO_TINY:
    YOLO_STRIDES = [16, 32]
    # YOLO_ANCHORS            = [[[23, 27],  [37, 58],   [81,  82]], # this line can be uncommented for default coco weights
    YOLO_ANCHORS = [[[10, 14],  [23, 27],   [37, 58]],
                    [[81,  82], [135, 169], [344, 319]]]


####################### yolov3.yolov4 ###############################

STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T/STRIDES).T


def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


class BatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = LeakyReLU(alpha=0.1)(conv)
        elif activate_type == "mish":
            conv = mish(conv)

    return conv


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(
        1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv, filters_shape=(
        3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def darknet53(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)

    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


def cspdarknet53(input_data):
    input_data = convolutional(
        input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = convolutional(
        input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = residual_block(
            input_data,  64,  32, 64, activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(
        input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(
        input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = residual_block(
            input_data, 64,  64, 64, activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(
        input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = convolutional(
        input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = residual_block(
            input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(
        input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = convolutional(
        input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = residual_block(
            input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(
        input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = convolutional(
        input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = residual_block(
            input_data, 512, 512, 512, activate_type="mish")
    input_data = convolutional(
        input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(
        input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    max_pooling_1 = tf.keras.layers.MaxPool2D(
        pool_size=13, padding='SAME', strides=1)(input_data)
    max_pooling_2 = tf.keras.layers.MaxPool2D(
        pool_size=9, padding='SAME', strides=1)(input_data)
    max_pooling_3 = tf.keras.layers.MaxPool2D(
        pool_size=5, padding='SAME', strides=1)(input_data)
    input_data = tf.concat(
        [max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)

    input_data = convolutional(input_data, (1, 1, 2048, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data


def darknet19_tiny(input_data):
    input_data = convolutional(input_data, (3, 3, 3, 16))
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 16, 32))
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 32, 64))
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 64, 128))
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 128, 256))
    route_1 = input_data
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 256, 512))
    input_data = MaxPool2D(2, 1, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 512, 1024))

    return route_1, input_data


def cspdarknet53_tiny(input_data):  # not sure how this should be called
    input_data = convolutional(input_data, (3, 3, 3, 32), downsample=True)
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True)
    input_data = convolutional(input_data, (3, 3, 64, 64))

    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 64, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 256))
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 512, 512))

    return route_1, input_data


def YOLOv3(input_layer, NUM_CLASS):
    # After the input layer enters the Darknet-53 network, we get three branches
    route_1, route_2, conv = darknet53(input_layer)
    # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))

    # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255]
    conv_lbbox = convolutional(
        conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1,  512,  256))
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))

    # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
    conv_mbbox = convolutional(
        conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))

    # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
    conv_sbbox = convolutional(
        conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, conv = cspdarknet53(input_layer)

    route = conv
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = upsample(conv)
    route_2 = convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    route_1 = convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(
        conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(
        conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(
        conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv3_tiny(input_layer, NUM_CLASS):
    # After the input layer enters the Darknet-53 network, we get three branches
    route_1, conv = darknet19_tiny(input_layer)

    conv = convolutional(conv, (1, 1, 1024, 256))
    conv_lobj_branch = convolutional(conv, (3, 3, 256, 512))

    # conv_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
    conv_lbbox = convolutional(
        conv_lobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv_mobj_branch = convolutional(conv, (3, 3, 128, 256))
    # conv_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
    conv_mbbox = convolutional(
        conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]


def YOLOv4_tiny(input_layer, NUM_CLASS):
    route_1, conv = cspdarknet53_tiny(input_layer)

    conv = convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = convolutional(
        conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = convolutional(
        conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]


def Create_Yolo(input_size=416, channels=3, training=False, CLASSES=YOLO_COCO_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    input_layer = Input([input_size, input_size, channels])

    if TRAIN_YOLO_TINY:
        if YOLO_TYPE == "yolov4":
            conv_tensors = YOLOv4_tiny(input_layer, NUM_CLASS)
        if YOLO_TYPE == "yolov3":
            conv_tensors = YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if YOLO_TYPE == "yolov4":
            conv_tensors = YOLOv4(input_layer, NUM_CLASS)
        if YOLO_TYPE == "yolov3":
            conv_tensors = YOLOv3(input_layer, NUM_CLASS)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training:
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    Yolo = tf.keras.Model(input_layer, output_tensors)
    return Yolo


def decode(conv_output, NUM_CLASS, i=0):
    # where i = 0, 1 or 2 to correspond to the three grid scales
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position
    # conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
    # conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
    # conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52
    #y = tf.range(output_size, dtype=tf.int32)
    #y = tf.expand_dims(y, -1)
    #y = tf.tile(y, [1, output_size])
    #x = tf.range(output_size,dtype=tf.int32)
    #x = tf.expand_dims(x, 0)
    #x = tf.tile(x, [output_size, 1])
    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1),
                             axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0),
                      [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    #xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    #xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    #y_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    # object box calculates the predicted confidence
    pred_conf = tf.sigmoid(conv_raw_conf)
    # calculating the predicted probability category box object
    pred_prob = tf.sigmoid(conv_raw_prob)

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

# testing (should be better than giou)


def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + \
        (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) -
                                     tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term


def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=YOLO_COCO_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size,
                      output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * \
        label_xywh[:, :, :, :, 2:3] * \
        label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                   bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Find the value of IoU with the real box The largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou <
                                                 YOLO_IOU_LOSS_THRESH, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * (
        respond_bbox *
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=respond_bbox, logits=conv_raw_conf)
        +
        respond_bgd *
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * \
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss

################################## Yolov3.utils ####################################


def load_yolo_weights(model, weights_file):
    tf.keras.backend.clear_session()  # used to reset layer names
    # load Darknet original weights to TensorFlow model
    if YOLO_TYPE == "yolov3":
        range1 = 75 if not TRAIN_YOLO_TINY else 13
        range2 = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]
    if YOLO_TYPE == "yolov4":
        range1 = 110 if not TRAIN_YOLO_TINY else 21
        range2 = [93, 101, 109] if not TRAIN_YOLO_TINY else [17, 20]

    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(
            wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' % i
            else:
                conv_layer_name = 'conv2d'

            if j > 0:
                bn_layer_name = 'batch_normalization_%d' % j
            else:
                bn_layer_name = 'batch_normalization'

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'


def Load_Yolo_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f'GPUs {gpus}')
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    if YOLO_FRAMEWORK == "tf":  # TensorFlow detection
        if YOLO_TYPE == "yolov4":
            Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
        if YOLO_TYPE == "yolov3":
            Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

        if YOLO_CUSTOM_WEIGHTS == False:
            print("Loading Darknet_weights from:", Darknet_weights)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE,
                               CLASSES=YOLO_COCO_CLASSES)
            load_yolo_weights(yolo, Darknet_weights)  # use Darknet weights
        else:
            checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
            if TRAIN_YOLO_TINY:
                checkpoint += "_Tiny"
            print("Loading custom weights from:", checkpoint)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE,
                               CLASSES=TRAIN_CLASSES)
            yolo.load_weights(checkpoint)  # use custom weights

    elif YOLO_FRAMEWORK == "trt":  # TensorRT detection
        saved_model_loaded = tf.saved_model.load(
            YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        yolo = saved_model_loaded.signatures['serving_default']

    return yolo


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h,  w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1:
            bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking:
                score_str = " "+str(score)

            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
            except KeyError:
                print(
                    "You received KeyError, this might be that you are trying to use yolo original weights")
                print(
                    "while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 -
                          text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or(
        (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(
        pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and(
        (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def detect_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [
                                  input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(
        pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES,
                      rectangle_colors=rectangle_colors)
    # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

    if output_path != '':
        cv2.imwrite(output_path, image)
    if show:
        # Show the image
        cv2.imshow("predicted image", image)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()

    return image


def Predict_bbox_mp(Frames_data, Predicted_data, Processing_times):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")
    Yolo = Load_Yolo_model()
    times = []
    while True:
        if Frames_data.qsize() > 0:
            image_data = Frames_data.get()
            t1 = time.time()
            Processing_times.put(time.time())

            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1]))
                         for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            Predicted_data.put(pred_bbox)


def postprocess_mp(Predicted_data, original_frames, Processed_frames, Processing_times, input_size, CLASSES, score_threshold, iou_threshold, rectangle_colors, realtime):
    times = []
    while True:
        if Predicted_data.qsize() > 0:
            pred_bbox = Predicted_data.get()
            if realtime:
                while original_frames.qsize() > 1:
                    original_image = original_frames.get()
            else:
                original_image = original_frames.get()

            bboxes = postprocess_boxes(
                pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')
            image = draw_bbox(original_image, bboxes,
                              CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            times.append(time.time()-Processing_times.get())
            times = times[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            image = cv2.putText(image, "Time: {:.1f}FPS".format(
                fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            #print("Time: {:.2f}ms, Final FPS: {:.1f}".format(ms, fps))

            Processed_frames.put(image)


def Show_Image_mp(Processed_frames, show, Final_frames):
    while True:
        if Processed_frames.qsize() > 0:
            image = Processed_frames.get()
            Final_frames.put(image)
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

# detect from webcam


def detect_video_realtime_mp(video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', realtime=False):
    if realtime:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    # output_path must be .mp4
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    no_of_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    original_frames = Queue()
    Frames_data = Queue()
    Predicted_data = Queue()
    Processed_frames = Queue()
    Processing_times = Queue()
    Final_frames = Queue()

    p1 = Process(target=Predict_bbox_mp, args=(
        Frames_data, Predicted_data, Processing_times))
    p2 = Process(target=postprocess_mp, args=(Predicted_data, original_frames, Processed_frames,
                 Processing_times, input_size, CLASSES, score_threshold, iou_threshold, rectangle_colors, realtime))
    p3 = Process(target=Show_Image_mp, args=(
        Processed_frames, show, Final_frames))
    p1.start()
    p2.start()
    p3.start()

    while True:
        ret, img = vid.read()
        if not ret:
            break

        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_frames.put(original_image)

        image_data = image_preprocess(np.copy(original_image), [
                                      input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        Frames_data.put(image_data)

    while True:
        if original_frames.qsize() == 0 and Frames_data.qsize() == 0 and Predicted_data.qsize() == 0 and Processed_frames.qsize() == 0 and Processing_times.qsize() == 0 and Final_frames.qsize() == 0:
            p1.terminate()
            p2.terminate()
            p3.terminate()
            break
        elif Final_frames.qsize() > 0:
            image = Final_frames.get()
            if output_path != '':
                out.write(image)

    cv2.destroyAllWindows()


def detect_video(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times, times_2 = [], []
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    # output_path must be .mp4
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while True:
        _, img = vid.read()

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_image), [
                                      input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(
            pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        image = draw_bbox(original_image, bboxes, CLASSES=CLASSES,
                          rectangle_colors=rectangle_colors)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        image = cv2.putText(image, "Time: {:.1f}FPS".format(
            fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(
            ms, fps, fps2))
        if output_path != '':
            out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

# detect from webcam


def detect_realtime(Yolo, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times = []
    vid = cv2.VideoCapture(0)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    # output_path must be .mp4
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        image_data = image_preprocess(np.copy(original_frame), [
                                      input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(
            pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        times.append(t2-t1)
        times = times[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms

        print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

        frame = draw_bbox(original_frame, bboxes, CLASSES=CLASSES,
                          rectangle_colors=rectangle_colors)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_frame, bboxes, read_class_names(CLASSES))
        image = cv2.putText(frame, "Time: {:.1f}FPS".format(fps), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if output_path != '':
            out.write(frame)
        if show:
            cv2.imshow('output', frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()


image_path = "./IMAGES/cat.webp"

yolo = Load_Yolo_model()
detect_image(yolo, image_path, "./IMAGES/cat.jpg",
             input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))
