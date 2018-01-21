from cnn.object_detection.utils import visualization_utils as vu
from tensorflow.contrib.keras.python.keras.preprocessing import image as kimg
from cnn.object_detection.utils import label_map_util as lmu
from os.path import isdir
import tensorflow as tf
from PIL import Image
import cPickle as p
import numpy as np
import cv2
import os

dim = None
imgtype = None


def if_elephant(scores, classes):
    objects = []
    for (score, cla) in zip(scores, classes):
        if score > 0.1:
            objects.append(cla)
    return 22 in objects


def if_irregular_array(data, clf):
    if imgtype.startswith("rgb"):
        f = cv2.resize(data, (dim, dim)).flatten()
        #f = np.array(Image.fromarray(data).resize((dim, dim))).flatten()
    else:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        f = cv2.resize(data, (dim, dim)).flatten()
        #f = np.array(Image.fromarray(data).convert('1').resize((dim, dim))).flatten()
    pred = clf.predict([f])
    return pred[0]


def if_irregular_file(data, clf):
    if imgtype.startswith("rgb"):
        image_ = kimg.load_img(data, target_size=(dim, dim))
        f = kimg.img_to_array(image_).flatten()
        #f = np.array(Image.open(data).resize((dim, dim))).flatten()
    else:
        image_ = kimg.load_img(data, target_size=(dim, dim), grayscale=True)
        f = kimg.img_to_array(image_).flatten()
        print f
        #f = np.array(Image.open(data).convert('1').resize((dim, dim))).flatten()
    pred = clf.predict([f])
    return pred[0]


def prepareCNN(cnn_model, labels_path):
    g = tf.Graph()
    with g.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(cnn_model, 'rb') as fid:
            serialized = fid.read()
            od_graph_def.ParseFromString(serialized)
            tf.import_graph_def(od_graph_def, name='')

    # prepare the label and category database
    label_map = lmu.load_labelmap(labels_path)
    categories = lmu.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = lmu.create_category_index(categories)
    return g, category_index


def process(data):
    if isImage(data):
        return "IMG"
    if isVideo(data):
        return "VID"
    if isdir(data):
        return "DIR"


def isImage(file):
    return file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png")


def isVideo(file):
    return file.endswith(".mov") or file.endswith(".mp4") or file.endswith(".mkv") or file.endswith(".avi") or file == "picam"


def to_numpy(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def perform_detection(img_np, sess, packed, category_index):
    image_np_expanded = np.expand_dims(img_np, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [packed[1], packed[2], packed[3], packed[4]],
        feed_dict={packed[0]: image_np_expanded})
    result = if_elephant(scores[0][:5], classes[0][:5])
    if result:
        print np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), num
        vu.visualize_boxes_and_labels_on_image_array(img_np, np.squeeze(boxes), np.squeeze(classes).astype(
            np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=4)
    return int(result)
