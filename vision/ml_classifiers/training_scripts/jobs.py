from PIL import Image
from tensorflow.contrib.keras.python.keras.preprocessing import image as kimg
import cPickle as p
import numpy as np
import os


def load_image_dataset(filepath, DIM, rgb=None):
    dataset = []
    global IMGTYPE
    if rgb:
        IMGTYPE = "rgb"
    else:
        IMGTYPE = "gray"
    for f in os.listdir(filepath):
        if not f.startswith("."):
            try:
                if rgb:
                    #img = np.array(Image.open(filepath + f).resize((DIM, DIM))).flatten()
                    image_ = kimg.load_img(filepath + f, target_size=(DIM, DIM))
                    f = kimg.img_to_array(image_).flatten()
                    if f.shape == (DIM * DIM * 3,):
                        dataset.append(f)
                else:
                    image_ = kimg.load_img(filepath + f, target_size=(DIM, DIM), grayscale=True)
                    f = kimg.img_to_array(image_).flatten()
                    if f.shape == (DIM * DIM,):
                        dataset.append(f)
            except Exception as e:
                pass
    return dataset


def save(clf, name):
    name = "../../models/" + name + "-" + IMGTYPE + ".model"
    with open(name, 'wb') as fp:
        p.dump(clf, fp)
    print "saved model: {}".format(name)
