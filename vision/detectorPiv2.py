'''
A pipelined approach to detect the presence of an elephant in a video feed.
The pipeline is as follows:

Image -> ML Model -> CNN -> Result

The raw image is fed into an ML model which determines the presence of
any irregularities in it. If there is an irregularity, the image is
further passed on to the Convolutional Neural Network for confirmation
if the irregegularity is an elephant. This approach reduces overhead on the
computational device used.
'''

print "Initial startup time is ~2mins"

from utils import *
import argparse
import utils
import time

pi = True
try:
    import RPi.GPIO as GPIO
    from picamera import PiCamera as pc
    from picamera.array import PiRGBArray as pcrgb
except:
    pi = False


if pi:
    led = 8
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(led, GPIO.OUT, initial=0)


parser = argparse.ArgumentParser()
parser.add_argument(
    "data", help="The image, video or directory of images to perform recognition on.")
parser.add_argument(
    "model", help="The model to be used to check for irregularities.")
parser.add_argument(
    "--cnn", help="Use the CNN directly (increases overhead on the computer)", action="store_true")
parser.add_argument(
    "--irr", help="Only check for irregularities", action="store_true")

args = parser.parse_args()
skip_irr = False
skip_cnn = False
if args.cnn:
    skip_irr = True
if args.irr:
    skip_cnn = True
data = args.data
model = args.model
utils.dim, utils.imgtype = model.split("-")[2:]
utils.dim = int(utils.dim)


def load(model):
    name = "models/" + model
    with open(name, 'rb') as fp:
        clf = p.load(fp)
    return clf


cnn_model = 'models/ssd_mobilenet_v1_coco_11_06_2017.pb'
labels_path = 'cnn/object_detection/data/mscoco_label_map.pbtxt'


filetype = process(data)
alert = 0

if not skip_cnn:
    print "Prepping CNN"
    g, category_index = prepareCNN(cnn_model, labels_path)
    sess = tf.Session(graph=g)
    image_tensor = g.get_tensor_by_name('image_tensor:0')
    detection_boxes = g.get_tensor_by_name('detection_boxes:0')
    detection_scores = g.get_tensor_by_name('detection_scores:0')
    detection_classes = g.get_tensor_by_name('detection_classes:0')
    num_detections = g.get_tensor_by_name('num_detections:0')
    packed = [image_tensor, detection_boxes,
              detection_scores, detection_classes, num_detections]
    print "CNN loaded"


if not skip_irr:
    print "Prepping ML model"
    clf = load(model)
    print "ML Model loaded"

if filetype == "IMG":
    items = [data]

if filetype == "DIR":
    files = os.listdir(data)
    for i in range(len(files)):
        if files[i].startswith("."):
            files[i] = ""
    unique = list(set(files))
    unique.remove('')
    items = [data + f for f in unique]


if filetype in ["IMG", "DIR"]:
    c = 0
    print "Running detection"
    for item in items:
        t = time.time()
        alert = 0
        if not skip_irr:
            result = if_irregular_file(item, clf)
            if result:
                alert = 1

        if (not skip_cnn and alert == 1) or (not skip_cnn and skip_irr):
            image_ = kimg.load_img(item)
            img_np = kimg.img_to_array(image_)
            res = perform_detection(img_np, sess, packed, category_index)
            if res:
                alert = 2
                cv2.imwrite("out/res_" + str(c) + ".jpg", cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        if pi:
            if alert == 2:
                GPIO.output(led, GPIO.HIGH)
            else:
                GPIO.output(led, GPIO.LOW)
        print c
        print item, "= Prediction:", ["Safe", "Irregularity", "Elephant!"][alert]
        c += 1
        dt = time.time()
        print "lolol", dt - t
    if not skip_cnn:
        sess.close()

if filetype == "VID" and data == "picam":
    from picamera import PiCamera as pc
    from picamera.array import PiRGBArray as pcrgb
    camera = pc()
    camera.framerate = 32
    rawCapture = pcrgb(camera)

    c = 1
    print "Initializing PiCamera"
    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
        alert = 0
        img_np = frame.array
        if not skip_irr:
            result = if_irregular_array(img_np, clf)
            if result:
                alert = 1

        if (not skip_cnn and alert == 1) or (not skip_cnn and skip_irr):
            res = perform_detection(img_np, sess, packed, category_index)
            if res:
                alert = 2
                cv2.imwrite("out/res_" + str(c) + ".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        print c
        if pi:
            if alert == 2:
                GPIO.output(led, GPIO.HIGH)
            else:
                GPIO.output(led, GPIO.LOW)
        print "frame" + str(c), "= Prediction:", ["Safe", "Irregularity", "Elephant!"][alert]
        rawCapture.truncate(0)
        c += 1
    sess.close()

if filetype == "VID" and data != "cam":
    cap = cv2.VideoCapture(data)
    c = 1
    while True:
        t = time.time()
        _, item = cap.read()

        if _:
            item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            alert = 0
            if not skip_irr:
                result = if_irregular_array(item, clf)
                if result:
                    alert = 1

            if (not skip_cnn and alert == 1) or (not skip_cnn and skip_irr):
                res = perform_detection(item, sess, packed, category_index)
                if res:
                    alert = 2
                    cv2.imwrite("out/res_" + str(c) + ".jpg", cv2.cvtColor(item, cv2.COLOR_RGB2BGR))

            print c
            if pi:
                if alert == 2:
                    GPIO.output(led, GPIO.HIGH)
                else:
                    GPIO.output(led, GPIO.LOW)
            print "frame" + str(c), "= Prediction:", ["Safe", "Irregularity", "Elephant!"][alert]
            c += 1
            dt = time.time()
            print "that took", dt - t
        else:
            print "no frame read"
            break

    sess.close()
    cap.release()
