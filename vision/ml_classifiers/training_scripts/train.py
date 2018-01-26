'''Train an ML model to detect the presence of an elephant in
an image/video'''

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from jobs import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("recipe", help="Specify the ML Recipe to use - dt, rf, svm supported.")
parser.add_argument("dim", help="Convert the images into specifed dimension. 32,64,128 supported.")
parser.add_argument(
    "-g", "--gray", help="Convert images to grayscale instead of RGB", action="store_true")
args = parser.parse_args()
use_rgb = True
if args.gray:
    use_rgb = False
dim = int(args.dim)
recipe = args.recipe

print "\nImage Type:", ["Grayscale", "RGB"][int(use_rgb)]
print "Image dimension:", "{}x{}".format(dim, dim)
print "ML Recipe:",
if recipe == "rf":
    print "Random Forest Classifier"
elif recipe == "dt":
    print "Decision Tree Classifier"
elif recipe == "svm":
    print "Support Vector Machine"
else:
    print "Unknown Recipe"
    exit()

# load the images
e = load_image_dataset("../datasets/elephants/", dim, use_rgb)
f = load_image_dataset("../datasets/timelapse-forest/", dim, use_rgb)

# make the labels
labels = [1] * len(e) + [0] * len(f)

# combine the datasets
features = e + f

# split the datatset
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.9)

if recipe == "svm":
    clf = SVC(gamma='auto', C=10)
if recipe == "rf":
    clf = RandomForestClassifier()
if recipe == "dt":
    clf = DecisionTreeClassifier()

# train
print "Training.."
clf.fit(train_x, train_y)
preds = clf.predict(test_x)

# file name must of this format, because other programs have hard coded dependencies
save(clf, "{}-classify-{}".format(recipe, dim))

print "Confusion Matrix:"
print confusion_matrix(test_y, preds)
cval = cross_val_score(clf, features, labels, cv=10)
print "Cross val avg: ", sum(cval) / len(cval)
print "Accuracy:", accuracy_score(test_y, preds)
print "Classifier snapshot:"
print clf
