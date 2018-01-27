<h1>detectorPiv2.py</h1>

<h3>Optimized for Python 2.7</h3>

<h4>Dependencies:</h4>
* numpy
* scipy
* scikit-learn
* TensorFlow
* argparse
* cPickle
* OpenCV
* PIL

**Usage:**
	* `python detectorPiv2.py test_img.jpg svm-classify-64-gray.model`
	* `python detectorPiv2.py test_video.mp4 svm-classify-64-gray.model`
	* `python detectorPiv2.py directory_of_imgs/ svm-classify-64-gray.model`
	
The second positional argument is the name of the ML model used for the first stage of detection. All models must be stored in the models/ directory.

Optional arguments:
	--cnn      Use the CNN directly, skipping the first stage of detection (ML model).
	--irr         Only use the ML model, skip the CNN.




The processing time was about 2 seconds per frame, if an elephant was present.
For a frame with no irregularities, the processing time was less than 0.5 seconds.

Note:
	A 5V 2Amp power supply is required for running.
	The threshold for an elephant detection is set to 10% confidence.
	All processed images are stored in the out/ directory.
