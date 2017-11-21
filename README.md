## Gaze Detection with OpenCV (IN-PROGRESS)
currently working on pupil tracking using python and OpenCV.
## Overview
This code roughly detects pupils using low-budget webcams.
Code uses access to webcam to draw red dots on pupils.
First I use a [Guassian blur](https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html) 
and [CLAHE](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) 
Histogram equalization to fix background noise and uneven lighting.
Next, I use a [Haar Cascade](https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html)
to locate general location of the eyes and face.
Finally, I use [thresholding](https://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html)
to detect the lowest brightness in the eye region (commonly the pupil).
### Dependencies:
* OpenCV
* numpy
### Required Files:
* haarcascade_frontalface_default.xml
* haarcascade_eye.xml <br />
NOTE: Found on [opencv](https://github.com/opencv/opencv/tree/master/data/haarcascades)'s github repo
### TODO:
* Further Limit Haar Cascade search area by recognizing sclera of eye
* Gaze Detection
