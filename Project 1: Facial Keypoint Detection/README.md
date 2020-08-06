## Abstract:
Following up with the instructions from [Udacity README](README_Udacity.md), this project aims to leverage knowledge of computer vision techniques and deep learning architectures to build a full pipeline of facial keypoint detections.

## Computer Vision Pipeline:
A computer vision pipeline is a series of steps that most computer vision applications will go through. Many vision applications start off by acquiring images and data, then processing that data, performing some analysis and recognition steps, then finally performing an action.
![img](images/cv_pipeline.png)

## Workflow:

For face detection, the OpenCV function CascadeClassifier::detectMultiScale was used with an OpenCV XML file encoding a Haar Cascade classifier.

For keypoint detection, a CNN architecture was chosen, similar to that described in the [NaimishNet paper](https://github.com/jonathanyeh0723/Udacity-CVND-Projects/blob/master/Project%201:%20Facial%20Keypoint%20Detection/1710.00977.pdf).

## Results:

[This notebook](https://github.com/jonathanyeh0723/Udacity-CVND-Projects/blob/master/Project%201:%20Facial%20Keypoint%20Detection/3.%20Facial%20Keypoint%20Detection%2C%20Complete%20Pipeline.ipynb) contains the full pipeline: face detection, extraction and preprocessing, and keypoint prediction, as well as the final results.

After face detection, with bounding boxes drawn around the detected faces:

![img](images/image_with_detections_beatles_1.png)

After keypoint prediction, using the trained "naimishlite" model:

![img](images/image_with_detections_beatles_2.png)

Placement of keypoints on some of the faces is suboptimal due to limited training time and data.

