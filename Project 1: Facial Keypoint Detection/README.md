## Abstract:
Following up with the instructions from [Udacity README](README_Udacity.md), this project aims to leverage knowledge of computer vision techniques and deep learning architectures to build a full pipeline of facial keypoint detections.

## Computer Vision Pipeline:
A computer vision pipeline is a series of steps that most computer vision applications will go through. Many vision applications start off by acquiring images and data, then processing that data, performing some analysis and recognition steps, then finally performing an action.
![img](images/cv_pipeline.png)

## Workflow:
After we’ve gone through the previous step to train a neural network to detect facial features, we’ll have the network to apply to any image that includes faces.

Based on the [NaimishNet paper](https://github.com/jonathanyeh0723/Udacity-CVND-Projects/blob/master/Project%201:%20Facial%20Keypoint%20Detection/1710.00977.pdf) addressed, a CNN architecture was built for the implementation of facial key points detector.

By leveraging CascadeClassifier and fine-tune the parameter, we shall be able to find the region of interest for our selected image. Further, to help enhance the quality of data to promote the extraction of meaningful insights, we then perform data preprocessing. Concretely, these steps are:

- Convert the image from RGB to grayscale.
- Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255].
- Rescale the detected face to be the expected square size of our CNN (224x224 as suggested).
- Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W).
- Convert images to FloatTensors.

After that, we’re ready to make facial keypoint predictions using our pre-trained network. Finally, we display each detected face and the corresponding keypoints to check the results.

## Results:

[This notebook](https://github.com/jonathanyeh0723/Udacity-CVND-Projects/blob/master/Project%201:%20Facial%20Keypoint%20Detection/3.%20Facial%20Keypoint%20Detection%2C%20Complete%20Pipeline.ipynb) contains the full pipeline: face detection, extraction and preprocessing, and keypoint prediction, as well as the final results.

The picture shown below is the detected face area with bounding boxes drawn:

![img](images/image_with_detections_beatles_1.png)

After keypoint prediction, using the loaded model we've already trained, eventually we get the facial keypoints!

![img](images/image_with_detections_beatles_2.png)

It is observed that the shift of the keypoints. They are so-called errors. The result can be improved by applying batch normalization as well as transfer learning.

