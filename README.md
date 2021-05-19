# Mask R-CNN for Automatic-Extraction-of-Outcrop-Cavity
This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.


## Installation
From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download `mask_rcnn_balloon.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Create "datasets" folder in the root directory, and then create two new folders "tarin" and "val" in it.
3. Open VGg image annotator.zip and mark the cavity position in the picture with the annotation software.Put the pictures of the training set and the corresponding JSON into the train folder, and put the pictures of the verification set and the corresponding JSON into the Val folder.

## Train the cavity model

Train a new model starting from pre-trained COCO weights
```
python3 balloon.py train --dataset=/path/to/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 balloon.py train --dataset=/path/to/dataset --weights=last
```

## Model prediction

Put the model from the previous training into the demo_ Test file can predict the location of the cavity.



