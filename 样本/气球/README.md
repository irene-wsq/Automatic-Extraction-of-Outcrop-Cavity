
## Installation
From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download `mask_rcnn_balloon.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Create "datasets" folder in the root directory, and then create two new folders "tarin" and "val" in it.
3. Open VGg image annotator.zip and mark the cavity position in the picture with the annotation software.Put the pictures of the training set and the corresponding JSON into the train folder, and put the pictures of the verification set and the corresponding JSON into the Val folder.
4. 
## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
```

