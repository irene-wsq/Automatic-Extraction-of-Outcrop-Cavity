
## Installation
From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download `mask_rcnn_balloon.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download `balloon_dataset.zip`. Expand it such that it's in the path `mask_rcnn/datasets/balloon/`.


## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
```

The code in `balloon.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.
