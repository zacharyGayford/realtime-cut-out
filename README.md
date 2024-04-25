# Realtime Cut-Out

## How To Build
1. Download and install [Anaconda](https://anaconda.org/)
2. Clone this repository locally and navigate to its folder.
3. Create and activate the conda environment
```
conda create -n FastSAM
conda activate FastSAM
pip install -r requirements.txt
```
5. Run the script `python main.py`

## Overview
This project was created as an introduction to realtime video in machine learning. The model takes a video input from a camera and detects the object in the foreground using [YOLOv8](https://github.com/ultralytics/ultralytics). The bounding box and object data from YOLOv8 is passed to [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM). SAM stands for _S_egment _A_nything _M_odel. This model takes in image data and possibly a bounding box and create a segmentation mask for the whole image. This segmentation mask is a new image that outlines every object from the original image. If a bounding box is provided, like in this case, the model segments the foreground of whatever is in the bounding box. The segmentation mask can be used, with a bit of image processing, to gain a transparent image of the subject separate from the background, a cut-out.

## Shortcommings
This model does not perform well on devices that do not have discrete graphics cards. This might be aleviated by using some of the smaller and more performant versions of the FastSAM model. These can speed up performance but come at the cost of reduced accuracy in segmentation.
