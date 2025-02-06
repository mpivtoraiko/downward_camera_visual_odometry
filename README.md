# Simple Visual Odometry with a Downward Camera

## Overview
This repository features an implementation of monocular visual odometry with a downward camera on a mobile robot. The only dependency is OpenCV. To run the code, first:

```
pip install opencv-python
```

then run:

```
python src/demo.py
```

You will see two windows: 
- The bigger one is the overhead view of the world with the robot's camera overlaid as a green frame in the center,
- The smaller one is a view of what the robot's camera sees.

You may use the arrow keys to navigate the robot around its world: 
- Up and down keys move the robot forward and back, resp.,
- Left and right keys rotate the robot (diff drive style).

Red line is the estimate of the robot's motion, and the blue line is the ground truth. 

Hit `q` to quit. Enjoy!

## Discussion

Because the camera looks at the ground (which we assume here to be flat), we use sparse optical flow to track features (Lukas-Kanade). As such, it is not quite as robust as more sophisticated methods, such as Structure from Motion involving the Essential matrix computation, but should be faster. 

Further improvement would be doing a better job estimating rotations from optical flow. The provided groundplane texture image of gravel is low on contrast, and we currently lose quite a lot of features during the rotation. As such, OpenCV's affine matrix estimator isn't able to resolve rotations accurately. It would be interesting to investigate if higher-textured surfaces would yield better results, or further tuning of the affine estimator (or a more sophisticated method) would work better.  