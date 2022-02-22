import numpy as np


# image
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

ICUB_OFF = False  # run without icub

# open pose
JOINTS_POSE = [0, 15, 16, 17, 18]
JOINTS_FACE = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 68, 69]
NUM_JOINTS = len(JOINTS_FACE) + len(JOINTS_POSE)



