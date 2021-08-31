import cv2
import numpy as np


def get_depth(dataset, file):
    depth = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth = cv2.resize(depth, tuple(dataset.labels_size), interpolation=cv2.INTER_NEAREST)
    depth = 65536.0 / (depth + 1)  # inverse depth
    return depth


def get_depth_cityscapes(dataset, file):
    depth = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256. + 1.
    depth = cv2.resize(depth, tuple(dataset.labels_size), interpolation=cv2.INTER_NEAREST)
    return depth
