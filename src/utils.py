import numpy as np
from enum import IntEnum


class Dims(IntEnum):
    X = 0
    Y = 1
    Z = 2
    W = 3


def ang_norm(angle):
    two_pi = 2 * np.pi
    while angle > np.pi:
        angle -= two_pi
    while angle < -np.pi:
        angle += two_pi
    return angle
