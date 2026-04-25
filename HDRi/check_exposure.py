import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np


def channelHistogramCheck(ch_img):
    hs = 64
    hist = cv2.calcHist([ch_img], [0], None, [hs], [0, 256])
    underexposure = False
    overexposure = False
    #print(f"low: {hist[0]}, high: {hist[hs-1]}")
    if hist[0] > 0:
        underexposure = True
    if hist[hs-1] > 0:
        overexposure = True
    return underexposure, overexposure


def histogramCheck(img):
    rgb_img = cv2.imread(img)
    b, g, r = cv2.split(rgb_img)

    ret_code = 0

    for ch in [b, g, r]:
        u, o = channelHistogramCheck(ch)
        if u:
            ret_code |= 1
        if o:
            ret_code |= 2
        # If the image is both over and under exposed, no more checks are needed
        if ret_code == 3:
            break;
    return ret_code


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the image")
    args = vars(ap.parse_args())

    print(histogramCheck(args["image"]))

# 0 - ok
# 1 - underexposed
# 2 - overexposed
# 3 - underexposed and overexposed
