#############################################################################
# Image matching for 3D printing to detect failures
#
# Description: Images are loaded from disk, applied thresholding and
#   computed hu moments to calculate the difference between them
#
# References:
#   - Computer Vision Course 2020, Laboratory 5, from Technical University
#       Iasi, Faculty of Automatic Control and Computer Engineering
#############################################################################

import cv2
import os
import glob
import sys

import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt


IMG_SLICER = 0
IMG_PRINTED = 1


def draw_image(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 600)
    cv2.imshow(name, image)
    #cv2.imwrite("images/export/" + name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def filter_slicer_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Range for lower red
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper red
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2

    return mask


def filter_printer_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filtered_image = cv2.medianBlur(gray, 11)
    _, im = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((11, 11), np.uint8)
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    return opening


def calculate_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)

    return hu_moments


def compare_moments(moment_1, moment_2):
    difference = distance.euclidean(moment_1, moment_2)

    return difference



def main():
    slicer_image = cv2.imread(sys.argv[1])
    target_image = cv2.imread(sys.argv[2])
    filtered_image = filter_slicer_image(slicer_image)
    sl_hu_moments = calculate_hu_moments(filtered_image)
    cv2.imwrite(sys.argv[3], filtered_image)
    draw_image(filtered_image, sys.argv[1])

    filtered_image = filter_printer_image(target_image)
    tg_hu_moments = calculate_hu_moments(filtered_image)
    cv2.imwrite(sys.argv[4], filtered_image)
    draw_image(filtered_image, sys.argv[2])

    diff = compare_moments(sl_hu_moments, tg_hu_moments)
    print("Similarity(" + sys.argv[1] + ", " + sys.argv[2] + " = " + str(diff) + ")")


if __name__== "__main__":
    main()
