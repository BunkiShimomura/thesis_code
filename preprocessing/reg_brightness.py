import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
# normalize brightness to reduce variation
def reg_brightness(img):
    # img = cv2.imread(img)
    img = img
    img = (img - np.mean(img))/np.std(img)*16 + 64 # https://cvtech.cc/std/
    cv2.imshow('image', img)
    cv2.waitKey(0)
    return img
'''

def reg_brightness(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv_img[:, :, (2)]
    value = value * (120/np.mean(value))
    hsv_img[:, :, (2)] = value

    hsv_img = np.asarray(hsv_img)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    return img
