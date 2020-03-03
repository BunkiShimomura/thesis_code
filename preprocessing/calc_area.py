import cv2
import numpy as np
from PIL import ImageGrab, Image

def calc_area(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_gaus = cv2.GaussianBlur(hsv, (5, 5), 0) #ぼかす
    hsv_min = np.array([30, 80, 60])
    hsv_max = np.array([255, 255, 255])
    mask = cv2.inRange(hsv_gaus, hsv_min, hsv_max)
    im_list = np.asarray(mask)
    kernel = np.ones((7,7), np.uint8) #ノイズの除去
    result = cv2.morphologyEx(im_list, cv2.MORPH_OPEN, kernel) #オブジェクト背後の細かいノイズを除去
    contours, heirarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # contoursは輪郭のリスト

    areaList = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 2: #輪郭部分の配列は2になる。輪郭しか持たないものを排除。ifはあってもなくても良い
            areaList.append(area) #輪郭を除いた葉の面積を加算する

    return (sum(areaList), result)
