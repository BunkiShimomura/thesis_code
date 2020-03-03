
import cv2
import numpy as np


def masked_img(src, msk):
    height, width, color = src.shape # get shape
    dst = np.zeros((height, width, 3), dtype = "uint8") # generate base image for new image

    for y in range(0, height):
        for x in range(0, width):
            if (msk[y][x] > 240).all():
                dst[y][x] = src[y][x]
            else:
                dst[y][x] = 0

    while 1:
    	cv2.imshow("dst", dst) # 画像を表示

    	k = cv2.waitKey(1) # キー入力
    	if k == ord('q'): # もしQが押されたら終了する
    		break
    cv2.destroyAllWindows() # 終了処理

    return dst
