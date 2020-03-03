# python tool for trimming designated area of photos

import os # module for using path
import glob # module for getting files
import shutil # module for editting files and folders

import cv2

# for confirmation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_files(path, num):
    # create a folder for editted files
    dir = os.path.dirname(path)
    os.mkdir(dir + "/editted")
    new_folder = dir + "/" + "editted"

    # get files, trim, resize, rename, and store
    files = sorted(glob.glob(path + '/*'))
    for i, file in enumerate(files, start = int(num)):
        img = cv2.imread(file, cv2.IMREAD_COLOR)

        #trim images
        trimmed = img[600:1600, 1000:2000]
        cv2.imwrite(new_folder + "/" + str(i) + ".JPEG", trimmed)

if __name__ == '__main__':
    path = input('画像フォルダへの絶対パス: ')
    num = input('通し番号のスタート地点: ')

    get_files(path, num)
