# a programme to judge if root taking was done

import os, sys
from natsort import natsorted
import glob
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image

from create_album import get_files # get files from folder
from reg_brightness import reg_brightness # normalizes brightness
from calc_area import calc_area # returns area of green leaves
from masked_img import masked_img # show only green area
from data_aug import data_aug # data augmentation (flip and rotate)

if __name__ == "__main__":
    path = input("path to foler:")

    # 除外するポットの番号を指定
    delete = [644, 646, 647, 652, 659, 665, 666, 667, 668, 669, 670, 672, 676, 680, 687, 699, 700, 701, 707, 708, 713, 715, 716, 717, 726, 727, 732, 734, 735, 736, 738, 740, 748, 751, 756, 757]
    dir = os.path.dirname(path)
    files = natsorted(glob.glob(path + '/*'))
    os.mkdir(path + '/reg')
    os.mkdir(path + '/bw')
    os.mkdir(path + '/masked')
    os.mkdir(path + '/trimmed')
    os.mkdir(path + '/dataset')
    with open(path + '/area.csv', 'w') as f:
        title = ('ポット番号', '面積')
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(title)
        for i, file in enumerate(files, 721):
            if int(i) in delete:
                pass
            else:
                img = cv2.imread(file)
                # 画像のトリミング
                trimmed_img = img[600:1600, 1000:2000]
                cv2.imwrite(path + '/trimmed/' + str(i) + '_trimmed.png', trimmed_img)
                # 明度の調整
                reg_img = reg_brightness(trimmed_img)
                cv2.imwrite(path + '/reg/' + str(i) + '_reg.png', reg_img)
                # 緑地面積の計算
                area = calc_area(reg_img)
                green_area = area[0]

                # pillowで緑地部分のみの切り抜き画像を作成
                cv2.imwrite(path + '/bw/' + str(i) + '_bw.png', area[1])
                src = Image.open(path + '/reg/' + str(i) + '_reg.png')
                mask = Image.open(path + '/bw/' + str(i) + '_bw.png')
                masked_src = src.copy()
                masked_src.putalpha(mask)
                masked_src.save(path + '/masked/' + str(i) + '_masked.png')

                # データ水増し
                masked_src.save(path + '/dataset/' + str(i) +'_dataset_0.png')
                masked_src.transpose(Image.ROTATE_180).save(path + '/dataset/' + str(i) +'_dataset_1.png')
                masked_src.transpose(Image.FLIP_LEFT_RIGHT).save(path + '/dataset/' + str(i) +'_dataset_2.png')

                writer.writerow((i, green_area))
                '''
                if area[0] >= 11000:
                    cv2.imwrite('/Users/Bunki/Desktop/screened_image/' + str(i) + '.png', reg_img) # save images that have green area of over 11,000
                else:
                    pass
                '''
    f.close()
