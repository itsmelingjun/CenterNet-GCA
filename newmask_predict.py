# -----------------------------------------------------------------------#
#   预测时通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from newmask_centernet import CenterNet

if __name__ == "__main__":
    centernet = CenterNet()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'predict_mask'      表示预测单张图片的mask。
    #   'dir_mask_predict'  表示遍历文件夹进行检测并保存mask。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完目标框之后图片结果的保存路径
    #   dir_mask_save_path  指定了检测完掩码之后图片结果的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    dir_mask_save_path = "img_mask_out/"
    # -------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   heatmap_save_path仅在mode='heatmap'有效
    # -------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"

    if mode == "predict":
        
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = centernet.detect_image(image, crop=crop, count=count)
                r_image.show()

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = centernet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                centernet.detect_heatmap(image, heatmap_save_path)

    elif mode == "predict_mask":
        while True:
            img_path = input('Input image filename:')
            try:
                image = Image.open(img_path)
            except:
                print('Open Error! Try again!')
                continue
            else:
                overlay = centernet.detect_mask(image, alpha=0.5)
                overlay.show()

    elif mode == "dir_mask_predict":
        import os
        from tqdm import tqdm
        from PIL import Image

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                overlay = centernet.detect_mask(image, alpha=0.5)
                if not os.path.exists(dir_mask_save_path):
                    os.makedirs(dir_mask_save_path)
                save_path = os.path.join(dir_mask_save_path, img_name.rsplit('.', 1)[0] + "_mask.png")
                overlay.save(save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")

