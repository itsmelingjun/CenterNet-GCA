import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def rotated_gaussian2D(shape, sigma_x, sigma_y, angle_rad):
    h, w = shape
    y, x = np.ogrid[0:h, 0:w]
    y0, x0 = (h - 1) / 2., (w - 1) / 2.
    x = x - x0
    y = y - y0

    # 旋转坐标变换
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    xr = cos_a * x - sin_a * y
    yr = sin_a * x + cos_a * y

    gaussian = np.exp(-(xr**2 / (2 * sigma_x**2) + yr**2 / (2 * sigma_y**2)))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def safe_sigma(value, min_val=1.0):
    return max(value, min_val)


def draw_ellipse_gaussian(heatmap, center, radius_x, radius_y, angle_rad, k=1):
    diameter_x = 2 * radius_x + 1
    diameter_y = 2 * radius_y + 1
    sigma_x = safe_sigma(radius_x / 3, min_val=1.0)
    sigma_y = safe_sigma(radius_y / 3, min_val=1.0)
    gaussian = rotated_gaussian2D((diameter_y, diameter_x), sigma_x, sigma_y, angle_rad=angle_rad)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class CenternetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(CenternetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)

        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / 4), int(input_shape[1] / 4))
        self.num_classes = num_classes
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # -------------------------------------------------#
        #   进行数据增强
        # -------------------------------------------------#
        image, box, angle, mask = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        if len(box) != 0:
            boxes = np.array(box[:, :4], dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0,
                                       self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0,
                                       self.output_shape[0] - 1)

        ellipse_ratio = 1.6  # 控制椭圆长短轴比例，可调参数

        for i in range(len(box)):
            bbox = boxes[i].copy()
            cls_id = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # -------------------------------------------------#
                #   计算真实框所属的特征点
                # -------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                # ----------------------------#
                #   绘制椭圆高斯热力图
                # ----------------------------#
                angle_deg = angle[i]  # 取出该目标的角度
                angle_rad = np.deg2rad(-(angle_deg + 90))   # 将角度转换为弧度
                # 采用radius为椭圆半轴短轴，乘以比例参数得到椭圆半轴长轴
                radius_y = radius
                radius_x = int(radius * ellipse_ratio)
                batch_hm[:, :, cls_id] = draw_ellipse_gaussian(batch_hm[:, :, cls_id], ct, radius_x, radius_y, angle_rad)
                #batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                # ---------------------------------------------------#
                #   计算宽高真实值
                # ---------------------------------------------------#
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                # ---------------------------------------------------#
                #   计算中心偏移量
                # ---------------------------------------------------#
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                # ---------------------------------------------------#
                #   将对应的mask设置为1
                # ---------------------------------------------------#
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        image = np.transpose(preprocess_input(image), (2, 0, 1))

        return image, batch_hm, batch_wh, batch_reg, batch_reg_mask, mask

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        image_path = line[0]
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])  # line[0]是图片路径
        image = cvtColor(image)  # line[1:]是一系列目标框，格式：x1,y1,x2,y2,angle,class_id
        mask_path = image_path.replace("JPEGImages", "Mask").replace(".jpg", ".png") # 获取mask路径
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")  
            mask = mask.point(lambda p: 255 if p > 127 else 0)
        else:
            mask = Image.new("L", image.size, 0)  
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框和角度
        # ------------------------------#
        box_list = []
        angle_list = []
        for box_str in line[1:]:
            x1, y1, x2, y2, angle, class_id = box_str.split(',')
            # 坐标和类别保留整数
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(class_id)
            box_list.append([x1, y1, x2, y2, class_id])
            # 角度保留浮点
            angle_list.append(float(angle))
        box = np.array(box_list, dtype=np.int32)
        angle_array = np.array(angle_list, dtype=np.float32)

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            mask = mask.resize((nw, nh), Image.NEAREST)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_mask = Image.new("L", (w, h), 0)
            new_image.paste(image, (dx, dy))
            new_mask.paste(mask, (dx, dy))
            image_data = np.array(new_image, np.float32)
            mask_data = np.array(new_mask, np.uint8) // 255
            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                if random:  # 仅在训练时打乱
                    idxs = np.arange(len(box))
                    np.random.shuffle(idxs)
                    box = box[idxs]
                    angle_array = angle_array[idxs]
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                keep = np.logical_and(box_w > 1, box_h > 1)
                box = box[keep]
                angle_array = angle_array[keep]

            return image_data, box, angle_array, mask_data
        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        mask = mask.resize((nw, nh), Image.NEAREST)
        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_mask = Image.new("L", (w, h), 0)
        new_image.paste(image, (dx, dy))
        new_mask.paste(mask, (dx, dy))
        image = new_image
        mask = new_mask
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        mask_data = np.array(mask, np.uint8) // 255
        # ---------------------------------#
        #   对图像进行色域变换，不影响mask
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            idxs = np.arange(len(box))
            np.random.shuffle(idxs)
            box = box[idxs]
            angle_array = angle_array[idxs]
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
                angle_array = 180 - angle_array
                angle_array = np.mod(angle_array, 180)

            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            keep = np.logical_and(box_w > 1, box_h > 1)
            box = box[keep]
            angle_array = angle_array[keep]

        return image_data, box, angle_array, mask_data


# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks, masks = [], [], [], [], [], []
    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask, mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)
        masks.append(mask)
    return (torch.from_numpy(np.array(imgs)).float(),
            torch.from_numpy(np.array(batch_hms)).float(),
            torch.from_numpy(np.array(batch_whs)).float(),
            torch.from_numpy(np.array(batch_regs)).float(),
            torch.from_numpy(np.array(batch_reg_masks)).float(),
            torch.from_numpy(np.array(masks)).unsqueeze(1).float())  # mask格式: (B, 1, H, W)


