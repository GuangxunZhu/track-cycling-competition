import cv2
import numpy as np
import re

#根据提取出的点，对圈出的区域进行掩码

def get_points(data):

   # 正则表达式模式
    pattern = r"\((\d+),(\d+)\)"

    # 提取坐标
    bottom_coords = re.findall(pattern, data["bottom"])

    # # 转换为数组
    # left_array = np.array(bottom_coords, dtype=int)
    # right_array = np.array(right_coords, dtype=int)
    # 转换为三维数组
    bottom_array = np.array(bottom_coords, dtype=int).reshape((1, -1, 2))

    # points=np.loadtxt(pth,dtype=np.int32, delimiter = ',').reshape((1, -1, 2))
    return bottom_array

def creat_mask(img_shape,img, points,fill,color):
    mask=np.zeros(img_shape,dtype=np.uint8)
    mask.fill(fill)
    cv2.fillPoly(mask,points,(0,0,0))
    res = cv2.bitwise_and(img, mask)
    return res




