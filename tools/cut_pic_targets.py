import sys
# sys.path.append("..")
# from yolov7.utils.datasets import img2label_paths, letterbox
import os
import numpy as np
import torch
import cv2
import random


def load_img_and_labels(path):
    img_files_dict = None

    with open(path, 'r') as file:
        img_files_dict = file.readlines()
        img_files_dict = [x.strip() for x in img_files_dict]
        img_files_dict = list(filter(lambda x: len(x) > 0, img_files_dict))

    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels_with_ids' + os.sep  # /images/, /labels/ substrings
    label_files_dict = ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_files_dict]
    # label_files_dict = img2label_paths(img_files_dict)
    return img_files_dict, label_files_dict



def block_picture(img_files, label_files, block_size = [540,960]):     
    
    img = cv2.imread(img_files)  # BGR
    # assert img is not None, 'Image Not Found ' + img_files
    h0, w0 = img.shape[:2]  # orig hw
    # r = img_size / max(h0, w0)  # resize image to img_size
    # if r != 1:  # always resize down, only resize up if training with augmentation
    #     interp = cv2.INTER_LINEAR
    #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

    with open(label_files, 'r') as f:
        label = [x.split() for x in f.read().strip().splitlines()]
        label = np.array(label, dtype=np.float32)
    # h ,w
    num_block_high =  int(img.shape[0] / block_size[0])
    num_block_width = int(img.shape[1] / block_size[1])

    cropped_images = []

    for h in range(num_block_high):
        for w in range(num_block_width):
            cropped = img[h*block_size[0]:(h+1)*block_size[0], w*block_size[1]:(w+1)*block_size[1],:]  # 剪裁小图片
            cropped_images.append(cropped)

    # 合并小图片为(16, 540, 960, 3)的张量
    merged_images = np.stack(cropped_images, axis=0)
    new_labels = []
    removed_labels = []
    # for i in range(batch_size):
    #     image = imgs[i]  # 获取第i张图片
    #     labels_per_img = labels[labels[:, 0] == i]  # 获取第i张图片对应的标签

    for h in range(num_block_high):
        for w in range(num_block_width):
            # 计算剪裁小图片的左上角坐标
            top = h * block_size[0]
            left = w * block_size[1]
            
            # 获取该小图片在合并后的张量中的编号
            idx = h * num_block_high + w
            # idx = h * num_block_high + w

            
            # 处理每个标签
            for j in range(len(label)):
                cls, id, x_c, y_c, w_c, h_c , mark = label[j]
                
                origin_area = w_c * w0 * h_c * h0
                #xywh to xyxy
                x1 = img.shape[1] * (x_c - w_c / 2)  # top left x
                y1 = img.shape[0] * (y_c - h_c / 2)  # top left y
                x2 = img.shape[1] * (x_c + w_c / 2)  # bottom right x
                y2 = img.shape[0] * (y_c + h_c / 2 ) # bottom right y
                
                # 将坐标值映射到对应的小图片中
                x1 -= left
                x2 -= left
                y1 -= top
                y2 -= top

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min( block_size[1], x2)
                y2 = min( block_size[0], y2)
                
                new_w = x2 - x1
                new_h = y2 - y1

                if x1 < x2 and y1 < y2:
                
                    # 计算相对坐标值并归一化
                    x1_norm = x1 / block_size[1]
                    x2_norm = x2 / block_size[1]
                    y1_norm = y1 / block_size[0]
                    y2_norm = y2 / block_size[0]
                    #xyxy to xywh
                    x_c = (x1_norm + x2_norm) / 2  # x center
                    y_c = (y1_norm + y2_norm) / 2  # y center
                    w_c = x2_norm - x1_norm  # width
                    h_c = y2_norm - y1_norm  # height

                    new_area = new_w * new_h
                    new_label = [idx, cls, id, x_c, y_c, w_c, h_c, mark]
                    # 将标签信息加入新的标签列表中
                    if new_area >= origin_area * 0.8 :
                        new_labels.append(new_label)
                    else:
                        removed_labels.append(new_label)

    for rmv_label in removed_labels:
        mask = np.zeros(block_size,np.uint8)
        mask.fill(255)
        pts = xywh_to_4pts(rmv_label[3:7],block_size)
        cv2.fillPoly(mask, [pts], (0, 0, 0))
        for label_other in new_labels:
            if label_other[0] == rmv_label[0]:
                pts_other = xywh_to_4pts(label_other[3:7],block_size)
                cv2.fillPoly(mask, [pts_other], (255, 255, 255))
        merged_images[int(rmv_label[0])] = cv2.bitwise_and(merged_images[int(rmv_label[0])],merged_images[int(rmv_label[0])],mask=mask)
        merged_images[int(rmv_label[0])][mask == 0] = 0

    # 将新的标签列表转换为张量
    new_labels = np.array(new_labels)
    
    return merged_images, new_labels

def xywh_to_4pts(label,block_size):
    xywh = label.copy()
    xywh[0] *= block_size[1]
    xywh[1] *= block_size[0]
    xywh[2] *= block_size[1]
    xywh[3] *= block_size[0]

    tl = [xywh[0] - xywh[2]/2, xywh[1] - xywh[3] /2 ]
    br = [xywh[0] + xywh[2]/2, xywh[1] + xywh[3] /2 ]

    tr = [xywh[0] + xywh[2]/2, xywh[1] - xywh[3] /2 ]
    bl = [xywh[0] - xywh[2]/2, xywh[1] + xywh[3] /2 ]


    return np.array([tl,tr,br,bl],np.int32)


def read_img_and_plot(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]  # orig hw
    label_path = img_path.replace('images','labels_with_ids').replace('jpg','txt')
    with open(label_path, 'r') as f:
        label = [x.split() for x in f.read().strip().splitlines()]
        label = np.array([label[1]], dtype=np.float32)
    xywh = label[:,2:-1]
    tlbr = xywh.copy()
    tlbr[:,:2] -= tlbr[:,2:] / 2
    tlbr[:,2:] += tlbr[:,:2]
    tlbr[:,[0,2]] *= w
    tlbr[:,[1,3]] *= h
    for x in tlbr:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        tl =  round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.imwrite('/home/zhuguangxun/1111.jpg', img)


if __name__ == '__main__':
    # dataset_txt_path = "dataset_path/bicycle-3_val.txt"
    # img_files , label_files = load_img_and_labels(dataset_txt_path)
    # # img_files = img_files[:2]
    # # label_files = label_files[:2]
    # for  idx, img_file in enumerate(img_files):
    #     print('正在处理第{}张图片'.format(idx+1))
    #     label_file = label_files[idx]
    #     imgs, labels = block_picture(img_file, label_file)

    #     for label in labels:
    #         new_img_file = img_file.replace('Bicycle-3','Bicycle-3-cut').replace('.','-{}.'.format(int(label[0])))
    #         new_label_file = label_file.replace('Bicycle-3','Bicycle-3-cut').replace('.','-{}.'.format(int(label[0])))
    #         cv2.imwrite(new_img_file,imgs[int(label[0])])
    #         label_list = label.tolist()
    #         label_str = ' '.join(map(str,label_list[1:])) + '\n'
    #         with open(new_label_file,'a') as f :
    #             f.write(label_str)
    img_path = '/home/zhuguangxun/datasets/Bicycle-3/images/train/20221213_FX3_2724_01530.jpg'
    read_img_and_plot(img_path)
