import os.path as osp
import os
import numpy as np
import cv2
import random


images_root='/home/zhuguangxun/datasets/Bicycle/images/train'
labels_root='/home/zhuguangxun/datasets/Bicycle/labels_with_ids/train'
img_width = 3840
img_height = 2160

seqs = os.listdir(labels_root)

for seq in seqs:
    img_seq = seq.replace('txt','jpg')
    img_path = osp.join(images_root,img_seq)
    label_path = osp.join(labels_root,seq)
    img=cv2.imread(img_path,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    gt = np.loadtxt(label_path,dtype=np.float64, delimiter=' ')
    gt[:,[2,4]] *= img_width
    gt[:,[3,5]] *= img_height

    x = gt[:,2:]

    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    for i, det in enumerate(y):
        c1, c2 = (int(det[0]), int(det[1])), (int(det[2]), int(det[3]))
        color = [random.randint(0, 255) for _ in range(3)]
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        cv2.rectangle(img, c1, c2,color, thickness=tl, lineType=cv2.LINE_AA)

    # cv2.imshow('show', img)
    cv2.waitKey(1)  # 1 millisecond
    save_path = osp.join(labels_root,'111.jpg')
    cv2.imwrite(save_path, img)


            