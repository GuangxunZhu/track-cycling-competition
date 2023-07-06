import os.path as osp
import os
import numpy as np
import cv2

#前提条件：图片和txt文档均放在images_train_root中

def generate_labels(images_root, labels_root):

    seqs = os.listdir(images_root)

    img0_path = osp.join(images_root,seqs[1])
    img0 = cv2.imread(img0_path,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img_height, img_width =img0.shape[0:2]

    for seq in seqs:
        if seq.endswith('txt'):
            label_path = osp.join(images_root,seq)
            with open (label_path) as f:
                lines = f.readlines()
            for line in lines:
                gt = line.strip().split()
                gt = np.array(gt, dtype=str).reshape((-1,5))
                for id, x, y, w, h in gt:
                    mark = -1
                    if  not id[:2].isalpha() :
                        print('请确认数据集是否已经处理过')
                        break
                    if id[-1].lower() == 'd':
                        mark = 1
                        id = int(id[2:-1])
                    elif id == "领骑员":
                        id = int(32)
                    else:
                        id = int(id[2:])
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    x += w / 2
                    y += h / 2
                    label_fpath = osp.join(labels_root,seq)
                    label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:d}\n'.format(
                    id, x / img_width, y / img_height, w / img_width, h / img_height, mark)
                    with open(label_fpath, 'a') as f:
                        f.write(label_str)

#前提条件：图片在images_train_root中 ,txt文档在另外的文件夹里，先把原始的txt移走       
def generate_labels_2(images_root, labels_root):

    seqs = os.listdir(labels_root)

    img0_path = osp.join(images_root,seqs[2]).replace('txt','jpg')
    img0 = cv2.imread(img0_path,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img_height, img_width =img0.shape[0:2]

    for seq in seqs:
        label_path = osp.join(labels_root,seq)
        gt = np.loadtxt(label_path, dtype=str, delimiter=' ')
        gt = gt.reshape((-1,6))
        for id, x, y, w, h, mark in gt:
            if  not id[:2] == 'ID':
                print('请确认数据集是否已经处理过')
                break
            id = int(id[2:])
            x, y, w, h = int(x), int(y), int(w), int(h)
            x += w / 2
            y += h / 2
            label_fpath = osp.join(images_root.replace('images','labels_with_ids'),seq)
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            id, x / img_width, y / img_height, w / img_width, h / img_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)

def find_excess_img_or_txt(images_root, labels_root):
    img_seqs = os.listdir(images_root)
    label_seqs = os.listdir(labels_root)

    excess = [img for img in img_seqs if img.replace('txt','jpg') not in label_seqs]

    return excess
        

def del_imagesAndlabels(images_root, labels_root):
    labels_seqs = os.listdir(labels_root)
    images_seqs = os.listdir(images_root)

    for image_seq in images_seqs:
        if image_seq.endswith('txt'):
            label_path = osp.join(images_root,image_seq)
            os.remove(label_path)
        else:
            label_seq = image_seq.replace('jpg','txt')
            if label_seq not in labels_seqs:
                image_path = osp.join(images_root,image_seq)
                os.remove(image_path)


if __name__ == '__main__':
    images_train_root='/home/zhuguangxun/datasets/Bicycle-new-2/images/train'
    labels_train_root='/home/zhuguangxun/datasets/Bicycle-new-2/labels_with_ids/train'
    # generate_labels(images_train_root, labels_train_root)

    excess = find_excess_img_or_txt(labels_train_root,images_train_root)
    for seq in excess:
        img_path = osp.join(labels_train_root,seq)
        os.remove(img_path)
            



        



