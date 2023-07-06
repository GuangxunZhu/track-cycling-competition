import os
import glob


def gen_crowedhuman_path(root_path):
    label_path = 'crowdhuman/labels_with_ids/train'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.jpg'))
    with open('dataset_path/crowdhuman.train', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        for label in labels:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.jpg')
            if image in images_exist:
                print(image[:], file=f)
    f.close()


def gen_data_path_mot17_train(root_path):
    mot_path = 'MOT17-1/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('dataset_path/mot17_train.txt', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half):
                image = images[i]
                print(image[:], file=f)
    f.close()


def gen_data_path_mot17_val(root_path):
    mot_path = 'MOT17-1/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('dataset_path/mot17_val.txt', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half, len_all):
                image = images[i]
                print(image[:], file=f)
    f.close()


def gen_data_path_mot17_emb(root_path):
    mot_path = 'MOT17-1/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('dataset_path/mot17.emb', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half, len_all, 3):
                image = images[i]
                print(image[:], file=f)
    f.close()

def gen_data_path_mot20_train(root_path):
    mot_path = 'MOT20/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) ]
    with open('dataset_path/mot20train.txt', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half):
                image = images[i]
                print(image[:], file=f)
    f.close()

def gen_data_path_mot20_val(root_path):
    mot_path = 'MOT20/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) ]
    with open('dataset_path/mot20val.txt', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half,len_all):
                image = images[i]
                print(image[:], file=f)
    f.close()

def gen_data_path_ETHZ(root_path):
    mot_path = 'ETHZ/labels_with_ids/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) ]
    with open('dataset_path/ETHZ.txt', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'images')
            images = sorted(glob.glob(seq_path + '/*.png'))
            len_all = len(images)
            for i in range(len_all):
                image = images[i]
                print(image[:], file=f)
    f.close()

def modify_data_path_ETHZ(root_path):
    ori_data_path = 'dataset_path/ETHZ.txt'
    line_data=''
    with open(ori_data_path, 'r') as f:
        for line in f.readlines():
            line = root_path+'/'+line
            line_data += line
    with open(ori_data_path, 'w') as f:
        f.write(line_data)
            
    f.close()

def gen_data_path_cityperson_train(root_path):
    mot_path = 'Citypersons/Citypersons/labels_with_ids/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path))]
    with open('dataset_path/Citypersons_train.txt', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            labels = sorted(glob.glob(seq_path + '/*.txt'))
            images=[]
            for label in labels:
                with open(label, "r") as l:
                    if  l.readlines():
                        image = label.replace('labels_with_ids', 'images').replace('.txt', '.png')
                        images.append(image)
                l.close()
            len_all = len(images)
            for i in range(len_all):
                image = images[i]
                print(image[:], file=f)
    f.close()

def gen_data_path_cityperson_val(root_path):
    mot_path = 'Citypersons/Citypersons/labels_with_ids/val'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path))]
    with open('dataset_path/Citypersons_val.txt', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            labels = sorted(glob.glob(seq_path + '/*.txt'))
            images=[]
            for label in labels:
                with open(label, "r") as l:
                    if  l.readlines():
                        image = label.replace('labels_with_ids', 'images').replace('.txt', '.png')
                        images.append(image)
                l.close()
            len_all = len(images)
            for i in range(len_all):
                image = images[i]
                print(image[:], file=f)
    f.close()

def gen_caltech_path(root_path):
    label_path = 'Caltech/labels_with_ids'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.png'))
    with open('dataset_path/caltech_all.txt', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        for label in labels:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.png')
            if image in images_exist:
                print(image[:], file=f)
    f.close()

def gen_PRW_path(root_path):
    label_path = 'PRW/labels_with_ids'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.jpg'))
    with open('dataset_path/PRW_all.txt', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        for label in labels:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.jpg')
            if image in images_exist:
                print(image[:], file=f)
    f.close()


def gen_CUHK_path(root_path):
    label_path = 'CUHK-SYSU/labels_with_ids'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.jpg'))
    with open('dataset_path/CUHK-SYSU_all.txt', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        for label in labels:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.jpg')
            if image in images_exist:
                print(image[:], file=f)
    f.close()

def gen_Bicycle_path_train(root_path):
    label_path = 'Bicycle-cut/labels_with_ids/train'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.jpg'))
    with open('dataset_path/bicycle-cut_train.txt', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        len_all = len(labels)
        len_train = int(len_all * 0.8)
        for label in labels[:len_train]:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.jpg')
            if image in images_exist:
                print(image[:], file=f)
    f.close()

def gen_Bicycle_path_val(root_path):
    label_path = 'Bicycle-cut/labels_with_ids/train'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.jpg'))
    with open('dataset_path/bicycle-cut_val.txt', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        len_all = len(labels)
        len_train = int(len_all*0.8)
        for label in labels[len_train:]:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.jpg')
            if image in images_exist:
                print(image[:], file=f)
    f.close()


def modify_data_path_fromJDE(root_path):
    ori_data_path = 'dataset_path/caltech_10k_val.txt'
    line_data=''
    with open(ori_data_path, 'r') as f:
        for line in f.readlines():
            # line=line.replace('/data','') # caltech数据集
            line = root_path+'/'+line
            line_data += line
    with open(ori_data_path, 'w') as f:
        f.write(line_data)     
    f.close()


def gen_Bicycle_3_path(root_path):
    label_path = 'Bicycle-3/labels_with_ids/train'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.jpg'))
    with open('dataset_path/bicycle-3_train.txt', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        # len_all = len(labels)
        # len_train = int(len_all * 0.8)
        for label in labels:
            if not ('20221213_FX3_2724' in label or 'C3440' in label or 'C9690' in label) :
                image = label.replace('labels_with_ids', 'images').replace('.txt', '.jpg')
                if image in images_exist:
                    print(image[:], file=f)
    f.close()

def split_train3_to_val():
    train_path = 'dataset_path/bicycle-3_train.txt'
    val_path = 'dataset_path/bicycle-3_val.txt'
    with open(train_path, "r") as file1:
        # 读取文件内容
        content = file1.readlines()

        # 计算文件内容的长度
        length = len(content)

        # 将文件内容的一半追加到第二个文本文件
        with open(val_path, "a") as file2:
            file2.writelines(content[int(length * 0.85):])

        # 将文件内容的另一半覆盖到第一个文本文件
        with open(train_path, "w") as file1:
            file1.writelines(content[:int(length * 0.85)])
        


if __name__ == '__main__':
    root = '/home/zhuguangxun/datasets'
    gen_data_path_mot17_val(root)