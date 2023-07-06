from fast_reid.fast_reid_interfece import FastReIDInterface
import numpy as np
from scipy.spatial.distance import cdist
import cv2
#计算resnest50模型提取的reid特征的余弦距离

if __name__ == '__main__':

    fast_reid_config = "fast_reid/configs/MOT17/sbs_S50.yml "
    fast_reid_weights = "pretrained/mot17_sbs_S50.pth "
    device = " 2"
    img_1 = cv2.imread('fast_reid/datasets/Bicycle-ReID/bounding_box_train/0000001_Bicycle_0000010_acc_data.bmp')
    img_2 = cv2.flip(img_1, 1)

    height ,width, channels = img_1.shape

    img_1 = np.array(img_1)
    img_2 = np.array(img_2)


    det = np.array([0,0,width, height], dtype = np.float32)



    encoder = FastReIDInterface(fast_reid_config, fast_reid_weights, device)

    id_feature_1 = encoder.inference(img_1, det)
    id_feature_2 = encoder.inference(img_2, det)

    cost_matrix = np.zeros((1, 1), dtype=np.float)

    feature_1 = np.asarray(id_feature_1, dtype=np.float)
    feature_2 = np.asarray(id_feature_2, dtype=np.float)

    cost_matrix = np.maximum(0.0, cdist(feature_1, feature_2, 'cosine'))  

    print(cost_matrix)


