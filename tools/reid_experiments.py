# 加载目标的patch，并进行填充为原始图片大小，用于对JDE模型提取的reid特征进行计算余弦距离
# while video_cap_1.isOpened() or video_cap_2.isOpened():
    #     ret_1, img0_1 = video_cap_1.read()
    #     ret_2, img0_2 = video_cap_2.read()


    #     # img0_1 = cv2.imread('fast_reid/datasets/Bicycle-ReID/bounding_box_test/0000002_Bicycle_0006779_acc_data.bmp')
    #     # img0_2 = cv2.imread('fast_reid/datasets/Bicycle-ReID/bounding_box_test/0000039_Bicycle_0008300_acc_data.bmp')
    #     # img0_2 = cv2.flip(img0_2, 1)


    #     # # 计算扩大后的图像的宽度和高度
    #     # new_width = 3840
    #     # new_height = 2160

    #     # # 计算需要填充的宽度和高度
    #     # delta_w = new_width - img0_1.shape[1]
    #     # delta_h = new_height - img0_1.shape[0]

    #     # # 计算填充的左、右、上、下边界
    #     # top = delta_h // 2
    #     # bottom = delta_h - top
    #     # left = delta_w // 2
    #     # right = delta_w - left

    #     # # 在图像周围进行填充
    #     # img0_1 = cv2.copyMakeBorder(img0_1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #     # print(img0_1.shape)

    #     # delta_w = new_width - img0_2.shape[1]
    #     # delta_h = new_height - img0_2.shape[0]

    #     # # 计算填充的左、右、上、下边界
    #     # top = delta_h // 2
    #     # bottom = delta_h - top
    #     # left = delta_w // 2
    #     # right = delta_w - left

    #     # # 在图像周围进行填充
    #     # img0_2 = cv2.copyMakeBorder(img0_2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #     # print(img0_2.shape)