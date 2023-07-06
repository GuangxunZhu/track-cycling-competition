import numpy as np
import cv2

# 将图片写成视频

size = (3840,2160)#这个是图片的尺寸，一定要和要用的图片size一致
#完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
videowrite = cv2.VideoWriter('/home/zhuguangxun/xxtrack/yolov7_runs/detect/1.mp4',cv2.VideoWriter_fourcc(*'mp4v'),59,size)#20是帧数，size是图片尺寸
img_array=[]
for filename in ['/data/zhuguangxun/exp5/1_{}.jpg'.format(i) for i in range(1,14821)]:#这个循环是为了读取所有要用的图片文件
    img = cv2.imread(filename)
    if img is None:
        print(filename + " is error!")
        continue
    videowrite.write(img)
    print("正在处理图片{}".format(filename))
videowrite.release()
print('end!')
