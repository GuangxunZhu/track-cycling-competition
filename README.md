###  **介绍** 

场地自行车检测与跟踪推理部分代码，检测模型的训练直接参考YOLOv7，使用yolov7.pt模型进行训练，Reid模型训练参考fastreid，使用market_sbs_S50模型训练

###  **运行** 

```
python tracking/demo_tracking.py --weights <检测模型路径> --fast-reid-weights <reid模型路径> --source <中间视频路径>,<左侧视频路径>,<右侧视频路径> --fuse-score --agnostic-nms --save-txt
```
###  **权重文件** 

百度网盘：链接：[https://pan.baidu.com/s/1F0xJz8IOMIsUkE0x2wb1eQ?pwd=898r](http://)
提取码：898r 

