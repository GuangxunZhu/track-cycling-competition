import argparse
import time
from pathlib import Path
import sys
import numpy as np
import os
import cv2  
import torch
import torch.backends.cudnn as cudnn
from numpy import random
sys.path.append("/home/zhuguangxun/track-cycling-competition")
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from yolov7.tracker.byte_tracker_reid import BYTETracker
from fast_reid.fast_reid_interfece import FastReIDInterface

from evaluator.mot_eval import Evaluator
import motmetrics as mm
import multiprocessing
from yolov7.tracker import matching
import copy
# import cv2
import subprocess
from yolov7.tracker.multi_camera_matching_new import preprocessing_img, predict, track_single_camera,  make_global_assignments, collect_results_write_video
from competition_rules.code.three_camera_competion_rules import Competition
from tools.mask import get_points, creat_mask
import datetime
import pytz
import ffmpeg



sys.path.insert(0, './yolov7')
sys.path.append('.')

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))

def align_videos(video_paths, video_captures):
    # 存储每个视频的拍摄时间
    start_timestamps = []
    finish_timestamps = []
    # 遍历视频路径列表
    for path in video_paths[:1]:
        probe = ffmpeg.probe(path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        format_data = probe['format']

        framerate = eval(video_stream['r_frame_rate']) 

        creation_year = format_data.get('tags', {}).get('creation_time').split('T')[0]
        timecode = format_data.get('tags', {}).get('timecode')
        duration = float(format_data['duration'])

        timecode2s = timecode[:-3] + '.' + f"{str(round(int(timecode[-2:])/framerate, 3)).split('.')[-1]}"

        finish_time = creation_year + 'T' + timecode2s

        # 将时间戳转换为datetime对象
        finish_time = datetime.datetime.strptime(finish_time, "%Y-%m-%dT%H:%M:%S.%f")
        finish_timestamps.append(finish_time)
        start_time = finish_time - datetime.timedelta(seconds=duration)
        start_timestamps.append(start_time)
        start_timestamps.append(start_time)
        start_timestamps.append(start_time)


    # # 找到最晚的拍摄时间
    # latest_timestamp = max(start_timestamps)

    # # 使用OpenCV逐帧打开对齐的视频
    # for i,  video_cap in enumerate(video_captures):

    #     # 计算最晚拍摄时间的那一帧的时间索引
    #     fps = video_cap.get(cv2.CAP_PROP_FPS)
        
    #     frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #     # 计算最晚拍摄时间的那一帧的时间偏移量（相对于视频开始时间）
    #     latest_frame_offset = (datetime.datetime.combine(datetime.datetime.min,latest_timestamp.time()) - 
    #                             datetime.datetime.combine(datetime.datetime.min,start_timestamps[i].time())).total_seconds()

    #     start_timestamps[i] += datetime.timedelta(seconds = latest_frame_offset)
    #     # 计算最晚拍摄时间的那一帧的帧索引
    #     latest_frame_index = int(fps * latest_frame_offset)

    #     # 设置视频捕捉对象的帧索引
    #     video_cap.set(cv2.CAP_PROP_POS_FRAMES, latest_frame_index)


    return video_captures, start_timestamps



def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace

    if ',' in source:
        source = source.split(",")
    else:
        source = [source]

    num_source = len(source)

    save_img = not opt.nosave and not source[0].endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    modelc = None
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader

    vid_path = [None for _ in range(num_source)]
    vid_writer = [None for _ in range(num_source)] 


    video_cap = [cv2.VideoCapture(item) for item in source]
    import datetime
    # c_time = [ str(datetime.datetime.fromtimestamp(os.path.getctime(item)))  for item in source]


    frame = [1 for _ in range(num_source)] 


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # ReID model

    encoder = FastReIDInterface(opt.fast_reid_config, opt.fast_reid_weights, device)

    # Create tracker
    tracker = [BYTETracker(opt, encoder, frame_rate=60.0) for _ in range(num_source)]


    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    track_pool = []
    leader = []
    online_targets = [[] for _ in range(num_source)]
    results = [[] for _ in range(num_source)]

    ret , img0 = [None for _ in range(num_source)], [None for _ in range(num_source)] 
    # timecode = [None for _ in range(num_source)]

    available_captures = [cap for cap in video_cap if cap.isOpened()]
    total_turns = opt.total_turns

    video_cap, start_time = align_videos(source, video_cap)

    competition = Competition(opt.keyline_path, start_time, num_source, frame,video_cap, total_turns, opt)


    #保存每个环节的耗时
    t_save = {'read_video': 0 ,'single_tracking': 0, 'global_assignment': 0, 'competition': 0, 'writer': 0, 'all': 0,'img_processing':0 , 
    'pred_non_max_sup':0, 'process_dets':0, 'tracking': 0, 'collect_results':0,'plot_box':0,'plot_competition':0,'write_video':0}


    while len(available_captures):

        t_read = time.time()

        # ret , img0 = [cap.read() for cap in video_cap]
        for i, cap in enumerate(video_cap):
            ret[i] , img0[i] = cap.read()
            # timecode[i] = cap.get(cv2.CAP_PROP_POS_MSEC)
        if all(not r for r in ret):
            break
        for i, r in enumerate(ret):
            if not r:
                online_targets[i] = []
                print("视频{}处理完成！".format(i+1))

        t_save['read_video'] += (time.time() - t_read)

        print("正在处理视频{}，第{}帧！".format(i+1,frame[i]))

        # 图片预处理
        img = preprocessing_img(ret, opt, img0, imgsz, stride, device, half, t_save) 

        # 模型预测
        dets = predict(img, model, opt, ret, t_save)

        # 单相机多目标跟踪

        online_targets = track_single_camera(ret, dets, img, img0, t_save, tracker)


        # t_single_tracking = time.time()

        # #预测及单相机跟踪
        # online_targets, im0, p = single_camera_track(t_save, ret , img0, imgsz, source, tracker, stride, device, half, model, opt, classify, modelc, frame)

        # t_save['single_tracking'] += (time.time() - t_single_tracking)

        t_global_assign = time.time()
        #全局的轨迹匹配
        online_targets, track_pool, virtual_track = make_global_assignments(online_targets, track_pool, leader, opt, frame, img0)

        t_save['global_assignment'] += (time.time() - t_global_assign)

        t_competition = time.time()

        #输出比赛规则
        competition.output(track_pool, frame, virtual_track)

        t_save['competition'] +=  (time.time() - t_competition)

        t_writer = time.time()
        #保存信息和视频的写入
        video_cap, vid_writer = collect_results_write_video(t_save, ret, track_pool, virtual_track, competition, img0, video_cap,vid_path, vid_writer,  source, opt, results, save_img, view_img, colors, save_dir, frame)
        
        t_save['writer'] += (time.time() - t_writer)

        for i in range(len(frame)):
            if ret[i]:
                frame[i] += 1
        t_save['all'] += (time.time() - t_read)

    import json
    with open(os.path.join(save_dir,'data.json'),'w') as f:
        json.dump(competition.data_dic, f, indent=2)
        
    for i in range(len(video_cap)):
        video_cap[i].release()
        vid_writer[i].release()

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')   

    for i in range(len(results)):
        with  open(os.path.join(save_dir,'labels/result_{}.txt'.format(i)),'a') as gt:
            for item in results[i]:
                gt.write(item)

    with open('time_record.txt', 'w') as f:
        f.write(str(t_save))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='models/yolov7_34_best_rep.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/home/zhuguangxun/datasets/20221212 iPhone拍摄/Camera Angle1/C9690.MP4,/home/zhuguangxun/datasets/20221212 iPhone拍摄/Camera Angle2/20221213_FX3_2724.MP4,/home/zhuguangxun/datasets/20221212 iPhone拍摄/Camera Angle3/C3440.MP4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.8, help='IOU threshold for NMS')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='yolov7_runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    # parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    # parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")

    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=2000, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/Market1501/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"models/Bicycle_sbs_s50_best_2_new.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    #Competition
    parser.add_argument('--total_turns', type=int, default= 6, help='number of total_turns')
    parser.add_argument('--num_objects', type=int, default= 7, help='number of the objects in MOT')

    parser.add_argument('--keyline_path', default='competition_rules/files/new_keyline.json', help='path of points of key line')
    parser.add_argument('--mask_points_path', default='competition_rules/files/mask_left_points.txt,competition_rules/files/mask_points.txt', help='path of mask points')

    parser.add_argument('--xml_file', default='competition_rules/files/C9690M01.XML,competition_rules/files/20221213_FX3_2724M01.XML,competition_rules/files/C3440M01.XML', help='')

    

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
