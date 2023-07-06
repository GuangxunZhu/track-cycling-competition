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
sys.path.append("/home/zhuguangxun/xxtrack")
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from yolov7.tracker.byte_tracker_reid import BYTETracker

from evaluator.mot_eval import Evaluator
import motmetrics as mm
import multiprocessing
from yolov7.tracker import matching
import copy

from yolov7.tracker.multi_camera_matching import single_camera_track, make_global_assignments, collect_results_write_video
from competition_rules.code.three_camera_competion_rules import Competition
from tools.mask import get_points, creat_mask





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

    frame = [1 for _ in range(num_source)] 


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = [BYTETracker(opt, frame_rate=60.0) for _ in range(num_source)]


    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    track_pool = []
    online_targets = [[] for _ in range(num_source)]
    results = [[] for _ in range(num_source)]

    results_val = []
    ret , img0 = [None for _ in range(num_source)], [None for _ in range(num_source)] 

    available_captures = [cap for cap in video_cap if cap.isOpened()]
    total_turns = opt.total_turns
    competition = Competition(opt.keyline_path,opt.xml_file, num_source, frame,video_cap, total_turns, opt)

    #获取标签对应的视频帧
    val_dic = {'0':[],'1':[],'2':[]}
    with open('dataset_path/bicycle-demo_val.txt', "r") as f:
        # 读取文件内容
        key_path = f.readlines()
        for p in key_path:
            name = p.split('/')[-1]
            key_frame = p.split('_')[-1].split('.')[0]
            key_frame = int(key_frame)
            if 'FX3' in name:
                val_dic['1'].append(key_frame)
            elif 'C3440' in name:
                val_dic['2'].append(key_frame)
            else:
                val_dic['0'].append(key_frame)

    #保存每个环节的耗时
    t_save = {'read_video': 0 ,'single_tracking': 0, 'global_assignment': 0, 'competition': 0, 'writer': 0, 'all': 0,'img_processing':0 , 
    'pred_non_max_sup':0, 'process_dets':0, 'tracking': 0, 'others':0,'collect_results':0,'plot_box':0,'plot_competition':0,'write_video':0}


    while len(available_captures):

        t_read = time.time()

        # ret , img0 = [cap.read() for cap in video_cap]
        for i, cap in enumerate(video_cap):
            ret[i] , img0[i] = cap.read()
        if all(not r for r in ret):
            break
        for i, r in enumerate(ret):
            if not r:
                online_targets[i] = []
                print("视频{}处理完成！".format(i+1))

        t_save['read_video'] += (time.time() - t_read)

        t_single_tracking = time.time()

        #预测及单相机跟踪
        online_targets, im0, p = single_camera_track(t_save, ret , img0, imgsz, source, tracker, stride, device, half, model, opt, classify, modelc, frame)

        t_save['single_tracking'] += (time.time() - t_single_tracking)

        t_global_assign = time.time()
        #全局的轨迹匹配
        online_targets, track_pool = make_global_assignments(online_targets, track_pool, opt, frame, img0)

        t_save['global_assignment'] += (time.time() - t_global_assign)

        t_competition = time.time()
        #输出比赛规则
        competition.output(track_pool, frame)

        t_save['competition'] +=  (time.time() - t_competition)

        t_writer = time.time()
        #保存信息和视频的写入
        video_cap, vid_writer = collect_results_write_video(t_save, val_dic, results_val, ret, track_pool,competition, im0, video_cap,vid_path, vid_writer, p, source, opt, results, save_img, view_img, colors, save_dir, frame)
        
        t_save['writer'] += (time.time() - t_writer)

        for i in range(len(frame)):
            if ret[i]:
                frame[i] += 1
        t_save['all'] += (time.time() - t_read)

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

    # result_filename = os.path.join(save_dir, 'labels/result.txt')

    # # evaluation
    # accs = []
    # print('Evaluating')
    # evaluator = Evaluator('/home/zhuguangxun/datasets','Bicycle-3',  'MOTChallenge') #/home/zhuguangxun/datasets/MOT17/train  MOTChallenge
    # accs.append(evaluator.eval_file(result_filename, 0, 600)) #'./results/CSTrack/result_MOT17/MOT17-02-SDP.txt' 0 600

    # # get summary
    # metrics = mm.metrics.motchallenge_metrics
    # mh = mm.metrics.create()
    # summary = Evaluator.get_summary(accs, ['val'], metrics)
    # strsummary = mm.io.render_summary(
    #     summary,
    #     formatters=mh.formatters,
    #     namemap=mm.io.motchallenge_metric_names
    # )
    # print(strsummary)
    # # print("detection_num:", result_detection, sum(result_detection))
    # # print("id_num:", result_id, sum(result_id))
    # Evaluator.save_summary(summary, os.path.join(save_dir, 'summary.xlsx'))

    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='models/bicycle-3-nocut.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/home/zhuguangxun/datasets/20221212 iPhone拍摄/Camera Angle1/C9690.MP4,/home/zhuguangxun/datasets/20221212 iPhone拍摄/Camera Angle2/20221213_FX3_2724.MP4,/home/zhuguangxun/datasets/20221212 iPhone拍摄/Camera Angle3/C3440.MP4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
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
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"logs/Bicycle/sbs_S50/model_final.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    #Competition
    parser.add_argument('--total_turns', type=int, default=10, help='number of total_turns')
    parser.add_argument('--num_objects', type=int, default=3, help='number of the objects in MOT')
    parser.add_argument('--keyline_path', default='competition_rules/files/3camera_key_line_points.json', help='path of points of key line')
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
