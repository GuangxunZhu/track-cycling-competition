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

from yolov7.tracker.byte_tracker_reid_global import BYTETracker

from evaluator.mot_eval import Evaluator
import motmetrics as mm
import multiprocessing
from yolov7.tracker import matching
import copy





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
        source_1 = source.split(',',1)[0]
        source_2 = source.split(',',1)[1]

    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

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
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path_1,vid_path_2, vid_writer_1, vid_writer_2 = None, None, None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset_1 = LoadStreams(source_1, img_size=imgsz, stride=stride)
        dataset_2 = LoadStreams(source_2, img_size=imgsz, stride=stride)

    else:
        dataset_1 = LoadImages(source_1, img_size=imgsz, stride=stride)
        dataset_2 = LoadImages(source_2, img_size=imgsz, stride=stride)
    video_cap_1 = cv2.VideoCapture(source_1)
    video_cap_2 = cv2.VideoCapture(source_2)

    frame_1 = 1
    frame_2 = 1

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker_1 = BYTETracker(opt, frame_rate=30.0)
    tracker_2 = BYTETracker(opt, frame_rate=30.0)






    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    track_pool = []
    
    while video_cap_1.isOpened() or video_cap_2.isOpened():
        ret_1, img0_1 = video_cap_1.read()
        ret_2, img0_2 = video_cap_2.read()

        if not ret_1 and not ret_2:
            break

        if not ret_1:
            # video_cap_1.release()
            # vid_writer_1.release()
            online_targets_1 = []
            print("视频一处理完成！")

        if not ret_2:
            # video_cap_2.release()
            # vid_writer_2.release()
            online_targets_2 = []
            print("视频二处理完成！")
        if ret_1:
            img_1 = letterbox(img0_1, imgsz, stride)[0]
            img_1 = img_1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img_1 = np.ascontiguousarray(img_1)
            img_1 = torch.from_numpy(img_1).to(device)
            img_1 = img_1.half() if half else img_1.float()  # uint8 to fp16/32
            img_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img_1.ndimension() == 3:
                img_1 = img_1.unsqueeze(0)
            print("正在处理第一个视频第 {} 帧".format(frame_1))
            t1 = time_synchronized()
            id_feature_1, pred_1 = model(img_1, augment=opt.augment)
            pred_1 = non_max_suppression(pred_1[0], opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred_1 = apply_classifier(pred_1[0], modelc, img_1, img0_1)
            results = []
            det_1 = pred_1[0]

            if webcam:  # batch_size >= 1
                p_1, s, im0_1, frame_1 = source_1[0], '%g: ' % 0, img0_1[0].copy(), frame_1

            else:
                p_1, s, im0_1, frame_1 = source_1, '', img0_1, frame_1
            detections_1 = []
            if len(det_1):
                boxes_1 = scale_coords(img_1.shape[2:], det_1[:, :4], im0_1.shape)
                boxes_1 = boxes_1.cpu().numpy()
                detections_1 = det_1.cpu().numpy()
                detections_1[:, :4] = boxes_1
            online_targets_1 = tracker_1.update(detections_1, id_feature_1, im0_1)
            
        if ret_2:

            # for path, img, im0s, vid_cap  in dataset_1:
            img_2 = letterbox(img0_2, imgsz, stride)[0]
            # Convert
            img_2 = img_2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

            img_2 = np.ascontiguousarray(img_2)
            img_2 = torch.from_numpy(img_2).to(device)

            img_2 = img_2.half() if half else img_2.float()  # uint8 to fp16/32

            img_2 /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img_2.ndimension() == 3:
                img_2 = img_2.unsqueeze(0)

            # Inference
            print("正在处理第二个视频第 {} 帧".format(frame_2))

            id_feature_2, pred_2 = model(img_2, augment=opt.augment)
            # Apply NMS
            pred_2 = non_max_suppression(pred_2[0], opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # Apply Classifier
            if classify:
                pred_2 = apply_classifier(pred_2[0], modelc, img_2, img0_2)
            # Process detections
            results = []
            det_2 = pred_2[0]

            if webcam:  # batch_size >= 1
                p_2, s, im0_2, frame_2 = source_2[0], '%g: ' % 0, img0_2[0].copy(), frame_2

            else:
                p_2, s, im0_2, frame_2 = source_2, '', img0_2, frame_2

                # Run tracker
            detections_2 = []

            if len(det_2):
                boxes_2 = scale_coords(img_2.shape[2:], det_2[:, :4], im0_2.shape)
                boxes_2 = boxes_2.cpu().numpy()
                detections_2 = det_2.cpu().numpy()
                detections_2[:, :4] = boxes_2

            online_targets_2 = tracker_2.update(detections_2, id_feature_2, im0_2)

        '''if len(online_targets_1) or len(online_targets_2): # 两个视频中至少有一个出现跟踪目标

            # if len(online_targets_1) == 2:
            #     quuu = 2
            # if len(online_targets_1) == 3:
            #     quuu = 3
            # if len(online_targets_1) == 4:
            #     quuu = 4
            # if len(online_targets_1) == 5:
            #     quuu = 5
            # if len(online_targets_1) == 6:
            #     quuu = 6

            dists_1 = matching.targets_embedding_distance(online_targets_1,track_pool) 
            dists_1 = matching.targets_fuse_iou_1(dists_1, online_targets_1, track_pool)
            matches_1, u_matched_1, u_track_pool_1 =matching.linear_assignment(dists_1,thresh=opt.match_thresh - 0.2)
            # dists_pp = matching.targets_embedding_distance(track_pool,track_pool) 

            for itracked_1, itrack_pool_1 in matches_1:
                track_1 = online_targets_1[itracked_1]
                track_pool_1 = track_pool[itrack_pool_1]

                track_1.track_id = track_pool_1.track_id
                feat_1 =track_1.curr_feat / np.linalg.norm(track_1.curr_feat) 
                track_pool_1.smooth_feat = 0.9 * track_pool_1.smooth_feat + (1 - 0.9) * feat_1
                track_pool_1.smooth_feat /=np.linalg.norm(track_pool_1.smooth_feat)
                track_pool_1.tlbr_1 = track_1.tlbr


            id_max = 0
            for u_m in u_matched_1:
                u_m_track_1 = online_targets_1[u_m]
                for t in track_pool:
                    id_max = max(id_max, t.track_id)
                u_m_track_1.track_id = id_max + 1
                track_pool.append(copy.deepcopy(u_m_track_1))
                track_pool[-1].tlbr_1 = track_pool[-1].tlbr
                track_pool[-1].tlbr_2 = np.zeros(4)

            dists_2 = matching.targets_embedding_distance(online_targets_2,track_pool)
            dists_2 = matching.targets_fuse_iou_2(dists_2, online_targets_2, track_pool)
            matches_2, u_matched_2, u_track_pool_2 =matching.linear_assignment(dists_2,thresh=opt.match_thresh - 0.2)

            for itracked_2, itrack_pool_2 in matches_2:
                track_2 = online_targets_2[itracked_2]
                track_pool_2 = track_pool[itrack_pool_2]


                track_2.track_id = track_pool_2.track_id
                feat_2 = track_2.curr_feat / np.linalg.norm(track_2.curr_feat) 
                track_pool_2.smooth_feat = 0.9 * track_pool_2.smooth_feat + (1 - 0.9) * feat_2
                track_pool_2.smooth_feat /=np.linalg.norm(track_pool_2.smooth_feat)
                track_pool_2.tlbr_2 = track_2.tlbr

            for u_m in u_matched_2:
                u_m_track_2 = online_targets_2[u_m]
                for t in track_pool:
                    id_max = max(id_max, t.track_id)
                u_m_track_2.track_id = id_max + 1
                track_pool.append(copy.deepcopy(u_m_track_2))

                track_pool[-1].tlbr_2 = track_pool[-1].tlbr
                track_pool[-1].tlbr_1 = np.zeros(4)'''



        if ret_1:

            online_tlwhs = []
            online_ids = []
            online_scores = []
            # online_cls = []
            for t in online_targets_1:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                # tcls = t.cls
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # online_cls.append(t.cls)

                    # save results
                    results.append(
                        f"{0 + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid}'
                        else:
                            label = f'{tid}'
                        plot_one_box(tlbr, im0_1, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            p = Path(p_1)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                # cv2.imshow('BoT-SORT', im0_1)
                cv2.imwrite(str(save_dir/"1_{}.jpg".format(frame_1)),im0_1)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if not source.endswith('MP4'):
                    cv2.imwrite(save_path, im0_1)

                else:  # 'video' or 'stream'
                    if vid_path_1 != save_path:  # new video
                        vid_path_1 = save_path
                        if isinstance(vid_writer_1, cv2.VideoWriter):
                            vid_writer_1.release()  # release previous video writer
                        if video_cap_1:  # video
                            fps = video_cap_1.get(cv2.CAP_PROP_FPS)
                            w = int(video_cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(video_cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0_1.shape[1], im0_1.shape[0]
                            save_path += '.mp4'
                        vid_writer_1 = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer_1.write(im0_1)

        if ret_2:
            online_tlwhs = []
            online_ids = []
            online_scores = []
            # online_cls = []
            for t in online_targets_2:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                # tcls = t.cls
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # online_cls.append(t.cls)

                    # save results
                    results.append(
                        f"{0 + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid}'
                        else:
                            label = f'{tid}'
                        plot_one_box(tlbr, im0_2, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            p = Path(p_2)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                # cv2.imshow('BoT-SORT', im0_2)
                cv2.imwrite(str(save_dir/"2_{}.jpg".format(frame_2)),im0_2)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if not source.endswith('MP4'):
                    cv2.imwrite(save_path, im0_2)

                else:  # 'video' or 'stream'
                    if vid_path_2 != save_path:  # new video
                        vid_path_2 = save_path
                        if isinstance(vid_writer_2, cv2.VideoWriter):
                            vid_writer_2.release()  # release previous video writer
                        if video_cap_2:  # video
                            fps = video_cap_2.get(cv2.CAP_PROP_FPS)
                            w = int(video_cap_2.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(video_cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0_2.shape[1], im0_2.shape[0]
                            save_path += '.mp4'
                        vid_writer_2 = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer_2.write(im0_2)

                
        frame_1 +=1
        frame_2 +=1




        # for i, det in enumerate(pred_1):  # detections per image
        #     if webcam:  # batch_size >= 1
        #         p, s, im0, frame = source_1[i], '%g: ' % i, img0_1[i].copy(), frame_1
        #     else:
        #         p, s, im0, frame = source_1, '', img0_1, frame_1

        #     # Run tracker
        #     detections = []
        #     if len(det):
        #         boxes = scale_coords(img_1.shape[2:], det[:, :4], im0.shape)
        #         boxes = boxes.cpu().numpy()
        #         detections = det.cpu().numpy()
        #         detections[:, :4] = boxes

        #     online_targets_1 = tracker.update(detections, id_feature_1, im0)

        #     online_tlwhs = []
        #     online_ids = []
        #     online_scores = []
        #     # online_cls = []
        #     for t in online_targets_1:
        #         tlwh = t.tlwh
        #         tlbr = t.tlbr
        #         tid = t.track_id
        #         # tcls = t.cls
        #         if tlwh[2] * tlwh[3] > opt.min_box_area:
        #             online_tlwhs.append(tlwh)
        #             online_ids.append(tid)
        #             online_scores.append(t.score)
        #             # online_cls.append(t.cls)

        #             # save results
        #             results.append(
        #                 f"{i + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
        #             )

        #             if save_img or view_img:  # Add bbox to image
        #                 if opt.hide_labels_name:
        #                     label = f'{tid}'
        #                 else:
        #                     label = f'{tid}'
        #                 plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
        #     p = Path(p)  # to Path
        #     save_path = str(save_dir / p.name)  # img.jpg

        #     # Print time (inference + NMS)
        #     # print(f'{s}Done. ({t2 - t1:.3f}s)')

        #     # Stream results
        #     if view_img:
        #         cv2.imshow('BoT-SORT', im0)
        #         cv2.waitKey(1)  # 1 millisecond

        #     # Save results (image with detections)
        #     if save_img:
        #         if not source.endswith('MP4'):
        #             cv2.imwrite(save_path, im0)

        #         else:  # 'video' or 'stream'
        #             if vid_path != save_path:  # new video
        #                 vid_path = save_path
        #                 if isinstance(vid_writer, cv2.VideoWriter):
        #                     vid_writer.release()  # release previous video writer
        #                 if video_cap_1:  # video
        #                     fps = video_cap_1.get(cv2.CAP_PROP_FPS)
        #                     w = int(video_cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                     h = int(video_cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #                 else:  # stream
        #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
        #                     save_path += '.mp4'
        #                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #             vid_writer.write(im0)

        #     frame_1 += 1
        #     frame_2 += 1
    video_cap_1.release()
    video_cap_2.release()
    vid_writer_1.release()
    vid_writer_2.release()

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='models/yolov7_reid_ch_cp_17_bicycle.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/home/zhuguangxun/datasets/自行车数据/20221006/C001_l.MP4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

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
