from yolov7.utils.datasets import  letterbox
import numpy as np
import torch
from yolov7.utils.torch_utils import time_synchronized
from yolov7.utils.general import  non_max_suppression, apply_classifier, scale_coords
from yolov7.utils.plots import plot_one_box, plot_competition_results

import cv2  
import os

from yolov7.tracker import matching
import copy
from pathlib import Path
from yolov7.tracker.basetrack import BaseTrack
from collections import deque
import time
from tools.mask import get_points, creat_mask


class PoolTrack(object):

    def __init__(self, num_camera, camera_idx, new_track,track_id, shape, buffer_size = 30):
        
        self.img_shape = shape
        self.num_camera = num_camera
        self.track_id = track_id
        self.score = new_track.score
        self.camera_idx = camera_idx
        self.tracklet_len = [0] * num_camera
        self.frame_id = [None] * num_camera
        self.start_frame = [None] * num_camera

        self.tlwh = [np.zeros(4)  for _ in range(num_camera)] 
        self.tlbr = [np.zeros(4)  for _ in range(num_camera)] 
        self.features = [deque([], maxlen=buffer_size) for _ in range(num_camera)] 

        self.current_feat = [None] * num_camera
        self.smooth_feat = None

        self.alpha = 0.9

        line = get_points('competition_rules/files/mask_left_points.txt')
        self.line = line[0,:2,:]

    
    def initialize(self, frame_id,new_track):
        
        self.frame_id[self.camera_idx] = frame_id
        self.start_frame[self.camera_idx] = frame_id

        self.tlwh[self.camera_idx] = new_track.tlwh
        self.tlbr[self.camera_idx] = new_track.tlbr
        self.update_feat(new_track.curr_feat, new_track.score)
        # self.current_feat[self.camera_idx] = current_feat
        # self.smooth_feat[self.camera_idx] = smooth_feat

    def update(self, frame_id, camera_idx, new_track):
        
        if self.camera_idx == -1:
            self.start_frame[camera_idx] = frame_id
            self.tracklet_len[camera_idx] = 0

        self.camera_idx = camera_idx

        self.tracklet_len[camera_idx] += 1
        self.frame_id[camera_idx] = frame_id

        self.tlwh[camera_idx] = new_track.tlwh
        self.tlbr[camera_idx] = new_track.tlbr
        self.update_feat(new_track.curr_feat, new_track.score)
        self.score = new_track.score


    def update_feat(self, feat, score):
        feat /= np.linalg.norm(feat)
        self.current_feat[self.camera_idx] = feat
        if self.smooth_feat is None:           
            self.smooth_feat = feat
        else:

            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

        if self.distance_from_border() >= 15 and score >= 0.95:

            self.features[self.camera_idx].append(feat)

        self.smooth_feat /= np.linalg.norm(self.smooth_feat)


    def mark_unmatched(self):
        # self.score = None
        # self.tracklet_len = [0] * self.num_camera
        # self.start_frame = [None] * self.num_camera
        self.camera_idx = -1

    @property
    def end_frame(self):
        return self.frame_id[self.camera_idx]

    def __repr__(self):
        return 'PT_{}_({}-{})'.format(self.track_id, self.start_frame[self.camera_idx], self.end_frame)
    
    def distance_from_border(self):
        tlbr = list(self.tlbr[self.camera_idx])
        height, width = self.img_shape[:2]
        distances = [tlbr[0],tlbr[1], width - tlbr[2], height - tlbr[3]]
        min_distance = min(distances)

        if self.camera_idx == 1:
            point = [tlbr[0],tlbr[3]]
            line_distance = self.point_to_line_distance(point, self.line)

            min_distance = min(min_distance, line_distance)

        return min_distance

    def point_to_line_distance(self, point, line):
        # 计算斜率和截距
        x1, y1 = line[0]
        x2, y2 = line[1]
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # 计算点到直线的距离
        xp, yp = point

        distance = abs(slope * xp - yp + intercept) / ((slope ** 2 + 1) ** 0.5)
        
        return distance







def single_camera_track(t_save, ret , img0, imgsz, source, tracker, stride, device, half, model, opt, classify, modelc, frame):

    img = [None for _ in range(len(ret))]
    # id_feature = [] * len(ret)
    online_targets = [[] for _ in range(len(ret))] 
    im0 = [None for _ in range(len(ret))] 
    p = [None for _ in range(len(ret))] 
    img1 = img0[:]

    # pred = [] * len(ret)
    # det = [] * len(ret)
    for i in range(len(ret)):
        if ret[i]:
            
            t_img_processing = time.time()
            if i == 1:
                mask_points_path = opt.mask_points_path.split(',')
                points_right = get_points(mask_points_path[0])
                points_left = get_points(mask_points_path[1])

                img1[i] = creat_mask(img0[1].shape, img0[1], points_right, 255, (0,0,0))
                img1[i] = creat_mask(img0[i].shape, img1[i], points_left, 255, (0,0,0))


            img[i] = letterbox(img1[i], imgsz, stride)[0]
            img[i] = img[i][:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img[i] = np.ascontiguousarray(img[i])
            img[i] = torch.from_numpy(img[i]).to(device)
            img[i] = img[i].half() if half else img[i].float()  # uint8 to fp16/32
            img[i] /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img[i].ndimension() == 3:
                img[i] = img[i].unsqueeze(0)
            print("正在处理第{}个视频第 {} 帧".format(i+1, frame[i]))

            t_save['img_processing'] += (time.time() - t_img_processing)

            t_pred =time.time()

            t1 = time_synchronized()



            id_feature,pred = model(img[i], augment=opt.augment)

            # pred = model(img[i], augment=opt.augment)

            pred = non_max_suppression(pred[0], opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            t_save['pred_non_max_sup'] += (time.time() - t_pred)
            t2 = time_synchronized()

            t_process_dets = time.time()
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred[0], modelc, img[i], img0[i])
            # results = []
            det = pred[0]

            p[i], s, im0[i], frame[i] = source[i], '', img0[i], frame[i]

            detections = []

            if len(det):
                boxes = scale_coords(img[i].shape[2:], det[:, :4], im0[i].shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes
                remove_idx = []
                for d in range(detections.shape[0]) :
                    if distance_from_border(detections[d,:4]) <= 15 :
                        remove_idx.append(d)
                detections = np.delete(detections, remove_idx, axis= 0 )
                if detections.shape[0] == 0:
                    detections = list(detections)

            t_save['process_dets'] += (time.time() - t_process_dets)

            t_tracking = time.time()
            
            online_targets[i] = tracker[i].update(detections, id_feature, im0[i])

            t_save['tracking'] += (time.time() - t_tracking)


    return online_targets, im0, p

def distance_from_border(det):
        tlbr = list(det)
        height, width = [2160, 3840]
        distances = [tlbr[0],tlbr[1], width - tlbr[2], height - tlbr[3]]
        min_distance = min(distances)

        return min_distance

def make_global_assignments(online_targets, track_pool, opt, frame, img, tracklet_len = 5):

    has_non_empty_sublist = any(online_target for online_target in online_targets if online_target)

    global_left_pools = []

    if has_non_empty_sublist: # 所有视频中至少有一个出现跟踪目标

        for i in range(len(online_targets)):
            
            left_pool_indx = []

            dists = matching.targets_embedding_distance(online_targets[i],track_pool, i) 
            dists = matching.targets_fuse_iou(dists, online_targets[i], track_pool, i)
            # dists_1 = matching.target_fuse_score(dists_1, online_targets_1)

            matches, u_matched, u_track_pool =matching.linear_assignment(dists,thresh = opt.match_thresh - 0.3)
            # dists_pp = matching.targets_embedding_distance(track_pool,track_pool) 

            for itracked, itrack_pool in matches:

                track = online_targets[i][itracked]
                pool_tracklet = track_pool[itrack_pool]

                pool_tracklet.update(frame[i], i, track)

            for u_p in u_track_pool:
                left_pool_indx.append(u_p)

            if len(track_pool) == opt.num_objects and len(u_matched):
                left_pool_indx = []
                left_trks = []
                for j in u_matched:
                    if online_targets[i][j].tracklet_len >= tracklet_len:
                        left_trks.append(online_targets[i][j])

                # left_trks_1 = [online_targets_1[i] for i in u_matched_1]
                left_pools = [track_pool[j] for j in u_track_pool]

                left_dists = matching.targets_embedding_distance(left_trks,left_pools,i) 

                left_matches, left_u_matched, left_u_track_pool =matching.linear_assignment(left_dists,thresh=opt.match_thresh)
                
                for itracked, itrack_pool in left_matches:
                    track = left_trks[itracked]
                    pool_tracklet = left_pools[itrack_pool]
                    # pool_tracklet.tlwh[i] = track.tlwh
                    # pool_tracklet.tlbr[i] = track.tlbr
                    pool_tracklet.update(frame[i], i, track)

                
                for u_m in left_u_track_pool:
                    u_m_pool = left_pools[u_m]
                    for k, pool in enumerate(track_pool):
                        if pool is u_m_pool:
                            left_pool_indx.append(k)


            if len(track_pool) < opt.num_objects and len(u_matched):
                left_pool_indx = []

                thresh_offset = -0.55

                if len(u_track_pool):
                    left_trks = []
                    for j in u_matched:
                        if online_targets[i][j].tracklet_len >= tracklet_len:
                            left_trks.append(online_targets[i][j])
                    # left_trks_1 = [online_targets_1[i] for i in left_u_matched_1]
                    left_pools = [track_pool[i] for i in u_track_pool]

                    left_dists = matching.targets_embedding_distance(left_trks, left_pools, i) 

                    left_matches, left_u_matched, left_u_track_pool =matching.linear_assignment(left_dists,thresh=opt.match_thresh + thresh_offset)

                    for itracked, itrack_pool in left_matches:
                        track = left_trks[itracked]
                        pool_tracklet = left_pools[itrack_pool]
                        # pool_tracklet.tlwh[i] = track.tlwh
                        # pool_tracklet.tlbr[i] = track.tlbr

                        pool_tracklet.update(frame[i], i, track)


                    for u_m in left_u_matched:

                        u_m_track = left_trks[u_m]

                        if u_m_track.tracklet_len >= tracklet_len:
                            if len(track_pool) < opt.num_objects:

                                track_pool.append(PoolTrack(len(online_targets), i, u_m_track, len(track_pool) + 1, img[i].shape))
                                track_pool[-1].initialize( frame[i], u_m_track)

                    for u_m in left_u_track_pool:
                        u_m_pool = left_pools[u_m]
                        for k, pool in enumerate(track_pool):
                            if pool is u_m_pool:
                                left_pool_indx.append(k)

                else:
                    for j in u_matched:
                        if online_targets[i][j].tracklet_len >= tracklet_len:

                            u_m_track = online_targets[i][j]
                            if len(track_pool) < opt.num_objects:

                                track_pool.append(PoolTrack(len(online_targets), i, u_m_track, len(track_pool) + 1, img[i].shape))
                                track_pool[-1].initialize( frame[i], u_m_track)


            global_left_pools.append(left_pool_indx)

        sets = [set(x) for x in global_left_pools]
        # 使用交集操作获取共有元素
        all_unmatched_pools = list(set.intersection(*sets))
        for u_m in all_unmatched_pools:
            u_m_pool = track_pool[u_m]
            u_m_pool.mark_unmatched()
    else:
        for u_m_pool in track_pool:
            u_m_pool.mark_unmatched()


    return online_targets, track_pool


def collect_results_write_video(t_save, val_dic, results_val, ret, track_pool,competition, im0, video_cap,vid_path, vid_writer, p, source, opt, results, save_img, view_img, colors, save_dir, frame):

    t_others = time.time()

    tracklets_correspond_cameras = [[] for _ in range(len(ret))]
    for t in track_pool:
        camera_idx = t.camera_idx
        if camera_idx != -1:
            tracklets_correspond_cameras[camera_idx].append(t)

    for i in range(len(ret)):
        if ret[i]:
            online_tlwhs = []
            online_ids = []
            online_scores = []
            # online_cls = []

            
            for t in tracklets_correspond_cameras[i]:
                
                tlwh = t.tlwh[i]
                conf = t.score
                tlbr = t.tlbr[i]

                tid = t.track_id
                # tcls = t.cls
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # online_cls.append(t.cls)
                    # save results

                    key = '{}'.format(i)
                    key_frame = val_dic[key]

                    t_save['others'] +=  (time.time() - t_others)
                    
                    t_collect_results = time.time()

                    if frame[i] + 1 in key_frame:
                        results[i].append(
                        f"{frame[i] + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                        results_val.append(
                            f"{frame[i] + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                    t_save['collect_results'] +=  (time.time() - t_collect_results)
                    
                    t_plot_box = time.time()

                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid} {conf:.2f}'
                        else:
                            label = f'{tid} {conf:.2f}'

                        color = [[0,0,255],[0,255,0],[255,0,0]]
                        # plot_one_box(tlbr, im0[i], label=label, color=colors[int(tid) % len(colors)], line_thickness=3)
                        plot_one_box(tlbr, im0[i], label=label, color=colors[int(tid)], line_thickness=3)

                    t_save['plot_box'] +=  (time.time() - t_plot_box)

            t_plot_competition = time.time()

            plot_competition_results(competition, im0[i])

            t_save['plot_competition'] +=  (time.time() - t_plot_competition)

            t_write_video = time.time()
            
            path = Path(p[i])  # to Path
            save_path = str(save_dir / path.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                # cv2.imshow('BoT-SORT', im0_1)
                cv2.imwrite(str(save_dir/"1_{}.jpg".format(frame[i])),im0[i])
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                with  open(os.path.join(save_dir,'labels/result.txt'),'a') as gt:
                    for item in results_val:
                        gt.write(item)
                if not source[i].endswith('MP4'):
                    cv2.imwrite(save_path, im0[i])

                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if video_cap[i]:  # video
                            fps = video_cap[i].get(cv2.CAP_PROP_FPS)
                            w = int(video_cap[i].get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(video_cap[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0[i].shape[1], im0[i].shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0[i])

            t_save['write_video'] +=  (time.time() - t_write_video)
                    
    return video_cap, vid_writer