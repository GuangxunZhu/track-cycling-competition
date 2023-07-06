 # 原始进行目标跟踪和跨相机匹配以及视频写入的代码
 #     if not ret_1 and not ret_2:
    #         break

    #     if not ret_1:
    #         # video_cap_1.release()
    #         # vid_writer_1.release()
    #         online_targets_1 = []
    #         print("视频一处理完成！")

    #     if not ret_2:
    #         # video_cap_2.release()
    #         # vid_writer_2.release()
    #         online_targets_2 = []
    #         print("视频二处理完成！")
    #     if ret_1:
    #         img_1 = letterbox(img0_1, imgsz, stride)[0]
    #         img_1 = img_1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    #         img_1 = np.ascontiguousarray(img_1)
    #         img_1 = torch.from_numpy(img_1).to(device)
    #         img_1 = img_1.half() if half else img_1.float()  # uint8 to fp16/32
    #         img_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
    #         if img_1.ndimension() == 3:
    #             img_1 = img_1.unsqueeze(0)
    #         print("正在处理第一个视频第 {} 帧".format(frame_1))
    #         t1 = time_synchronized()

    #         id_feature_1, pred_1 = model(img_1, augment=opt.augment)

    #         pred_1 = non_max_suppression(pred_1[0], opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #         t2 = time_synchronized()

    #         # Apply Classifier
    #         if classify:
    #             pred_1 = apply_classifier(pred_1[0], modelc, img_1, img0_1)
    #         results = []
    #         det_1 = pred_1[0]

    #         if webcam:  # batch_size >= 1
    #             p_1, s, im0_1, frame_1 = source_1[0], '%g: ' % 0, img0_1[0].copy(), frame_1

    #         else:
    #             p_1, s, im0_1, frame_1 = source_1, '', img0_1, frame_1
    #         detections_1 = []
    #         if len(det_1):
    #             boxes_1 = scale_coords(img_1.shape[2:], det_1[:, :4], im0_1.shape)
    #             boxes_1 = boxes_1.cpu().numpy()
    #             detections_1 = det_1.cpu().numpy()
    #             detections_1[:, :4] = boxes_1
    #         online_targets_1 = tracker_1.update(detections_1, id_feature_1, im0_1)
            
    #     if ret_2:

    #         # for path, img, im0s, vid_cap  in dataset_1:
    #         img_2 = letterbox(img0_2, imgsz, stride)[0]
    #         # Convert
    #         img_2 = img_2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

    #         img_2 = np.ascontiguousarray(img_2)
    #         img_2 = torch.from_numpy(img_2).to(device)

    #         img_2 = img_2.half() if half else img_2.float()  # uint8 to fp16/32

    #         img_2 /= 255.0  # 0 - 255 to 0.0 - 1.0

    #         if img_2.ndimension() == 3:
    #             img_2 = img_2.unsqueeze(0)

    #         # Inference
    #         print("正在处理第二个视频第 {} 帧".format(frame_2))

    #         id_feature_2, pred_2 = model(img_2, augment=opt.augment)
    #         # Apply NMS
    #         pred_2 = non_max_suppression(pred_2[0], opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #         # Apply Classifier
    #         if classify:
    #             pred_2 = apply_classifier(pred_2[0], modelc, img_2, img0_2)
    #         # Process detections
    #         results = []
    #         det_2 = pred_2[0]

    #         if webcam:  # batch_size >= 1
    #             p_2, s, im0_2, frame_2 = source_2[0], '%g: ' % 0, img0_2[0].copy(), frame_2

    #         else:
    #             p_2, s, im0_2, frame_2 = source_2, '', img0_2, frame_2

    #             # Run tracker
    #         detections_2 = []

    #         if len(det_2):
    #             boxes_2 = scale_coords(img_2.shape[2:], det_2[:, :4], im0_2.shape)
    #             boxes_2 = boxes_2.cpu().numpy()
    #             detections_2 = det_2.cpu().numpy()
    #             detections_2[:, :4] = boxes_2

    #         online_targets_2 = tracker_2.update(detections_2, id_feature_2, im0_2)


    #     # if detections_1.shape[1] == 5:
    #     #     scores_1 = detections_1[:, 4]
    #     #     bboxes_1 = detections_1[:, :4]
    #     # else:
    #     #     # output_results = output_results.cpu().numpy()
    #     #     scores_1 = detections_1[:, 4] 
    #     #     bboxes_1 = detections_1[:, :4]  # x1y1x2y2
    #     # from yolov7.utils.general import get_id_inds
    #     # remain_inds_1 = scores_1 > 0.4
    #     # dets_1 = bboxes_1[remain_inds_1]
    #     # x_inds_1, y_inds_1 = get_id_inds(dets_1)
    #     # if detections_2.shape[1] == 5:
    #     #     scores_2 = detections_2[:, 4]
    #     #     bboxes_2 = detections_2[:, :4]
    #     # else:
    #     #     # output_results = output_results.cpu().numpy()
    #     #     scores_2 = detections_2[:, 4] 
    #     #     bboxes_2 = detections_2[:, :4]  # x1y1x2y2
    #     # remain_inds_2 = scores_2 > 0.4
    #     # dets_2 = bboxes_2[remain_inds_2]
    #     # x_inds_2, y_inds_2 = get_id_inds(dets_2)
    #     # id_feature_1 = id_feature_1[0, x_inds_1, y_inds_1, :].cpu().numpy()
    #     # id_feature_2 = id_feature_2[0, x_inds_2, y_inds_2, :].cpu().numpy()
    #     # cost_matrix = np.zeros((1, 1), dtype=np.float32)
    #     # from scipy.spatial.distance import cdist
    #     # cost_matrix = np.maximum(0.0, cdist(id_feature_1, id_feature_2, 'cosine'))  
    #     # print(cost_matrix)

        
        
    #     id_max = 0

    #     if len(online_targets_1) or len(online_targets_2): # 两个视频中至少有一个出现跟踪目标

    #         # if len(online_targets_1) == 2:
    #         #     quuu = 2
    #         # if len(online_targets_1) == 3:
    #         #     quuu = 3
    #         # if len(online_targets_1) == 4:
    #         #     quuu = 4
    #         # if len(online_targets_1) == 5:
    #         #     quuu = 5
    #         # if len(online_targets_1) == 6:
    #         #     quuu = 6

    #         dists_1 = matching.targets_embedding_distance(online_targets_1,track_pool) 
    #         dists_1 = matching.targets_fuse_iou_1(dists_1, online_targets_1, track_pool)
    #         # dists_1 = matching.target_fuse_score(dists_1, online_targets_1)

    #         matches_1, u_matched_1, u_track_pool_1 =matching.linear_assignment(dists_1,thresh = opt.match_thresh - 0.2)
    #         # dists_pp = matching.targets_embedding_distance(track_pool,track_pool) 

    #         for itracked_1, itrack_pool_1 in matches_1:
    #             track_1 = online_targets_1[itracked_1]
    #             track_pool_1 = track_pool[itrack_pool_1]

    #             track_1.track_id = track_pool_1.track_id
               
    #             feat_1 =track_1.curr_feat / np.linalg.norm(track_1.curr_feat) 
    #             track_pool_1.smooth_feat = 0.9 * track_pool_1.smooth_feat + (1 - 0.9) * feat_1
    #             track_pool_1.smooth_feat /=np.linalg.norm(track_pool_1.smooth_feat)
    #             track_pool_1.tlbr_1 = track_1.tlbr

    #         if len(track_pool) == 6 and len(u_matched_1):
    #             indices_less_twenty = []
    #             left_trks_1 = []
    #             for i in u_matched_1:
    #                 if online_targets_1[i].tracklet_len < 20:
    #                     indices_less_twenty.append(i)
    #                 else:
    #                     left_trks_1.append(online_targets_1[i])

    #             # left_trks_1 = [online_targets_1[i] for i in u_matched_1]
    #             left_pools_1 = [track_pool[i] for i in u_track_pool_1]

    #             left_dists_1 = matching.targets_embedding_distance(left_trks_1,left_pools_1) 
    #             left_matches_1, left_u_matched_1, left_u_track_pool_1 =matching.linear_assignment(left_dists_1,thresh=opt.match_thresh + 10)
                
    #             for itracked_1, itrack_pool_1 in left_matches_1:
    #                 track_1 = left_trks_1[itracked_1]
    #                 track_pool_1 = left_pools_1[itrack_pool_1]

    #                 track_1.track_id = track_pool_1.track_id
                    
    #                 feat_1 =track_1.curr_feat / np.linalg.norm(track_1.curr_feat) 
    #                 track_pool_1.smooth_feat = 0.9 * track_pool_1.smooth_feat + (1 - 0.9) * feat_1
    #                 track_pool_1.smooth_feat /=np.linalg.norm(track_pool_1.smooth_feat)
    #                 track_pool_1.tlbr_1 = track_1.tlbr
                
    #             for i in reversed(indices_less_twenty):
    #                 online_targets_1.pop(i)

    #         if len(track_pool) < 6 and len(u_matched_1):
    #             thresh_offset = -0.55
    #             indices_less_twenty = []

    #             if len(u_track_pool_1):
    #                 left_trks_1 = []
    #                 for i in u_matched_1:
    #                     if online_targets_1[i].tracklet_len < 20:
    #                         indices_less_twenty.append(i)
    #                     else:
    #                         left_trks_1.append(online_targets_1[i])
    #                 # left_trks_1 = [online_targets_1[i] for i in left_u_matched_1]
    #                 left_pools_1 = [track_pool[i] for i in u_track_pool_1]

    #                 left_dists_1 = matching.targets_embedding_distance(left_trks_1,left_pools_1) 
    #                 left_matches_1, left_u_matched_1, left_u_track_pool_1 =matching.linear_assignment(left_dists_1,thresh=opt.match_thresh + thresh_offset)

    #                 for itracked_1, itrack_pool_1 in left_matches_1:
    #                     track_1 = left_trks_1[itracked_1]
    #                     track_pool_1 = left_pools_1[itrack_pool_1]

    #                     track_1.track_id = track_pool_1.track_id
                        
    #                     feat_1 =track_1.curr_feat / np.linalg.norm(track_1.curr_feat) 
    #                     track_pool_1.smooth_feat = 0.9 * track_pool_1.smooth_feat + (1 - 0.9) * feat_1
    #                     track_pool_1.smooth_feat /=np.linalg.norm(track_pool_1.smooth_feat)
    #                     track_pool_1.tlbr_1 = track_1.tlbr

    #                 for u_m in left_u_matched_1:
    #                     u_m_track_1 = left_trks_1[u_m]
    #                     if u_m_track_1.tracklet_len >= 20 :
    #                         if len(track_pool) < 6:
    #                             for t in track_pool:
    #                                 id_max = max(id_max, t.track_id)
    #                             u_m_track_1.track_id = id_max + 1
    #                             track_pool.append(copy.deepcopy(u_m_track_1))
    #                             # to_remove_um_indices.append(u_m)
    #                             track_pool[-1].tlbr_1 = track_pool[-1].tlbr
    #                             track_pool[-1].tlbr_2 = np.zeros(4)
    #             else:
    #                 for i in u_matched_1:
    #                     if online_targets_1[i].tracklet_len < 20:
    #                         indices_less_twenty.append(i)
    #                     else:
    #                         u_m_track_1 = online_targets_1[i]
    #                         if len(track_pool) < 6:
    #                             for t in track_pool:
    #                                 id_max = max(id_max, t.track_id)
    #                             u_m_track_1.track_id = id_max + 1
    #                             track_pool.append(copy.deepcopy(u_m_track_1))
    #                             # to_remove_um_indices.append(u_m)
    #                             track_pool[-1].tlbr_1 = track_pool[-1].tlbr
    #                             track_pool[-1].tlbr_2 = np.zeros(4)
                        
    #             for i in reversed(indices_less_twenty):
    #                 online_targets_1.pop(i)






                    



    #         dists_2 = matching.targets_embedding_distance(online_targets_2,track_pool)
    #         dists_2 = matching.targets_fuse_iou_2(dists_2, online_targets_2, track_pool)
    #         # dists_2 = matching.target_fuse_score(dists_2, online_targets_2)
    #         matches_2, u_matched_2, u_track_pool_2 =matching.linear_assignment(dists_2,thresh= opt.match_thresh - 0.2 )

    #         for itracked_2, itrack_pool_2 in matches_2:
    #             track_2 = online_targets_2[itracked_2]
    #             track_pool_2 = track_pool[itrack_pool_2]


    #             track_2.track_id = track_pool_2.track_id
                
    #             feat_2 = track_2.curr_feat / np.linalg.norm(track_2.curr_feat) 
    #             track_pool_2.smooth_feat = 0.9 * track_pool_2.smooth_feat + (1 - 0.9) * feat_2
    #             track_pool_2.smooth_feat /=np.linalg.norm(track_pool_2.smooth_feat)
    #             track_pool_2.tlbr_2 = track_2.tlbr

    #         if len(track_pool) == 6 and len(u_matched_2):
    #             indices_less_twenty = []
    #             left_trks_2 = []
    #             for i in u_matched_2:
    #                 if online_targets_2[i].tracklet_len < 20:
    #                     indices_less_twenty.append(i)
    #                 else:
    #                     left_trks_2.append(online_targets_2[i])

    #             # left_trks_2 = [online_targets_2[i] for i in u_matched_2]
    #             left_pools_2 = [track_pool[i] for i in u_track_pool_2]

    #             left_dists_2 = matching.targets_embedding_distance(left_trks_2,left_pools_2) 
    #             left_matches_2, left_u_matched_2, left_u_track_pool_2 =matching.linear_assignment(left_dists_2,thresh=opt.match_thresh + 10)
                
    #             for itracked_2, itrack_pool_2 in left_matches_2:
    #                 track_2 = left_trks_2[itracked_2]
    #                 track_pool_2 = left_pools_2[itrack_pool_2]

    #                 track_2.track_id = track_pool_2.track_id
                   
    #                 feat_2 = track_2.curr_feat / np.linalg.norm(track_2.curr_feat) 
    #                 track_pool_2.smooth_feat = 0.9 * track_pool_2.smooth_feat + (1 - 0.9) * feat_2
    #                 track_pool_2.smooth_feat /=np.linalg.norm(track_pool_2.smooth_feat)
    #                 track_pool_2.tlbr_2 = track_2.tlbr

    #             for i in reversed(indices_less_twenty):
    #                 online_targets_2.pop(i)

    #         if len(track_pool) < 6 and len(u_matched_2):
    #             thresh_offset = -0.55
    #             indices_less_twenty = []

    #             if len(u_track_pool_2):
    #                 left_trks_2 = []

    #                 for i in u_matched_2:
    #                     if online_targets_2[i].tracklet_len < 20:
    #                         indices_less_twenty.append(i)
    #                     else:
    #                         left_trks_2.append(online_targets_2[i])


    #                 # left_trks_2 = [online_targets_2[i] for i in left_u_matched_2]
    #                 left_pools_2 = [track_pool[i] for i in u_track_pool_2]

    #                 left_dists_2 = matching.targets_embedding_distance(left_trks_2,left_pools_2) 
    #                 left_matches_2, left_u_matched_2, left_u_track_pool_2 =matching.linear_assignment(left_dists_2,thresh=opt.match_thresh + thresh_offset)

    #                 for itracked_2, itrack_pool_2 in left_matches_2:
    #                     track_2 = left_trks_2[itracked_2]
    #                     track_pool_2 = left_pools_2[itrack_pool_2]

    #                     track_2.track_id = track_pool_2.track_id
                        
    #                     feat_2 = track_2.curr_feat / np.linalg.norm(track_2.curr_feat) 
    #                     track_pool_2.smooth_feat = 0.9 * track_pool_2.smooth_feat + (1 - 0.9) * feat_2
    #                     track_pool_2.smooth_feat /=np.linalg.norm(track_pool_2.smooth_feat)
    #                     track_pool_2.tlbr_2 = track_2.tlbr

    #                 for u_m in left_u_matched_2:
    #                     u_m_track_2 = left_trks_2[u_m]
    #                     if u_m_track_2.tracklet_len >= 20 :
    #                         if len(track_pool) < 6:
                                
    #                             for t in track_pool:
    #                                 id_max = max(id_max, t.track_id)
    #                             u_m_track_2.track_id = id_max + 1
    #                             track_pool.append(copy.deepcopy(u_m_track_2))
    #                             # to_remove_um_indices.append(u_m)
    #                             track_pool[-1].tlbr_2 = track_pool[-1].tlbr
    #                             track_pool[-1].tlbr_1 = np.zeros(4)
    #                     # else :
    #                         # to_remove_um_indices.append(u_m)     
    #             else:
    #                 for i in u_matched_2:
    #                     if online_targets_2[i].tracklet_len < 20:
    #                         indices_less_twenty.append(i)
    #                     else:
    #                         u_m_track_2 = online_targets_2[i]
    #                         if len(track_pool) < 6:
    #                             for t in track_pool:
    #                                 id_max = max(id_max, t.track_id)
    #                             u_m_track_2.track_id = id_max + 1
    #                             track_pool.append(copy.deepcopy(u_m_track_2))
    #                             # to_remove_um_indices.append(u_m)
    #                             track_pool[-1].tlbr_2 = track_pool[-1].tlbr
    #                             track_pool[-1].tlbr_1 = np.zeros(4)
                        
    #             for i in reversed(indices_less_twenty):
    #                 online_targets_2.pop(i)




    #                 # left_u_matched_2 = np.setdiff1d(left_u_matched_2, np.array(to_remove_um_indices))

    #         # for u_m in u_matched_2:
    #         #     u_m_track_2 = online_targets_2[u_m]
    #         #     for t in track_pool:
    #         #         id_max = max(id_max, t.track_id)
    #         #     u_m_track_2.track_id = id_max + 1
    #         #     track_pool.append(copy.deepcopy(u_m_track_2))

    #         #     track_pool[-1].tlbr_2 = track_pool[-1].tlbr
    #         #     track_pool[-1].tlbr_1 = np.zeros(4)



    #     if ret_1:

    #         online_tlwhs = []
    #         online_ids = []
    #         online_scores = []
    #         # online_cls = []
    #         for t in online_targets_1:
    #             tlwh = t.tlwh
    #             tlbr = t.tlbr
    #             tid = t.track_id
    #             # tcls = t.cls
    #             if tlwh[2] * tlwh[3] > opt.min_box_area:
    #                 online_tlwhs.append(tlwh)
    #                 online_ids.append(tid)
    #                 online_scores.append(t.score)
    #                 # online_cls.append(t.cls)

    #                 # save results
    #                 results.append(
    #                     f"{0 + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
    #                 )

    #                 if save_img or view_img:  # Add bbox to image
    #                     if opt.hide_labels_name:
    #                         label = f'{tid}'
    #                     else:
    #                         label = f'{tid}'
    #                     plot_one_box(tlbr, im0_1, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
    #         p = Path(p_1)  # to Path
    #         save_path = str(save_dir / p.name)  # img.jpg

    #         # Print time (inference + NMS)
    #         # print(f'{s}Done. ({t2 - t1:.3f}s)')

    #         # Stream results
    #         if view_img:
    #             # cv2.imshow('BoT-SORT', im0_1)
    #             cv2.imwrite(str(save_dir/"1_{}.jpg".format(frame_1)),im0_1)
    #             cv2.waitKey(1)  # 1 millisecond

    #         # Save results (image with detections)
    #         if save_img:
    #             if not source.endswith('MP4'):
    #                 cv2.imwrite(save_path, im0_1)

    #             else:  # 'video' or 'stream'
    #                 if vid_path_1 != save_path:  # new video
    #                     vid_path_1 = save_path
    #                     if isinstance(vid_writer_1, cv2.VideoWriter):
    #                         vid_writer_1.release()  # release previous video writer
    #                     if video_cap_1:  # video
    #                         fps = video_cap_1.get(cv2.CAP_PROP_FPS)
    #                         w = int(video_cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                         h = int(video_cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     else:  # stream
    #                         fps, w, h = 30, im0_1.shape[1], im0_1.shape[0]
    #                         save_path += '.mp4'
    #                     vid_writer_1 = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #                 vid_writer_1.write(im0_1)

    #     if ret_2:
    #         online_tlwhs = []
    #         online_ids = []
    #         online_scores = []
    #         # online_cls = []
    #         for t in online_targets_2:
    #             tlwh = t.tlwh
    #             tlbr = t.tlbr
    #             tid = t.track_id
    #             # tcls = t.cls
    #             if tlwh[2] * tlwh[3] > opt.min_box_area:
    #                 online_tlwhs.append(tlwh)
    #                 online_ids.append(tid)
    #                 online_scores.append(t.score)
    #                 # online_cls.append(t.cls)

    #                 # save results
    #                 results.append(
    #                     f"{0 + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
    #                 )

    #                 if save_img or view_img:  # Add bbox to image
    #                     if opt.hide_labels_name:
    #                         label = f'{tid}'
    #                     else:
    #                         label = f'{tid}'
    #                     plot_one_box(tlbr, im0_2, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
    #         p = Path(p_2)  # to Path
    #         save_path = str(save_dir / p.name)  # img.jpg

    #         # Print time (inference + NMS)
    #         # print(f'{s}Done. ({t2 - t1:.3f}s)')

    #         # Stream results
    #         if view_img:
    #             # cv2.imshow('BoT-SORT', im0_2)
    #             cv2.imwrite(str(save_dir/"2_{}.jpg".format(frame_2)),im0_2)
    #             cv2.waitKey(1)  # 1 millisecond

    #         # Save results (image with detections)
    #         if save_img:
    #             if not source.endswith('MP4'):
    #                 cv2.imwrite(save_path, im0_2)

    #             else:  # 'video' or 'stream'
    #                 if vid_path_2 != save_path:  # new video
    #                     vid_path_2 = save_path
    #                     if isinstance(vid_writer_2, cv2.VideoWriter):
    #                         vid_writer_2.release()  # release previous video writer
    #                     if video_cap_2:  # video
    #                         fps = video_cap_2.get(cv2.CAP_PROP_FPS)
    #                         w = int(video_cap_2.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                         h = int(video_cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     else:  # stream
    #                         fps, w, h = 30, im0_2.shape[1], im0_2.shape[0]
    #                         save_path += '.mp4'
    #                     vid_writer_2 = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #                 vid_writer_2.write(im0_2)

                
    #     frame_1 +=1
    #     frame_2 +=1




    #     # for i, det in enumerate(pred_1):  # detections per image
    #     #     if webcam:  # batch_size >= 1
    #     #         p, s, im0, frame = source_1[i], '%g: ' % i, img0_1[i].copy(), frame_1
    #     #     else:
    #     #         p, s, im0, frame = source_1, '', img0_1, frame_1

    #     #     # Run tracker
    #     #     detections = []
    #     #     if len(det):
    #     #         boxes = scale_coords(img_1.shape[2:], det[:, :4], im0.shape)
    #     #         boxes = boxes.cpu().numpy()
    #     #         detections = det.cpu().numpy()
    #     #         detections[:, :4] = boxes

    #     #     online_targets_1 = tracker.update(detections, id_feature_1, im0)

    #     #     online_tlwhs = []
    #     #     online_ids = []
    #     #     online_scores = []
    #     #     # online_cls = []
    #     #     for t in online_targets_1:
    #     #         tlwh = t.tlwh
    #     #         tlbr = t.tlbr
    #     #         tid = t.track_id
    #     #         # tcls = t.cls
    #     #         if tlwh[2] * tlwh[3] > opt.min_box_area:
    #     #             online_tlwhs.append(tlwh)
    #     #             online_ids.append(tid)
    #     #             online_scores.append(t.score)
    #     #             # online_cls.append(t.cls)

    #     #             # save results
    #     #             results.append(
    #     #                 f"{i + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
    #     #             )

    #     #             if save_img or view_img:  # Add bbox to image
    #     #                 if opt.hide_labels_name:
    #     #                     label = f'{tid}'
    #     #                 else:
    #     #                     label = f'{tid}'
    #     #                 plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
    #     #     p = Path(p)  # to Path
    #     #     save_path = str(save_dir / p.name)  # img.jpg

    #     #     # Print time (inference + NMS)
    #     #     # print(f'{s}Done. ({t2 - t1:.3f}s)')

    #     #     # Stream results
    #     #     if view_img:
    #     #         cv2.imshow('BoT-SORT', im0)
    #     #         cv2.waitKey(1)  # 1 millisecond

    #     #     # Save results (image with detections)
    #     #     if save_img:
    #     #         if not source.endswith('MP4'):
    #     #             cv2.imwrite(save_path, im0)

    #     #         else:  # 'video' or 'stream'
    #     #             if vid_path != save_path:  # new video
    #     #                 vid_path = save_path
    #     #                 if isinstance(vid_writer, cv2.VideoWriter):
    #     #                     vid_writer.release()  # release previous video writer
    #     #                 if video_cap_1:  # video
    #     #                     fps = video_cap_1.get(cv2.CAP_PROP_FPS)
    #     #                     w = int(video_cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     #                     h = int(video_cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     #                 else:  # stream
    #     #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
    #     #                     save_path += '.mp4'
    #     #                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #     #             vid_writer.write(im0)

    #     #     frame_1 += 1
    #     #     frame_2 += 1
    # video_cap_1.release()
    # video_cap_2.release()
    # vid_writer_1.release()
    # vid_writer_2.release()

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     # print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')

    #----------------------------------------------------------------------------------------------------------------------------------------------

   # has_non_empty_sublist = any(online_target for online_target in online_targets if online_target)
   #  tlbr_name = ['tlbr_{}'.format(i+1) for i in range(len(online_targets))]
   #  tlwh_name = ['tlwh_{}'.format(i+1) for i in range(len(online_targets))]

   #  id_max = 0
   #  global_left_pools = []

   #  if has_non_empty_sublist: # 所有视频中至少有一个出现跟踪目标

   #      for i in range(len(online_targets)):
            
   #          left_pool_indx = []

   #          dists = matching.targets_embedding_distance(online_targets[i],track_pool) 
   #          dists = matching.targets_fuse_iou(dists, online_targets[i], track_pool, tlbr_name[i])
   #          # dists_1 = matching.target_fuse_score(dists_1, online_targets_1)

   #          matches, u_matched, u_track_pool =matching.linear_assignment(dists,thresh = opt.match_thresh - 0.2)
   #          # dists_pp = matching.targets_embedding_distance(track_pool,track_pool) 

   #          for itracked, itrack_pool in matches:

   #              track = online_targets[i][itracked]
   #              pool_tracklet = track_pool[itrack_pool]

   #              # track.track_id = pool_tracklet.track_id
                
   #              feat =track.curr_feat / np.linalg.norm(track.curr_feat) 
   #              pool_tracklet.smooth_feat = 0.9 * pool_tracklet.smooth_feat + (1 - 0.9) * feat
   #              pool_tracklet.smooth_feat /=np.linalg.norm(pool_tracklet.smooth_feat)
   #              pool_tracklet.camera_idx = i
   #              pool_tracklet.score = track.score
   #              setattr(pool_tracklet, tlbr_name[i], track.tlbr)
   #              setattr(pool_tracklet, tlwh_name[i], track.tlwh)


   #          if len(track_pool) == 6 and len(u_matched):
   #              indices_less_twenty = []
   #              left_trks = []
   #              for j in u_matched:
   #                  if online_targets[i][j].tracklet_len < 20:
   #                      indices_less_twenty.append(j)
   #                  else:
   #                      left_trks.append(online_targets[i][j])

   #              # left_trks_1 = [online_targets_1[i] for i in u_matched_1]
   #              left_pools = [track_pool[j] for j in u_track_pool]

   #              left_dists = matching.targets_embedding_distance(left_trks,left_pools) 

   #              left_matches, left_u_matched, left_u_track_pool =matching.linear_assignment(left_dists,thresh=opt.match_thresh + 10)
                
   #              for itracked, itrack_pool in left_matches:
   #                  track = left_trks[itracked]
   #                  pool_tracklet = left_pools[itrack_pool]

   #                  # track.track_id = pool_tracklet.track_id
                    
   #                  feat =track.curr_feat / np.linalg.norm(track.curr_feat) 
   #                  pool_tracklet.smooth_feat = 0.9 * pool_tracklet.smooth_feat + (1 - 0.9) * feat
   #                  pool_tracklet.smooth_feat /=np.linalg.norm(pool_tracklet.smooth_feat)
   #                  # pool_tracklet.tlbr_1 = track_1.tlbr
   #                  pool_tracklet.camera_idx = i
   #                  pool_tracklet.score = track.score

   #                  setattr(pool_tracklet, tlbr_name[i], track.tlbr)
   #                  setattr(pool_tracklet, tlwh_name[i], track.tlwh)

                
   #              for u_m in left_u_track_pool:
   #                  u_m_pool = left_pools[u_m]
   #                  for k, pool in enumerate(track_pool):
   #                      if pool is u_m_pool:
   #                          left_pool_indx.append(k)

                
   #              for j in reversed(indices_less_twenty):
   #                  online_targets[i].pop(j)

   #          if len(track_pool) < 6 and len(u_matched):
   #              thresh_offset = -0.55
   #              indices_less_twenty = []

   #              if len(u_track_pool):
   #                  left_trks = []
   #                  for j in u_matched:
   #                      if online_targets[i][j].tracklet_len < 20:
   #                          indices_less_twenty.append(j)
   #                      else:
   #                          left_trks.append(online_targets[i][j])
   #                  # left_trks_1 = [online_targets_1[i] for i in left_u_matched_1]
   #                  left_pools = [track_pool[i] for i in u_track_pool]

   #                  left_dists = matching.targets_embedding_distance(left_trks, left_pools) 

   #                  left_matches, left_u_matched, left_u_track_pool =matching.linear_assignment(left_dists,thresh=opt.match_thresh + thresh_offset)

   #                  for itracked, itrack_pool in left_matches:
   #                      track = left_trks[itracked]
   #                      pool_tracklet = left_pools[itrack_pool]

   #                      # track.track_id = pool_tracklet.track_id
                        
   #                      feat =track.curr_feat / np.linalg.norm(track.curr_feat) 
   #                      pool_tracklet.smooth_feat = 0.9 * pool_tracklet.smooth_feat + (1 - 0.9) * feat
   #                      pool_tracklet.smooth_feat /=np.linalg.norm(pool_tracklet.smooth_feat)
   #                      # pool_tracklet.tlbr_1 = track.tlbr
   #                      pool_tracklet.camera_idx = i
   #                      pool_tracklet.score = track.score

   #                      setattr(pool_tracklet, tlbr_name[i], track.tlbr)
   #                      setattr(pool_tracklet, tlwh_name[i], track.tlwh)


   #                  for u_m in left_u_matched:

   #                      u_m_track = left_trks[u_m]

   #                      if u_m_track.tracklet_len >= 20 :
   #                          if len(track_pool) < 6:
   #                              # for t in track_pool:
   #                              #     id_max = max(id_max, t.track_id)
   #                              # u_m_track.track_id = id_max + 1
   #                              track_pool.append(copy.deepcopy(u_m_track))
   #                              track_pool[-1].track_id = len(track_pool)
   #                              track_pool[-1].camera_idx = i
   #                              # to_remove_um_indices.append(u_m)
   #                              setattr(track_pool[-1], tlbr_name[i], track_pool[-1].tlbr)
   #                              setattr(track_pool[-1], tlwh_name[i], track_pool[-1].tlwh)

   #                              for j in range(len(tlbr_name)):
   #                                  if not i == j:
   #                                      setattr(track_pool[-1], tlbr_name[j], np.zeros(4))
   #                                      setattr(track_pool[-1], tlwh_name[j], np.zeros(4))

   #                              # track_pool[-1].tlbr_1 = track_pool[-1].tlbr
   #                              # track_pool[-1].tlbr_2 = np.zeros(4)

   #                  for u_m in left_u_track_pool:
   #                      u_m_pool = left_pools[u_m]
   #                      for k, pool in enumerate(track_pool):
   #                          if pool is u_m_pool:
   #                              left_pool_indx.append(k)

   #              else:
   #                  for j in u_matched:
   #                      if online_targets[i][j].tracklet_len < 20:
   #                          indices_less_twenty.append(j)
   #                      else:
   #                          u_m_track = online_targets[i][j]
   #                          if len(track_pool) < 6:
   #                              # for t in track_pool:
   #                              #     id_max = max(id_max, t.track_id)
   #                              # u_m_track.track_id = id_max + 1
   #                              track_pool.append(copy.deepcopy(u_m_track))
   #                              track_pool[-1].track_id = len(track_pool)
   #                              track_pool[-1].camera_idx = i
                                
   #                              # to_remove_um_indices.append(u_m)
   #                              # track_pool[-1].tlbr_1 = track_pool[-1].tlbr
   #                              # track_pool[-1].tlbr_2 = np.zeros(4)
   #                              setattr(track_pool[-1], tlbr_name[i], track_pool[-1].tlbr)
   #                              setattr(track_pool[-1], tlwh_name[i], track_pool[-1].tlwh)

   #                              for j in range(len(tlbr_name)):
   #                                  if not i == j:
   #                                      setattr(track_pool[-1], tlbr_name[j], np.zeros(4))
   #                                      setattr(track_pool[-1], tlwh_name[j], np.zeros(4))

                        
   #              # for j in reversed(indices_less_twenty):
   #              #     online_targets[i].pop(j)

   #          global_left_pools = list(set(global_left_pools).intersection(set(left_pool_indx)))
        
   #      for u_m_pool in global_left_pools:
   #          pool_tracklet = track_pool[u_m_pool]
   #          pool_tracklet.camera_idx = -1

   #  return online_targets, track_pool

