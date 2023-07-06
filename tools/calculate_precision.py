import numpy as np

import torch


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

num_camera = 3

for i in range(num_camera):
    num_targets = 0
    num_acc = 0
    gt_frame_dict = dict()
    results_frame_dict = dict()
    with open('gt_{}_new.txt'.format(i),'r') as gt:
        labels = gt.readlines()
        # num_targets = len(labels)

        for line in labels:
                linelist = line.split(',')
                fid = int(linelist[0])
                gt_frame_dict.setdefault(fid, list())
                score = 1
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                gt_frame_dict[fid].append((tlwh, target_id, score))
    with open('yolov7_runs/detect/exp47/labels/result_{}.txt'.format(i)) as re:
        trks = re.readlines()
        num_targets = len(trks)

        for line in trks:
                linelist = line.split(',')
                fid = int(linelist[0])
                results_frame_dict.setdefault(fid, list())

                score = float(linelist[6])
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_frame_dict[fid].append((tlwh, target_id, score))

    frames = sorted(list(set(results_frame_dict.keys())))

    for frame_id in frames:
        trk_objs = results_frame_dict.get(frame_id, [])
        gt_objs = gt_frame_dict.get(frame_id, [])

        if len(trk_objs) > 0:
            trk_tlwhs, trk_ids, trk_scores = zip(*trk_objs)
        else:
            trk_tlwhs, trk_ids, trk_scores = [], [], []
        trk_tlwhs = np.asarray(trk_tlwhs, dtype=float).reshape(-1, 4)
        trk_ids = np.asarray(trk_ids)


        if len(gt_objs) > 0:
            gt_tlwhs, gt_ids, gt_scores = zip(*gt_objs)
        else:
            gt_tlwhs, gt_ids, gt_scores = [], [], []
        gt_tlwhs = np.asarray(gt_tlwhs, dtype=float).reshape(-1, 4)
        gt_ids = np.asarray(gt_ids)

        for j in range(len(trk_ids)):
            trk_tlwh = trk_tlwhs[j]

            trk_tlbr = trk_tlwh.copy()
            trk_tlbr[:2] -= trk_tlbr[2:]/2
            trk_tlbr[2:] += trk_tlbr[:2]
            trk_tlbr = torch.from_numpy(trk_tlbr).unsqueeze(0)

            trk_id = trk_ids[j]

            for k in range(len(gt_ids)):

                gt_tlwh = gt_tlwhs[k]

                gt_tlbr = gt_tlwh.copy()
                gt_tlbr[:2] -= gt_tlbr[2:]/2
                gt_tlbr[2:] += gt_tlbr[:2]
                gt_tlbr = torch.from_numpy(gt_tlbr).unsqueeze(0)

                gt_id = gt_ids[k]

                iou = box_iou (trk_tlbr, gt_tlbr)
                iou = float(iou)
                if iou >= 0.1 and gt_id == trk_id:
                    num_acc += 1

    accuracy = num_acc / num_targets
    print(accuracy)



        


