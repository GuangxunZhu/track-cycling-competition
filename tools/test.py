import sys
sys.path.append("/home/zhuguangxun/xxtrack")
from evaluator.mot_eval import Evaluator
import motmetrics as mm
import os

save_dir = 'yolov7_runs/detect/exp10'
result_filename = os.path.join(save_dir, 'labels/result.txt')

# evaluation
accs = []
print('Evaluating')
evaluator = Evaluator('/home/zhuguangxun/datasets','Bicycle-3',  'MOTChallenge') #/home/zhuguangxun/datasets/MOT17/train  MOTChallenge
accs.append(evaluator.eval_file(result_filename, 0, 600)) #'./results/CSTrack/result_MOT17/MOT17-02-SDP.txt' 0 600

# get summary
metrics = mm.metrics.motchallenge_metrics
mh = mm.metrics.create()
summary = Evaluator.get_summary(accs, ['val'], metrics)
strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
# print("detection_num:", result_detection, sum(result_detection))
# print("id_num:", result_id, sum(result_id))
Evaluator.save_summary(summary, os.path.join(save_dir, 'summary.xlsx'))

    