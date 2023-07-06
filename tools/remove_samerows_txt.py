# 打开文件并读取内容
with open('yolov7_runs/detect/exp14/labels/result.txt', 'r') as f:
    lines = f.readlines()

# 剔除重复的行
lines = list(set(lines))

# 将剔除重复行的结果写回文件
with open('file.txt', 'w') as f:
    for line in lines:
        f.write(line)