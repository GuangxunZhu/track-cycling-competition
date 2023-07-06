import os
# 数据集分析，计算剪裁出的每个目标的数量，即不同ID的数量
# 设置文件所在目录
dirs_path = ["fast_reid/datasets/Bicycle-ReID/bounding_box_train","fast_reid/datasets/Bicycle-ReID/bounding_box_test"]

# 用字典记录不同前缀的数量
prefix_counts = {}

# 遍历目录中的所有文件
for dir_path in dirs_path:
    for filename in os.listdir(dir_path):
        # 获取文件名前缀
        prefix = filename[:7]
        
        # 如果前缀不存在于字典中，则添加一个新键
        if prefix not in prefix_counts:
            prefix_counts[prefix] = 1
        # 如果前缀已经存在于字典中，则增加对应的值
        else:
            prefix_counts[prefix] += 1

# 将结果保存至txt文件
with open("prefix_counts.txt", "w") as f:
    for prefix, count in prefix_counts.items():
        f.write(f"{prefix} {count}\n")