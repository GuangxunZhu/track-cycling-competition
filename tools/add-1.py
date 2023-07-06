import os

path = "dataset_path/bicycle-cut_all.txt"

with open(path, 'r') as file:
        img_files_dict = file.readlines()
        img_files_dict = [x.strip() for x in img_files_dict]
        img_files_dict = list(filter(lambda x: len(x) > 0, img_files_dict))

sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels_with_ids' + os.sep  # /images/, /labels/ substrings
label_files_dict = ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_files_dict]

for label_path in label_files_dict:
    with open(label_path, "r") as f:
        lines = f.readlines()

    # 添加空格和-1到每一行的末尾
    lines_with_space_and_minus_one = [line.strip() + " -1.0" for line in lines]

    with open(label_path, "w") as f:
        f.write("\n".join(lines_with_space_and_minus_one))
