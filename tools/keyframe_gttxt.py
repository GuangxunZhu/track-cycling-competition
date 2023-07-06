
with open('dataset_path/bicycle-demo_val.txt', "r") as f:
    # 读取文件内容
    path = f.readlines()
    for p in path:
        label_pth = p.strip().replace('images','labels_with_ids').replace('jpg','txt')
        frame = p.split('_')[-1].split('.')[0]
        frame = int(frame)

        with open(label_pth,'r') as lf:
            labels = lf.readlines()
            
            for label in labels:
                label_list = [float(x)  for x in label.strip().split()]

                label_str = '{:d},{:d},{:.6f},{:.6f},{:.6f},{:.6f},1,1,1\n'.format(
                int(frame), int(label_list[1]),label_list[2],label_list[3],label_list[4],label_list[5])
                with open('gt_all.txt', 'a') as f:
                    f.write(label_str)

                if 'FX3' in label_pth:
                    with open('gt_1.txt', 'a') as f:
                        f.write(label_str)

                elif 'C3440' in label_pth:
                    with open('gt_2.txt', 'a') as f:
                        f.write(label_str)

                else:
                    with open('gt_0.txt', 'a') as f:
                        f.write(label_str)


                
        