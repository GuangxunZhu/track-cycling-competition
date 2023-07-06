



with open('dataset_path/bicycle-demo_val.txt', "r") as f:
    # 读取文件内容
    path = f.readlines()
    val_dic = {'0':[],'1':[],'2':[]}
    for p in path:
        name = p.split('/')[-1]
        frame = p.split('_')[-1].split('.')[0]
        frame = int(frame)
        if 'FX3' in name:
            val_dic['1'].append(frame)
        elif 'C3440' in name:
            val_dic['2'].append(frame)
        else:
            val_dic['0'].append(frame)
    print(val_dic)
