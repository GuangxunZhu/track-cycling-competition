with open('gt_0.txt', 'r') as file:
    lines = file.readlines()

new_lines = []

for line in lines:
    columns = line.split(',')

    if columns[1] == '38':
        columns[1] = '1'
    if columns[1] == '39':
        columns[1] = '2'
    if columns[1] == '40':
        columns[1] = '3'
    columns[2] = '{}'.format(float(columns[2])*3840)
    columns[3] = '{}'.format(float(columns[3])*2160)
    columns[4] = '{}'.format(float(columns[4])*3840)
    columns[5] = '{}'.format(float(columns[5])*2160)

    new_line = ','.join(columns)
    new_lines.append(new_line)

with open('gt_0_new.txt', 'w') as file:
    file.writelines(new_lines)