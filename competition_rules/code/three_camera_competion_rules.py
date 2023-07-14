import json
import numpy as np
import cv2
import time
import datetime
import xml.etree.ElementTree as ET
import math
from collections import Counter

class Competition(object):
    def __init__(self, key_points_path, start_time, num_source, frame,video_cap, total_turns, opt, im0sz = [2160,3840]):
        # xmlpath = xmlpath.split(',')
        self._height, self._width = im0sz
        # 打开json文件并读取内容
        with open(key_points_path) as f:
            data = f.read()
        # 将json字符串转换为Python对象（列表）
        lines = json.loads(data)
        
        self._line0, self._line50, self._line150 = self._get_key_line_points(lines)

        self._virtual_line150 =[self._rotate_point( point, (500,3000), -45) for point in self._line150] 

        self._creation_data = start_time


        self._dividing_line = self._height / 2
        self._start_frame = 0
        self._time_started = False
        self._fps = video_cap[0].get(cv2.CAP_PROP_FPS)
        self._total_turns = total_turns

        self._is_time_sprint = [False for _ in range(opt.num_objects)]
        self._sprint_frame = [0 for _ in range(opt.num_objects)]
        self._line0_crossed_frame = [0 for _ in range(opt.num_objects)]
        self._line50_crossed_frame = [0 for _ in range(opt.num_objects)]
        self._line150_crossed_frame = [0 for _ in range(opt.num_objects)]


        self.turns = [0 for _ in range(opt.num_objects)]
        self.locations = [None for _ in range(opt.num_objects)]
        self.rankings = [0 for _ in range(opt.num_objects)]
        self.competition_time = 0
        self.sprint_time = [0 for _ in range(opt.num_objects)]
        self.speed_0_50 = [[0] for _ in range(opt.num_objects)]
        self.speed_50_150 = [[0] for _ in range(opt.num_objects)]
        self.speed_150_250 = [[0] for _ in range(opt.num_objects)]

        self.current_frame = frame
        self.current_time = [0 for _ in range(num_source) ]

        self.system_time = [None for _ in range(num_source)]
        self._history_keypoint = [[np.zeros(2) for _ in range(num_source)] for _ in range(opt.num_objects)]

        self.data_dic = []

        self._frame_on150 = []

        

    def _record_info(self, track_pool):
        for idx, track in enumerate(track_pool):
            if track.camera_idx != -1:
                key_point = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                history_key_point = self._history_keypoint[idx][track.camera_idx]

                is_on_line_0 = (track.camera_idx == 0 and self._judge_line_crossed(self._line0, key_point, history_key_point))
                is_on_line_50 = (track.camera_idx == 1 and self._judge_line_crossed(self._line50, key_point, history_key_point))
                is_on_line_150 = (track.camera_idx == 2 and self._judge_line_crossed(self._virtual_line150, key_point, history_key_point))
                
                if is_on_line_0 or is_on_line_50 or is_on_line_150 :

                    curr_data_list = [data for data in self.data_dic[-len(self.turns):] if data['video_code'] == track.camera_idx]
                    curr_player_list = [data['player'] for data in curr_data_list]

                    if len(curr_data_list) == 1:
                        self._compensating_data(curr_data_list)

                    if track.track_id not in curr_player_list:
                        data = dict()
                        data['video_code'] = track.camera_idx
                        data['player'] = track.track_id
                        data['turns'] = self.turns[idx]
                        data['current_location'] = self.locations[idx]
                        data['current_ranking'] =self.rankings[idx]
                        data['competition_time'] = self.competition_time
                        data['sprint_time'] = self.sprint_time[idx]
                        data['interval_speed'] = [self.speed_0_50[idx][-1],self.speed_50_150[idx][-1], self.speed_150_250[idx][-1]]
                        data['current_time'] = self.current_time[track.camera_idx] + ';' + str(self.current_frame[track.camera_idx])
                        data['system_time'] = self.system_time[track.camera_idx]

                        if is_on_line_150:

                            data['competition_time'] = self._convert_frame150_time(track.camera_idx, idx, self.competition_time)
                            data['sprint_time'] = self._convert_frame150_time(track.camera_idx, idx, self.sprint_time[idx])
                            data['current_time'] = self._convert_frame150_time(track.camera_idx, idx, self.current_time[track.camera_idx]) + ';' + str(self._line150_crossed_frame[idx])
                            data['system_time'] = self.system_time[track.camera_idx].split(' ')[0]+' ' + self._convert_frame150_time(track.camera_idx, idx, self.system_time[track.camera_idx].split(' ')[-1])

                        self.data_dic.append(data)

    def _compensating_data(self, curr_data_list):

        previous_data_list = [data for data in self.data_dic if data not in curr_data_list]

        for turn in range(self._total_turns):

            for camera_idx in range(3):

                data_list = [data for data in previous_data_list if data['turns'] == turn+1 and data['video_code'] == camera_idx]

                if len(data_list) != len(self.turns) and len(data_list):

                    player_list = [data['player'] for data in data_list]
                    miss_player = []
                    for i in range(1, len(self.turns) + 1):
                        if i not in player_list:
                            miss_player.append(i)
                    for miss_id in miss_player:
                        new_data = dict()
                        new_data['video_code'] = camera_idx
                        new_data['player'] = miss_id
                        new_data['turns'] = turn + 1

                        new_data['current_location'] = data_list[len(data_list) // 2]['current_location']
                        new_data['current_ranking'] = data_list[len(data_list) // 2]['current_ranking']
                        new_data['competition_time'] = data_list[len(data_list) // 2]['competition_time']
                        new_data['sprint_time'] = data_list[len(data_list) // 2]['sprint_time']
                        new_data['interval_speed'] = data_list[len(data_list) // 2]['interval_speed']
                        new_data['current_time'] = data_list[len(data_list) // 2]['current_time']
                        new_data['system_time'] = data_list[len(data_list) // 2]['system_time']

                        index = self.data_dic.index(data_list[len(data_list) // 2])
                        self.data_dic.insert(index + 1, new_data)


        

    def _convert_frame150_time(self,camera_idx, track_idx, certain_time):
        
        if certain_time == 0:
            return certain_time

        current_frame = self.current_frame[camera_idx]
        frame150 = self._line150_crossed_frame[track_idx]

        frame_delta = current_frame - frame150

        seconds =  frame_delta / self._fps 

        time_delta = datetime.timedelta(seconds=seconds)

        certain_time = datetime.datetime.strptime(certain_time, '%H:%M:%S.%f') 

        time_frame150 = (certain_time - time_delta).strftime('%H:%M:%S.%f')[:-3]

        return time_frame150


        

    
    def output(self, track_pool, frame, virtual_track):
        
        self._record_current_time(frame)
        self._count_turns(track_pool)
        self._record_key_location(track_pool)
        self._rank(track_pool)
        self._record_competition_time()
        self._record_sprint_time()
        self._calculate_speed(track_pool)
        self._record_system_time()

        self._record_info(track_pool)

        self._update_history_point(track_pool)

        self._record_frameon150(virtual_track)

        # return self.turns, self.locations, self.rankings, self.competition_time, self.sprint_time, self.speed, self.current_frame
    #记录在150m线上的帧
    def _record_frameon150(self, virtual_track):
        if len(virtual_track):
            # 判断列表长度是否大于目标总数
            if len(virtual_track) > len(self.turns):
                # 根据得分对列表进行排序
                sorted_list = sorted(virtual_track, key=lambda x: x.score, reverse=True)
                # 选择得分最高的6个元素组成新的列表
                virtual_track = sorted_list[:len(self.turns)]

            track_bl = [self._tlbr_to_bl(track.tlbr) for track in virtual_track]
            
            bl_x = [tlbl[0] for tlbl in track_bl]
            bl_y = [tlbl[1] for tlbl in track_bl]

            distance_line150 = self._distance_to_line_150(bl_x, bl_y, self._line150) 


            for dis in distance_line150:
                if dis <= 10 :
                    self._frame_on150.append(self.current_frame[2])
            


          
    def _get_value(self, index):
        
        # 判断记录的帧数是否缺少

        while len(self._frame_on150) < len(self.turns):
            middle_index = len(self._frame_on150) // 2
            middle_value = self._frame_on150[middle_index]
            self._frame_on150.insert(middle_index, middle_value)

        # 对列表进行排序
        sorted_lst = sorted(self._frame_on150)

        # 计算相邻元素之间的差值
        diff_values = [sorted_lst[i+1] - sorted_lst[i] for i in range(len(sorted_lst)-1)]

        # 对差值列表进行排序，并取前n个最大的差值
        largest_diff_values = sorted(diff_values, reverse=True)[:len(self.turns) - 1]

        # 根据差值在diff_values中出现的顺序重新排列largest_diff_values
        reordered_diff_values = [diff for diff in diff_values if diff in largest_diff_values]

        merged_lst = []
        merge_flag = False

        for num in reordered_diff_values:
            if num == 1:
                if not merge_flag:
                    merged_lst.append(num)
                    merge_flag = True
            else:
                merged_lst.append(num)
                merge_flag = False

        merge_index = -1
        while len(merged_lst) < len(self.turns) - 1 :          
            merge_index = merged_lst.index(1, merge_index + 1)
            merged_lst.insert(merge_index + 1, 1)  # 在原来合并的位置补上1

        if len(merged_lst) > len(self.turns) - 1:
            merged_lst = merged_lst[:len(self.turns) - 1]

        # 根据差值将列表分为n+1个段
        segments = []
        start_index = 0
        for diff in merged_lst:
            end_index = start_index + diff_values[start_index:].index(diff) + 1
            segments.append(sorted_lst[start_index:end_index])
            start_index = end_index

        # 处理最后一个段
        segments.append(sorted_lst[start_index:])

        target_segment = segments[index]

        frame = int(sum(target_segment) / len(target_segment))

        return  frame



    def _count_turns(self, track_pool):
        for idx, track in enumerate(track_pool):
            if track.camera_idx == 0:
                key_point = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                history_key_point = self._history_keypoint[idx][track.camera_idx]
                if self._judge_line_crossed(self._line0, key_point, history_key_point):
                    self.turns[idx] += 1

            if track.camera_idx == 1:
                unique_value = set(self.turns)
                if len(unique_value) > 1 :
                    counter = Counter(self.turns)
                    most_common_value = counter.most_common(1)[0][0]
                    if self.turns[idx] > most_common_value :
                        self.turns[idx] -= 1
                    elif self.turns[idx] < most_common_value:
                        self.turns[idx] += 1



                
    def _record_key_location(self, track_pool):
        for idx, track in enumerate(track_pool) :
            if track.camera_idx == 0:
                key_point = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                # history_key_point = self._history_keypoint[idx][track.camera_idx]

                side_line0 = self._classify_point_position(key_point, self._line0)
                # side_line50 = self.classify_point_position(key_point, self._line50)
                if side_line0 <=0 :
                    self.locations[idx] = '0-50'
                else:
                    self.locations[idx] = '150-250'
            elif track.camera_idx == 1:
                key_point = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                # history_key_point = self._history_keypoint[idx][track.camera_idx]

                side_line50 = self._classify_point_position(key_point, self._line50)

                if side_line50 <= 0:
                    self.locations[idx] = '50-150'
                else:
                    self.locations[idx] = '0-50'

            elif track.camera_idx == 2:
                self.locations[idx] = '150-250'
                # key_point = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                # history_key_point = self._history_keypoint[idx][track.camera_idx]

                # side_line150 = self._classify_point_position(key_point, self._virtual_line150)

                # if side_line150 <= 0:
                #     self.locations[idx] = '150-250'
                # else:
                #     self.locations[idx] = '50-150'


    # history_key_point = self._history_keypoint[idx][track.camera_idx]
        
    def _rank(self, track_pool):
        turns = self.turns[:len(track_pool)]
        location_score = [0 if item == '0-50' or item == None else 0.2 if item == '50-150' else 0.6 if item == '150-250' else item for item in self.locations[:len(track_pool)]]
        turns_score = [i + j for i , j in zip(turns, location_score)]
        key_point = [np.zeros(2) for _ in range(len(track_pool))]
        for idx, track in enumerate(track_pool):
            if track.camera_idx != -1 :
                if track.camera_idx == 0:
                    key_point[idx] = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                if track.camera_idx == 1:
                    key_point[idx] = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                if track.camera_idx == 2:
                    key_point[idx] = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                # if self.tlbr_to_bl(track.tlbr[track.camera_idx])[1] >= self._dividing_line:
                #     key_point[idx] = self.tlbr_to_br(track.tlbr[track.camera_idx])[0]
                # else:
                #     key_point[idx] = self.tlbr_to_bl(track.tlbr[track.camera_idx])[0]


        # 为相同分数的元素分配相同的排名，跳过下一个排名
        rankings = [sorted(turns_score, reverse=True).index(score) + 1 for score in turns_score]
        # 构建元素到排名的映射
        rank_map = {}
        for i, rank in enumerate(rankings):
            if rank not in rank_map:
                rank_map[rank] = [i]
            else:
                rank_map[rank].append(i)
        # 示例：scores = [85, 92, 78, 90, 85, 88] rankings：[4, 2, 6, 1, 4, 3]  rank_map： {4: [0, 4], 2: [1], 6: [2], 1: [3], 3: [5]}
        for rank, indices in rank_map.items():
            if len(indices) > 1:
                re_track = [track_pool[index] for index in indices]
                re_key_point_x = [key_point[index][0] for index in indices]
                re_key_point_y = [key_point[index][1] for index in indices]
                location_list = [self.locations[index] for index in indices]
                camera_idx_list = [track.camera_idx for track in re_track]
                is_camera_idx_same = all(item == camera_idx_list[0] for item in camera_idx_list)

                if location_list[0] == '0-50':
                    if is_camera_idx_same:
                        re_rankings_frome_zero = [sorted(re_key_point_x, reverse = False).index(point)  for point in re_key_point_x]
                    else:
                        for idx, camera in enumerate(camera_idx_list) :
                            if camera == 1:
                                re_key_point_x[idx] -= self._width * 2
                            elif camera == -1:
                                re_key_point_x[idx] -= self._width
                        re_rankings_frome_zero = [sorted(re_key_point_x, reverse = False).index(point)  for point in re_key_point_x]

                elif location_list[0] == '50-150':
                    if is_camera_idx_same and camera_idx_list[0] == 1:
                        re_rankings_frome_zero = [sorted(re_key_point_x, reverse = False).index(point)  for point in re_key_point_x]
                    elif is_camera_idx_same and camera_idx_list[0] == 2:
                        distance = self._distance_to_line_150(re_key_point_x, re_key_point_y, self._virtual_line150)

                        re_rankings_frome_zero = [sorted(distance, reverse = False).index(d)  for d in distance]
                    else:
                        for idx, camera in enumerate(camera_idx_list) :

                            if camera == 1:
                                re_key_point_x[idx] -= self._width
                                # re_key_point_y[idx] -= self._height * 2
                            elif camera == -1:
                                re_key_point_x[idx] -= self._width * 2
                                # re_key_point_y[idx] -= self._height

                        # distance = self._distance_to_line_150(re_key_point_x, re_key_point_y, self._virtual_line150)
                        re_rankings_frome_zero = [sorted(re_key_point_x, reverse = False).index(point)  for point in re_key_point_x]

                elif location_list[0] == '150-250':
                    if is_camera_idx_same:
                        re_rankings_frome_zero = [sorted(re_key_point_x, reverse = False).index(point)  for point in re_key_point_x]
                    else:
                        for idx, camera in enumerate(camera_idx_list) :
                            if camera == 0:
                                re_key_point_x[idx] -= self._width * 2
                            elif camera == -1:
                                re_key_point_x[idx] -= self._width
                        re_rankings_frome_zero = [sorted(re_key_point_x, reverse = False).index(point)  for point in re_key_point_x]
                for i, ind in enumerate(indices) :
                    rankings[ind] += re_rankings_frome_zero[i]
        self.rankings = rankings

    def _distance_to_line_150(self, point_x, point_y, line):
        distance = []

        for idx, x0 in enumerate(point_x):

            y0 = point_y[idx]
            
            x1, y1 = line[0]
            x2, y2 = line[1]

            numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
            denominator = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
            d = numerator / denominator
            distance.append(d)
        return distance
            
    def _record_competition_time(self):
        if not self._time_started and 1 in self.turns:
            self._time_started = True
            self._start_frame = self.current_frame[0]
        if not all(map(lambda x: x == self._total_turns + 1, self.turns)):
            if self._start_frame != 0:
                competition_time =  (self.current_frame[0] - self._start_frame) / self._fps
                self.competition_time = self._convert_seconds_to_time(competition_time)

    def _record_sprint_time(self):

        for idx, turns in enumerate(self.turns) :
            if not self._is_time_sprint[idx] and turns == ( self._total_turns - 3 + 1):
                self._is_time_sprint[idx] = True
                self._sprint_frame[idx] = self.current_frame[0]
            
            if ( self._total_turns - 3 + 1) <= turns <= (self._total_turns ) and self._is_time_sprint[idx] == True:
                sprint_time = (self.current_frame[0] - self._sprint_frame[idx]) / self._fps
                self.sprint_time[idx] = self._convert_seconds_to_time(sprint_time)

    def _calculate_speed(self, track_pool):
         for idx, track in enumerate(track_pool) :
            if track.camera_idx == 0:
                key_point = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                history_key_point = self._history_keypoint[idx][track.camera_idx]
                is_line0_crossed = self._judge_line_crossed(self._line0, key_point, history_key_point)
                # is_line50_crossed = self.judge_line_crossed(self._line50, key_point, history_key_point)
                if is_line0_crossed:
                    self._line0_crossed_frame[idx] = self.current_frame[0]
                    if self._line150_crossed_frame[idx] != 0:
                        time = (self._line0_crossed_frame[idx] - self._line150_crossed_frame[idx] ) / self._fps
                        speed = (100  / time) * 3.6
                        speed = round(speed, 2)
                        self.speed_150_250[idx].append(speed)

            if track.camera_idx == 1:
                key_point = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                history_key_point = self._history_keypoint[idx][track.camera_idx]
                # is_line0_crossed = self.judge_line_crossed(self._line0, key_point, history_key_point)
                is_line50_crossed = self._judge_line_crossed(self._line50, key_point, history_key_point)
                if is_line50_crossed:
                    self._line50_crossed_frame[idx] = self.current_frame[1]
                    if self._line0_crossed_frame[idx] != 0:
                        time = (self._line50_crossed_frame[idx] - self._line0_crossed_frame[idx]) / self._fps
                        speed = (50 / time) * 3.6
                        speed = round(speed, 2)
                        self.speed_0_50[idx].append(speed)

            if track.camera_idx == 2:
                key_point = self._tlbr_to_bl(track.tlbr[track.camera_idx])
                history_key_point = self._history_keypoint[idx][track.camera_idx]
                is_line150_crossed = self._judge_line_crossed(self._virtual_line150, key_point, history_key_point)
                if is_line150_crossed:

                    self._line150_crossed_frame[idx] =  self._get_value(self.rankings[idx] - 1)

                    if self.rankings[idx] == len(self.rankings) :

                        self._frame_on150 = []

                    if self._line50_crossed_frame[idx] != 0:
                        time = (self._line150_crossed_frame[idx] - self._line50_crossed_frame[idx]) / self._fps
                        speed = (100  / time) * 3.6
                        speed = round(speed, 2)
                        self.speed_50_150[idx].append(speed)



    def _record_current_time(self, frame):
        self.current_frame = frame
        current_time =  [(f - 1) / self._fps for f in self.current_frame]
        for idx, time in enumerate(current_time):
            self.current_time[idx] = self._convert_seconds_to_time(time)

        
    def _record_system_time(self):
        for idx, current in enumerate(self.current_time) :

            # 将格式为 '00.00.00.000' 的时间转换为微秒数
            hours, minutes, seconds = map(float, current.split(':'))

            total_milliseconds = hours * 3600 *1000 + minutes * 60 *1000 + seconds*1000 

            # 创建一个 timedelta 对象，表示要添加的时间
            td = datetime.timedelta(milliseconds = total_milliseconds)

            # 将 timedelta 添加到 datetime 上
            new_dt = self._creation_data[idx] + td

            # 将 datetime 格式化为字符串，包括要添加的时间部分
            self.system_time[idx] = new_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def _rotate_point(self, point, center, angle):
        # 计算相对于旋转中心的原始点的偏移量
        offset_x = point[0] - center[0]
        offset_y = point[1] - center[1]

        # 将角度转换为弧度
        angle_rad = math.radians(angle)

        # 应用旋转变换公式，计算新的偏移量
        new_offset_x = offset_x * math.cos(angle_rad) - offset_y * math.sin(angle_rad)
        new_offset_y = offset_x * math.sin(angle_rad) + offset_y * math.cos(angle_rad)

        # 将新的偏移量添加到旋转中心，得到新的点的坐标
        new_x = int(center[0] + new_offset_x) 
        new_y = int(center[1] + new_offset_y) 

        return (new_x, new_y)

    def _get_key_line_points(self, data):

        line0 = []
        line50 = []
        line150 = []

        # 遍历lines列表中的每个字典
        for name, coords  in data.items():
            # 获取字典中线段的名称和坐标字符串
            # = list(line.items())[0]
            # 将坐标字符串中的左右括号和空格去掉，并将坐标用逗号分隔开
            coords = coords.replace('(', '').replace(')', '').replace(' ', '').split(',')
            # 将坐标字符串列表中的元素转换为整型，并分别取出每个点的x和y坐标
            x1, y1, x2, y2, *rest = map(int, coords)
            # 将每个点的坐标存放到相应的列表中
            if name == 'line_0':
                line0.append((x1, y1))
                line0.append((x2, y2))
            elif name == 'line_50':
                line50.append((x1, y1))
                line50.append((x2, y2))
                # line50.append((rest[0], rest[1]))
            elif name == 'line_150':
                line150.append((x1, y1))
                line150.append((x2, y2))
                # line150.append((rest[0], rest[1]))
        return line0, line50, line150

    def _classify_point_position(self, point, linex):
        """
        判断点(x, y)在三个点构成的两段折线的左边还是右边
        返回值为-1表示点在折线的左侧，为1表示点在折线的右侧，为0表示点在折线上
        """
        x, y = point
        x1, y1 = linex[0]
        x2, y2 = linex[1]

        # 判断点在第一条线段的左侧还是右侧

        position1 = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)

        if len(linex) == 3:
            x3, y3 = linex[2]
            # 判断点在第二条线段的左侧还是右侧
            position2 = (x3 - x2) * (y - y2) - (x - x2) * (y3 - y2)

            if y > y2 : # 
                if position1 < 0 :
                    return -1
                elif position1 > 0:
                    return 1
                else :
                    return 0
            else :
                if position2 < 0 :
                    return -1
                elif position2 > 0:
                    return 1
                else :
                    return 0
                        
        if position1 < 0 :
            return -1
        elif position1 > 0:
            return 1
        else :
            return 0

    def _update_history_point(self, track_pool):
        for idx, track in enumerate(track_pool):
            camera_idx = track.camera_idx
            key_point = self._tlbr_to_bl(track.tlbr[camera_idx])
            self._history_keypoint[idx][camera_idx] = key_point

    def _judge_line_crossed(self, line, curr_point, history_point):

        side_1 = self._classify_point_position(curr_point, line)
        side_2 = self._classify_point_position(history_point, line)
        if side_1 <= 0 and side_2 >= 0:
            return True
        else :
            return False

    def _tlbr_to_bl(self, tlbr):
        ret = np.asarray(tlbr).copy()
        bl = np.array([ret[0], ret[3]])
        return bl

    def _tlbr_to_br(self, tlbr):
        ret = np.asarray(tlbr).copy()
        br = np.array([ret[2], ret[3]])
        return br
    def _convert_seconds_to_time(self, seconds):
        time_delta = datetime.timedelta(seconds=seconds)
        time = datetime.datetime(1, 1, 1) + time_delta
        return time.strftime('%H:%M:%S.%f')[:-3]

    def _get_creation_data(self, path):

        # 解析XML文件
        tree = ET.parse(path)

        # 获取根元素
        root = tree.getroot()

        creation_date_str = root.find('.//{urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.20}CreationDate').attrib['value']
        # 解析ISO 8601日期时间格式
        creation_date = datetime.datetime.fromisoformat(creation_date_str)

        # 格式化日期和时间
        # creation_date_formatted = creation_date.strftime('%Y-%m-%d %H:%M:%S')

        return creation_date



