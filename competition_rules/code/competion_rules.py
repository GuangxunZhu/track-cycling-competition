import json
import numpy as np
import cv2
import time
import datetime

class Competition(object):
    def __init__(self, key_points_path, num_source, frame,video_cap, total_turns, opt, im0sz = [2160,3840]):

        self._height, self._width = im0sz

        self._line0, self._line50, self._line150 = self.get_key_line_points(key_points_path)

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

        # self.system_time = self.record_system_time()
        self.history_keypoint = [[np.zeros(2) for _ in range(num_source)] for _ in range(opt.num_objects)]
        
    def output(self, track_pool,frame):
        
        self.record_current_time(frame)
        self.count_turns(track_pool)
        self.record_key_location(track_pool)
        self.rank(track_pool)
        self.record_competition_time()
        self.record_sprint_time()
        self.calculate_speed(track_pool)
        # self.record_system_time()

        self.update_history_point(track_pool)

        # return self.turns, self.locations, self.rankings, self.competition_time, self.sprint_time, self.speed, self.current_frame

    def count_turns(self, track_pool):
        for idx, track in enumerate(track_pool):
            if track.camera_idx == 0:
                key_point = self.tlbr_to_bl(track.tlbr[track.camera_idx])
                history_key_point = self.history_keypoint[idx][track.camera_idx]
                if self.judge_line_crossed(self._line0, key_point, history_key_point):
                    self.turns[idx] += 1

                
    def record_key_location(self, track_pool):
        for idx, track in enumerate(track_pool) :
            if track.camera_idx == 0:
                key_point = self.tlbr_to_bl(track.tlbr[track.camera_idx])
                history_key_point = self.history_keypoint[idx][track.camera_idx]

                side_line0 = self.classify_point_position(key_point, self._line0)
                side_line50 = self.classify_point_position(key_point, self._line50)
                if side_line0 <=0 and side_line50 >=0 and key_point[1] < self._dividing_line:
                    self.locations[idx] = '0-50'
                elif self.judge_line_crossed(self._line50, key_point, history_key_point) or key_point[1] > self._dividing_line:
                    self.locations[idx] = '50-150'
                elif side_line0 > 0 and key_point[1] < self._dividing_line:
                    self.locations[idx] = '150-250'

            if track.camera_idx == 1:
                key_point = self.tlbr_to_br(track.tlbr[track.camera_idx])
                history_key_point = self.history_keypoint[idx][track.camera_idx]

                side_line150 = self.classify_point_position(key_point, self._line150)

                if side_line150 <= 0:
                    self.locations[idx] = '150-250'
                else:
                    self.locations[idx] = '50-150'


    # history_key_point = self.history_keypoint[idx][track.camera_idx]
        
    def rank(self, track_pool):
        turns = self.turns[:len(track_pool)]
        location_score = [0 if item == '0-50' or item == None else 0.2 if item == '50-150' else 0.6 if item == '150-250' else item for item in self.locations[:len(track_pool)]]
        turns_score = [i + j for i , j in zip(turns, location_score)]
        key_point = [0 for _ in range(len(track_pool))]
        for idx, track in enumerate(track_pool):
            if track.camera_idx != -1 :
                if self.tlbr_to_bl(track.tlbr[track.camera_idx])[1] >= self._dividing_line:
                    key_point[idx] = self.tlbr_to_br(track.tlbr[track.camera_idx])[0]
                else:
                    key_point[idx] = self.tlbr_to_bl(track.tlbr[track.camera_idx])[0]


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
                re_key_point = [key_point[index] for index in indices]
                location_list = [self.locations[index] for index in indices]
                camera_idx_list = [track.camera_idx for track in re_track]
                is_camera_idx_same = all(item == camera_idx_list[0] for item in camera_idx_list)
                if location_list[0] == '0-50':
                    re_rankings_frome_zero = [sorted(re_key_point, reverse = False).index(point)  for point in re_key_point]
                elif location_list[0] == '50-150':
                    if is_camera_idx_same:
                        re_rankings_frome_zero = [sorted(re_key_point, reverse = True).index(point)  for point in re_key_point]
                    else:
                        for idx, camera in enumerate(camera_idx_list) :
                            if camera == 1:
                                re_key_point[idx] += self._width * 2
                            elif camera == -1:
                                re_key_point[idx] += self._width
                        re_rankings_frome_zero = [sorted(re_key_point, reverse = True).index(point)  for point in re_key_point]
                elif location_list[0] == '150-250':
                    if is_camera_idx_same:
                        re_rankings_frome_zero = [sorted(re_key_point, reverse = False).index(point)  for point in re_key_point]
                    else:
                        for idx, camera in enumerate(camera_idx_list) :
                            if camera == 1:
                                re_key_point[idx] += self._width * 2
                            elif camera == -1:
                                re_key_point[idx] += self._width
                        re_rankings_frome_zero = [sorted(re_key_point, reverse = False).index(point)  for point in re_key_point]
                for i, ind in enumerate(indices) :
                    rankings[ind] += re_rankings_frome_zero[i]
        self.rankings = rankings


            
    def record_competition_time(self):
        if not self._time_started and 1 in self.turns:
            self._time_started = True
            self._start_frame = self.current_frame[0]
        if not all(map(lambda x: x == self._total_turns + 1, self.turns)):
            if self._start_frame != 0:
                competition_time =  (self.current_frame[0] - self._start_frame) / self._fps
                self.competition_time = self.convert_seconds_to_time(competition_time)

    def record_sprint_time(self):

        for idx, turns in enumerate(self.turns) :
            if not self._is_time_sprint[idx] and turns == ( self._total_turns - 3 + 1):
                self._is_time_sprint[idx] = True
                self._sprint_frame[idx] = self.current_frame[0]
            
            if ( self._total_turns - 3 + 1) <= turns <= (self._total_turns + 1) and self._is_time_sprint[idx] == True:
                sprint_time = (self.current_frame[0] - self._sprint_frame[idx]) / self._fps
                self.sprint_time[idx] = self.convert_seconds_to_time(sprint_time)

    def calculate_speed(self, track_pool):
         for idx, track in enumerate(track_pool) :
            if track.camera_idx == 0:
                key_point = self.tlbr_to_bl(track.tlbr[track.camera_idx])
                history_key_point = self.history_keypoint[idx][track.camera_idx]
                is_line0_crossed = self.judge_line_crossed(self._line0, key_point, history_key_point)
                is_line50_crossed = self.judge_line_crossed(self._line50, key_point, history_key_point)
                if is_line0_crossed:
                    self._line0_crossed_frame[idx] = self.current_frame[0]
                    if self._line150_crossed_frame[idx] != 0:
                        time = (self._line0_crossed_frame[idx] - self._line150_crossed_frame[idx]) / self._fps
                        speed = (100 / time) / 3.6
                        speed = round(speed, 2)
                        self.speed_150_250[idx].append(speed)
                if is_line50_crossed:
                    self._line50_crossed_frame[idx] = self.current_frame[0]
                    if self._line0_crossed_frame[idx] != 0:
                        time = (self._line50_crossed_frame[idx] - self._line0_crossed_frame[idx]) / self._fps
                        speed = (50 / time) / 3.6
                        speed = round(speed, 2)

                        self.speed_0_50[idx].append(speed)
            if track.camera_idx == 1:
                key_point = self.tlbr_to_br(track.tlbr[track.camera_idx])
                history_key_point = self.history_keypoint[idx][track.camera_idx]
                is_line150_crossed = self.judge_line_crossed(self._line150, key_point, history_key_point)
                if is_line150_crossed:
                    self._line150_crossed_frame[idx] = self.current_frame[1]
                    if self._line50_crossed_frame[idx] != 0:
                        time = (self._line150_crossed_frame[idx] - self._line50_crossed_frame[idx]) / self._fps
                        speed = (100 / time) / 3.6
                        speed = round(speed, 2)
                        self.speed_50_150[idx].append(speed)



    def record_current_time(self, frame):
        self.current_frame = frame
        current_time =  [(f - 1) / self._fps for f in self.current_frame]
        for idx, time in enumerate(current_time):
            self.current_time[idx] = self.convert_seconds_to_time(time)

        
    def record_system_time():
        pass


    def get_key_line_points(self, path):
        # 打开json文件并读取内容
        with open(path) as f:
            data = f.read()
        # 将json字符串转换为Python对象（列表）
        lines = json.loads(data)

        line0 = []
        line50 = []
        line150 = []

        # 遍历lines列表中的每个字典
        for name, coords  in lines.items():
            # 获取字典中线段的名称和坐标字符串
            # = list(line.items())[0]
            # 将坐标字符串中的左右括号和空格去掉，并将坐标用逗号分隔开
            coords = coords.replace('(', '').replace(')', '').replace(' ', '').split(',')
            # 将坐标字符串列表中的元素转换为整型，并分别取出每个点的x和y坐标
            x1, y1, x2, y2, *rest = map(int, coords)
            # 将每个点的坐标存放到相应的列表中
            if name == '0_line':
                line0.append((x1, y1))
                line0.append((x2, y2))
            elif name == '50_line':
                line50.append((x1, y1))
                line50.append((x2, y2))
                line50.append((rest[0], rest[1]))
            elif name == '150_line':
                line150.append((x1, y1))
                line150.append((x2, y2))
                line150.append((rest[0], rest[1]))
        return line0, line50, line150

    def classify_point_position(self, point, linex):
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

    def update_history_point(self, track_pool):
        for idx, track in enumerate(track_pool):
            camera_idx = track.camera_idx

            if camera_idx == 0:
                key_point = self.tlbr_to_bl(track.tlbr[camera_idx])
                self.history_keypoint[idx][camera_idx] = key_point
            elif camera_idx == 1:
                key_point = self.tlbr_to_br(track.tlbr[camera_idx])
                self.history_keypoint[idx][camera_idx] = key_point



    def judge_line_crossed(self, line, curr_point, history_point):

        side_1 = self.classify_point_position(curr_point, line)
        side_2 = self.classify_point_position(history_point, line)
        if side_1 <= 0 and side_2 >= 0:
            return True
        else :
            return False

    def tlbr_to_bl(self, tlbr):
        ret = np.asarray(tlbr).copy()
        bl = np.array([ret[0], ret[3]])
        return bl

    def tlbr_to_br(self, tlbr):
        ret = np.asarray(tlbr).copy()
        br = np.array([ret[2], ret[3]])
        return br
    def convert_seconds_to_time(self, seconds):
        time_delta = datetime.timedelta(seconds=seconds)
        time = datetime.datetime(1, 1, 1) + time_delta
        return time.strftime('%H:%M:%S.%f')[:-3]

if __name__ == '__main__':
    path = "competition_rules/files/key_line_points.json"

    competition = Competition(path)
    competition.count_turns(path)

    # print

