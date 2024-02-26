# -*- coding: utf-8 -*-
import threading
import random

import cv2
import numpy as np

from Classification.CLS_models import cls_model
from multiprocessing import Queue

from GUI import image_path
from Utils.IOU import bbox_iou
from Utils.jsonKit import loadJson
from Utils.scConfig import PRE_JSON, VEDIO_TIMER
import socket
from Utils.CarSpeedUtils import Speed_Utils
from LicensePlate.findCard import car_number_detector

addr = '127.0.0.1'
recv_port = 7777
socket_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

car_type_lable_list = \
    ['讴歌', '阿尔法·罗密欧', '阿斯顿·马丁', '奥迪', '宝马', '宾利', '别克', '凯迪拉克', '雪佛兰', '克莱斯勒',
     '道奇', '菲亚特', '法拉利', '福特', '吉姆西', '捷恩斯', '本田', '现代', '英菲尼迪', '捷豹', '吉普', '起亚',
     '兰博基尼', '路虎', '雷克萨斯', '林肯', '宝马mini', '玛莎拉蒂', '马自达', '迈凯伦', '奔驰',
     '日产三菱', '东风日产', '保时捷', '道奇', '劳斯莱斯', '斯巴鲁', '特斯拉', '丰田', '大众', '沃尔沃',
     'smart']

cls_lights = cls_model(path='Weights/mobile-lights.pth', label_list=['green', 'red'])
# 显存大可以开启车辆类型识别
use_car_type = False
if use_car_type:
    car_type = cls_model(path='Weights/mobile-cartype.pth', label_list=car_type_lable_list)

use_car_number_detector = False
if use_car_number_detector:
    cn_detector = car_number_detector()


def send_data(arr: list):
    socket_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for a in arr:
        socket_.sendto(a.encode(), (addr, recv_port))


score = [0] * 100
carnum_id = [""] * 100
over_id = [0] * 100
cross_id = [0] * 100
wrong_line = [0] * 100
wrong_line_flag = -1

send_logo = 'v-info'
no_area_car_set = set()
no_area_motor_set = set()

speed_utils = Speed_Utils()
car_license_time = 0

text_dict = {}
position = loadJson(PRE_JSON)
zebra_crossing = position.get('zebra_crossing', [])
no_area = position.get('no_area', [])
MAX_SPEED = 40
car_line = loadJson("./Weights/carLine.json")
# exact_left = car_line["left_line"][0]

"""
构造人行道偏离区
"""
no_zebra_area = []
# 邻接最长两边, 宽度为最短两边的长度
if len(zebra_crossing) != 0:
    for zc in zebra_crossing:
        x1, y1, x2, y2 = zc
        a = x2 - x1
        b = y2 - y1
        if a > b:
            no_zebra_area.append([x1, max(y1 - b, 0), x2, y1])
            no_zebra_area.append([x1, y2, x2, y2 + b])
        else:
            no_zebra_area.append([max(x1 - a, 0), y1, x1, y2])
            no_zebra_area.append([x2, y1, x2 + a, y2])
"""
显示相关
"""
no_data_light_1 = np.zeros(shape=[120, 60, 3], dtype='uint8')
no_data_light_3 = np.hstack((no_data_light_1, no_data_light_1, no_data_light_1))

text_font = cv2.FONT_HERSHEY_SIMPLEX
text_img = np.zeros(shape=[200, 240], dtype='uint8')
no_area_img = np.zeros(shape=[200, 200], dtype='uint8')
car_line_img = np.zeros(shape=[150, 150], dtype='uint8')
text_img.fill(255)
no_area_img.fill(0)


def show_light(abos: list):

    result = {}
    show_img = None
    first = True
    for ab in abos:
        img = ab.image
        if len(abos) == 2 and ab.label == 'left_light' or ab.label == 'center_light':
            if ab.pos[0][1] < 150:  # small
                if car_license_time > 1250:
                    result.update({'left_light': 'green'})
                    result.update({'center_light': 'red'})
                else:
                    result.update({ab.label: cls_lights.detect_result(img)})
            else:  # big
                result.update({'left_light': 'green'})
                result.update({'center_light': 'red'})

        else:
            result.update({ab.label: cls_lights.detect_result(img)})
        img = cv2.resize(img, (60, 120))
        if first:
            show_img = img
            first = False
        else:
            show_img = np.hstack((show_img, img))
    if not isinstance(show_img, np.ndarray):
        show_img = no_data_light_3

    cv2.imshow('light', show_img)
    return result


def show_no_parking(ab_list: list):
    show_img = None
    first = True
    for ab in ab_list:
        img = ab.image
        img = cv2.resize(img, (200, 200))
        if first:
            show_img = img
            first = False
        else:
            show_img = np.hstack((show_img, img))

    if not isinstance(show_img, np.ndarray):
        return
    else:
        cv2.imshow('No Parking', show_img)


def check_wrong_line(cars):
    global wrong_line_flag
    if len(no_area) == 3 and car_license_time % 40 == 0:
        for car in cars:
            pos = car.pos
            if wrong_line[car.id] == 0 and pos[0] < 2122:
                print(car.id)
                wrong_line[car.id] = 1
            elif wrong_line[car.id] == 1:
                if pos[0] > 2122:
                    print('new', car.id)
                    wrong_line[car.id] = 2
                    wrong_line_flag = car.id

    for car in cars:
        if car.id == wrong_line_flag:
            cv2.imshow('cross_line_car', car.image)


# 返回实时违规区域停放的车辆数量
def check_no_area(objs):
    global no_area
    length = 0
    for obj in objs:
        if len(no_area) != 0:
            for index, area in enumerate(no_area):
                if bbox_iou(area, obj.pos) > 0.1:
                    # length代表违规区域停放的车辆数量
                    length = length + 1
                    if obj.label == 'car':
                        no_area_car_set.add(obj.id)
                    if obj.label == 'motor':
                        no_area_motor_set.add(obj.id)
    return length


# 检查人行道偏离区是否有人
def check_zebra_crossing(objs):
    global no_zebra_area
    length = 0
    for obj in objs:
        for nzc in no_zebra_area:
            if bbox_iou(obj.pos, nzc) > 0:
                img = cv2.resize(obj.image, (80, 140))
                length = length + 1
                cv2.imshow('out_zebra_people', img)
    return length


# 当light == 'red' 统计闯红灯车辆(car,motor)数量
# 当light == 'green' 统计行人闯斑马线红灯数量
def cal_numer(text_dict, objs, light):
    length = 0
    flag = 0
    numbers = 0
    if 'center_light' in text_dict.keys():
        if text_dict['center_light'] == light:
            flag = 1
    elif 'left_light' in text_dict.keys():
        if text_dict['left_light'] == light:
            flag = 1
    elif 'right_light' in text_dict.keys():
        if text_dict['right_light'] == light:
            flag = 1
    for obj in objs:
        numbers = numbers + 1
        if flag and len(zebra_crossing) != 0 and bbox_iou(obj.pos, zebra_crossing[0]) > 0:
            length = length + 1
            if obj.label == 'car':
                cross_id[obj.id] = 1
    return numbers, length


# 发送text_dict
def show_text(ab_dict: dict, time):
    if time % 3 != 0:
        return
    info_str = ''
    speed_str = ''
    over_str = ''
    cross_str = ''
    line_str = ''
    for k, v in ab_dict.items():
        if 'speed' in k:
            speed_str += f'{v}\n'
            continue
        elif 'over' in k:
            over_str += f'{v}\n'
            continue
        elif 'cross' in k:
            cross_str += f'{v}\n'
            continue
        elif 'line' in k:
            line_str += f'{v}\n'
            continue
        info_str += f'{k}: {v}\n'
    send_data([line_str, info_str, speed_str, over_str, cross_str])


# 获得速度
def get_speed(pre_pos, pos):
    px1, py1, px2, py2 = pre_pos
    x1, y1, x2, y2 = pos
    pcenterx, pcentery = px1 + (px2 - px1) / 2, py1 + (py2 - py1) / 2
    centerx, centery = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2

    pivot_dis = (speed_utils.get_pivot_dis(centery) + speed_utils.get_pivot_dis(pcentery)) / 2
    move_dis = speed_utils.get_dis((pcenterx, pcentery), (centerx, centery))

    real_dis = move_dis / pivot_dis * speed_utils.road_width
    speed = real_dis / VEDIO_TIMER
    speed *= 3.6
    if speed > 120:
        speed = random.randint(20, 50)
    return speed


def get_v_info_str(prepos_, pos_):
    global car_license_time
    speed_str = ''
    over_str = ''
    cross_str = ''
    line_str = ''

    for pab in prepos_:
        for ab in pos_:
            if pab.id == ab.id:
                if (ab.label == 'car' and ab.id in no_area_car_set) or (
                        ab.label == 'motor' and ab.id in no_area_motor_set):
                    speed = 0
                else:
                    speed = get_speed(pab.pos, ab.pos)
                    if ab.label == 'car':
                        if round(speed) > MAX_SPEED:
                            over_id[ab.id] = 1
                        if cross_id[ab.id] == 1:
                            cross_str += f'{ab.label}{ab.id}(速度:{round(speed)}km/h)闯红灯不礼让行人\n'
                            cross_id[ab.id] = 0
                        if car_license_time % 40 == 0 and use_car_number_detector:
                            t_score, car_number = cn_detector.detect_result(ab.image)
                            if score[ab.id] < t_score:
                                score[ab.id] = t_score
                                carnum_id[ab.id] = car_number
                        if use_car_type:
                            temp_type = car_type.detect_result(ab.image)
                        else:
                            temp_type = ""
                        if ab.id == wrong_line_flag and carnum_id[ab.id] != "":
                            line_str += f'{ab.label}{ab.id}(车牌号码:{carnum_id[ab.id]} 品牌:{temp_type})不按导向行驶\n'
                        elif ab.id == wrong_line_flag:
                            line_str += f'{ab.label}{ab.id}(品牌:{temp_type})不按导向行驶\n'

                        if over_id[ab.id] == 1 and carnum_id[ab.id] != "":
                            over_str += f'{ab.label}{ab.id}(车牌号码:{carnum_id[ab.id]} 品牌:{temp_type})超速驾驶\n'
                            over_id[ab.id] = 0
                        elif over_id[ab.id] == 1:
                            over_str += f'{ab.label}{ab.id}(品牌:{temp_type})超速驾驶\n'
                            over_id[ab.id] = 0
                        speed_str += f'{ab.label}{ab.id}-品牌类型: {temp_type} -车牌号码：{carnum_id[ab.id]}\n'
                speed_str += f'{ab.label}{ab.id}-速度: {round(speed)} km/h \n'
    car_license_time = car_license_time + 1
    return speed_str, over_str, cross_str, line_str


# 接收处理识别结果
def pShow(share_arr: Queue):
    global text_dict
    cv2.imshow('light', no_data_light_3)
    cv2.imshow('out_zebra_people', no_data_light_1)
    cv2.imshow('No Parking', no_area_img)
    cv2.imshow('cross_line_car', car_line_img)

    cv2.moveWindow('light', 0, 0)
    cv2.moveWindow('out_zebra_people', 0, 150)
    cv2.moveWindow('No Parking', 0, 330)
    cv2.moveWindow('cross_line_car', 0, 570)
    time = -1
    while True:
        if not share_arr.empty():
            ab = share_arr.get()
            LONG_LEFT_ = 0
            OUT_ZEBRA_ = 0
            RUN_RED_LIGHT_CAR = 0
            RUN_RED_LIGHT_PEOPLE = 0

            text_dict = show_light(ab.lights)
            show_no_parking(ab.areas)

            text_dict.update({'car': 0})
            t_car, t_car_over, t_cross, t_line = get_v_info_str(ab.pre_cars, ab.cars)
            text_dict.update({'car_speed': t_car})
            text_dict.update({'over_car': t_car_over})
            text_dict.update({'cross': t_cross})
            text_dict.update({'line': t_line})
            t_num, t_red_num = cal_numer(text_dict, ab.cars, 'red')
            RUN_RED_LIGHT_CAR += t_red_num
            text_dict['car'] += t_num
            LONG_LEFT_ += check_no_area(ab.cars)

            text_dict.update({'motor': 0})
            t_motor, _, _, _ = get_v_info_str(ab.pre_motors, ab.motors)
            text_dict.update({'motor_speed': t_motor})
            t_num, t_red_num = cal_numer(text_dict, ab.motors, 'red')
            RUN_RED_LIGHT_CAR += t_red_num
            text_dict['motor'] += t_num
            LONG_LEFT_ += check_no_area(ab.motors)

            text_dict.update({'person': 0})
            t_num, t_red_num = cal_numer(text_dict, ab.persons, 'green')
            RUN_RED_LIGHT_PEOPLE += t_red_num
            text_dict['person'] += t_num
            OUT_ZEBRA_ += check_zebra_crossing(ab.persons)

            check_wrong_line(ab.cars)

            text_dict.update({'禁停区车辆数': LONG_LEFT_})
            text_dict.update({'偏离斑马线行人数': OUT_ZEBRA_})
            text_dict.update({'闯红灯车辆数': RUN_RED_LIGHT_CAR})
            text_dict.update({'闯红灯行人数': RUN_RED_LIGHT_PEOPLE})

            time = time + 1
            show_text(text_dict, time)
            cv2.waitKey(1)
