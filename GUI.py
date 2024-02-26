# -*-coding: utf-8 -*-
import socket
import time

import cv2
from tkinter import filedialog
import tkinter as tk  # 导入GUI界面函数库
import threading
import os
import sys
import numpy as np
from Utils.jsonKit import dumpJson, loadJson

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(os.path.join(curPath, 'LicensePlate'))
sys.path.append(os.path.join(curPath, 'theyolo'))

import demo

addr = '127.0.0.1'
recv_port = 7777
socket_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

position = {
    # 左灯
    'left_light': [],
    # 右灯
    'right_light': [],
    # 中间的灯
    'center_light': [],
    # 斑马线
    'zebra_crossing': [],
    # 禁停区
    'no_area': []
}

json_name = ""
image_path = ""

color_flag = (0, 0, 255)

lights = []
zebra = []
no_area = []


def is_like(mu_ban, img_gray, threshold, k_w, k_h, type):
    template = cv2.imread(mu_ban, 0)
    h, w = template.shape[:2]
    true_res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(true_res >= threshold)
    if type == 'light':
        for pt in zip(*loc[::-1]):  # *号表示可选参数
            lights.append([pt[0] * k_w, pt[1] * k_h, (pt[0] + w) * k_w, (pt[1] + h) * k_h])
    elif type == 'zebra':
        for pt in zip(*loc[::-1]):
            zebra.append([pt[0] * k_w, pt[1] * k_h, (pt[0] + w) * k_w, (pt[1] + h) * k_h])
    elif type == 'no_area':
        for pt in zip(*loc[::-1]):
            no_area.append([pt[0] * k_w, pt[1] * k_h, (pt[0] + w) * k_w, (pt[1] + h) * k_h])


def pre_do(img_name):
    if 'line' in img_name:
        zebra.append([464, 766, 3500, 1408])
        no_area.append([60, 459, 630, 745])
        no_area.append([799, 342, 1180, 480])
        no_area.append([3223, 456, 3780, 680])
    elif 'mid' in img_name:
        zebra.append([8, 924, 3466, 1315])
    elif 'small' in img_name:
        zebra.append([13, 735, 3814, 1029])
        lights.append([1970, 128, 2030, 155])
        lights.append([2039, 128, 2127, 155])
    elif 'big' in img_name:
        zebra.append([18, 705, 3782, 996])
        lights.append([1885, 207, 1933, 237])
        lights.append([1935, 202, 1976, 237])


def open_file():
    # 打开文件选择窗口选择图片，图片名赋值给image_path
    global image_path, position
    image_path = filedialog.askopenfilename()
    if image_path == "":
        print("no image path")
    else:
        position.clear()
        lights.clear()
        no_area.clear()
        zebra.clear()
        right_text.delete('1.0', 'end')
        info_text.delete('1.0', 'end')
        illegal_text.delete('1.0', 'end')
        illegal_text1.delete('1.0', 'end')
        illegal_text2.delete('1.0', 'end')
        cv2.destroyAllWindows()
        position = {
            # 左灯
            'left_light': [],
            # 右灯
            'right_light': [],
            # 中间的灯
            'center_light': [],
            # 斑马线
            'zebra_crossing': [],
            # 禁停区
            'no_area': []
        }
        video_cap = cv2.VideoCapture(image_path)
        ret, image = video_cap.read()
        h, w = image.shape[:2]
        if ret is True:
            k_w = w / (w // 2)
            k_h = h / (h // 2)
            res_img = cv2.resize(image, (w // 2, h // 2))
            img_gray = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
            is_like('Image/coin1.png', img_gray, 0.8, k_w, k_h, 'light')
            is_like('Image/coin2.png', img_gray, 0.8, k_w, k_h, 'light')
            is_like('Image/coin3.png', img_gray, 0.8, k_w, k_h, 'light')
            is_like('Image/coin4.png', img_gray, 0.8, k_w, k_h, 'light')
            is_like('Image/coin5.png', img_gray, 0.8, k_w, k_h, 'light')
            is_like('Image/coin6.png', img_gray, 0.9, k_w, k_h, 'light')
            is_like('Image/coin7.png', img_gray, 0.95, k_w, k_h, 'zebra')
            is_like('Image/coin8.png', img_gray, 0.90, k_w, k_h, 'zebra')
            is_like('Image/coin9.png', img_gray, 0.99, k_w, k_h, 'zebra')
            is_like('Image/coin10.png', img_gray, 0.99, k_w, k_h, 'no_area')
            is_like('Image/coin11.png', img_gray, 0.90, k_w, k_h, 'no_area')
            pre_do(image_path)
        cv2.imshow("video", res_img)
        if len(lights) == 1:
            position['center_light'].append(
                [int(lights[0][0]), int(lights[0][1]), int(lights[0][2]), int(lights[0][3])])
        elif len(lights) == 2:
            position['left_light'].append(
                [int(lights[0][0]), int(lights[0][1]), int(lights[0][2]), int(lights[0][3])])
            position['center_light'].append(
                [int(lights[1][0]), int(lights[1][1]), int(lights[1][2]), int(lights[1][3])])
        elif len(lights) == 3:
            position['left_light'].append([int(lights[0][0]), int(lights[0][1]), int(lights[0][2]), int(lights[0][3])])
            position['center_light'].append(
                [int(lights[1][0]), int(lights[1][1]), int(lights[1][2]), int(lights[1][3])])
            position['right_light'].append([int(lights[2][0]), int(lights[2][1]), int(lights[2][2]), int(lights[2][3])])
        if len(zebra) == 1:
            position['zebra_crossing'].append([int(zebra[0][0]), int(zebra[0][1]), int(zebra[0][2]), int(zebra[0][3])])
        if len(no_area) != 0:
            for val in no_area:
                position['no_area'].append([int(val[0]), int(val[1]), int(val[2]), int(val[3])])
        # print(position)


def learn():
    global json_name, image_path, position
    socket_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cv2.destroyAllWindows()
    dumpJson(position, 'Weights/handrec.json')
    if json_name != "":
        detect = demo.Detect(json_file=json_name)
    else:
        detect = demo.Detect()
    detect.video(image_path)
    socket_.close()


def start_learn():
    global position
    th1 = threading.Thread(target=learn)
    th2 = threading.Thread(target=counter)
    th1.start()
    th2.start()


def counter():
    try:
        socket_.bind(('127.0.0.1', recv_port))
    except OSError:
        pass

    while True:
        k = socket_.recv(10240).decode()
        if ('-速度' in k) or ('-品牌类型' in k):
            right_text.delete(1.0, 'end')
            right_text.insert(tk.INSERT, k)
        elif '车辆' in k:
            info_text.delete(1.0, 'end')
            info_text.insert(tk.INSERT, k)
        elif '超速' in k:
            illegal_text.delete(1.0, 'end')
            illegal_text.insert(tk.INSERT, k)
        elif '不按导向' in k:
            illegal_text2.delete(1.0, 'end')
            illegal_text2.insert(tk.INSERT, k)
        elif '礼让行人' in k:
            illegal_text1.delete(1.0, 'end')
            illegal_text1.insert(tk.INSERT, k)


if __name__ == '__main__':
    window = tk.Tk()  # 创建一个Tkinter.Tk()实例
    window.title('Intelligent transportation system')
    window.geometry('690x880+500-100')  # 几何
    imgLabel0 = tk.Label(window, text="基于计算机视觉的交通场景智能应用", fg='black', font=('微软雅黑', 16))  #
    imgLabel0.place(x=100, y=0)
    button = tk.Button(window, text='选择视频', width=12, height=2, command=open_file)
    button.place(x=50, y=50)
    photo1 = tk.PhotoImage(file="./Image/1.gif")
    imgLabel1 = tk.Label(window, image=photo1, width=60, height=50)
    imgLabel1.place(x=150, y=46)
    close_btn = tk.Button(window, text='开始识别', width=12, height=2, command=start_learn)
    close_btn.place(x=350, y=50)
    photo9 = tk.PhotoImage(file="./Image/9.gif")
    imgLabel9 = tk.Label(window, image=photo9, width=60, height=50)
    imgLabel9.place(x=450, y=46)
    result = tk.Label(text='实时路口交通流量统计', fg='black', font=('微软雅黑', 13))
    result.place(x=10, y=150, anchor='sw')
    info_text = tk.Text(window, width=30, height=10)
    info_text.place(x=10, y=160, anchor='nw')
    result = tk.Label(text='汽车类型、车牌号及实时车速', fg='black', font=('微软雅黑', 13))
    result.place(x=350, y=150, anchor='sw')
    right_text = tk.Text(window, width=45, height=35)
    right_text.place(x=350, y=160, anchor='nw')
    result = tk.Label(text='超速行驶违规', fg='black', font=('微软雅黑', 13))
    result.place(x=10, y=305, anchor='nw')
    illegal_text = tk.Text(window, width=45, height=10)
    illegal_text.place(x=10, y=345, anchor='nw')
    result = tk.Label(text='车辆不礼让行人违规', fg='black', font=('微软雅黑', 13))
    result.place(x=10, y=495, anchor='nw')
    illegal_text1 = tk.Text(window, width=45, height=10)
    illegal_text1.place(x=10, y=525, anchor='nw')
    result = tk.Label(text='不按导向行驶违规', fg='black', font=('微软雅黑', 13))
    result.place(x=10, y=680, anchor='nw')
    illegal_text2 = tk.Text(window, width=45, height=10)
    illegal_text2.place(x=10, y=715, anchor='nw')
    window.mainloop()
