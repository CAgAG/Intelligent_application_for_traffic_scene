import warnings

warnings.filterwarnings("ignore")

import copy
import sys
import os
import multiprocessing
import cv2

from Utils.jsonKit import loadJson, dumpJson
from Utils.carline import get_car_lane
from Utils.abObject import abObject, abArrs
from processShow import pShow

from Utils.scConfig import PRE_JSON

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(os.path.join(curPath, 'theyolo'))

from theyolo.utils.datasets import *
from theyolo.utils.utils import *


class OPT:
    def __init__(self):
        self.weights = 'Weights/yolov5m.pt'
        self.source = ''  # 视频路径, 调用video时是path参数
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.device = ''
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.img_size = 640

        self.car_color = (255, 0, 0)
        self.per_color = (0, 255, 0)
        self.motor_color = (0, 0, 255)

        self.Map = {
            0: 0,  # person
            1: 1, 3: 1,  # motor
            2: 2, 5: 2, 7: 2,  # car
            9: 4  # light
        }
        self.names = (
            'person', 'motor', 'car', 'other', 'light'
        )


class Detect(object):
    def __init__(self, json_file=PRE_JSON):
        super(Detect, self).__init__()
        self.objectArrs = abArrs()
        self.share_arr = multiprocessing.Manager().Queue(10)
        self.Process = multiprocessing.Process(target=pShow, args=(self.share_arr,))
        self.Process.daemon = True

        self.opt = OPT()
        self.device = torch_utils.select_device(self.opt.device)
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

        model = torch.load(self.opt.weights, map_location=self.device)['model'].float()  # load to FP32
        model.to(self.device).eval()
        self.model = model

        self.hand_area_pos = loadJson(json_file)
        self.car_dict = {}
        self.left_lines = []
        self.right_lines = []

    def process(self, img=None, o_img=None):
        # Inference
        pred = self.model(img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                   fast=True, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = o_img[i]
            origin_img = im0
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    label_name = self.opt.names[self.opt.Map.get(int(cls), 3)]
                    score = conf

                    bbox = xyxy
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])

                    pos = [x1, y1, x2, y2]
                    cut_img = im0[y1: y2, x1: x2]
                    tp = abObject()
                    tp.add(cut_img, label_name, pos)
                    self.objectArrs.add(tp)

                    if label_name == 'car':
                        color = self.opt.car_color
                    elif label_name == 'motor':
                        color = self.opt.motor_color
                    elif label_name == 'person':
                        color = self.opt.per_color
                    else:
                        color = (179, 255, 179)

                    cv2.rectangle(origin_img, (x1, y1),
                                  (x2, y2), color, 2, 1)

                    labelSize, baseLine = cv2.getTextSize('{}-{} {:.1f}'.format(
                        tp.id, label_name, float(score)), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    cv2.rectangle(
                        origin_img, (x1, y1 - labelSize[1]), (x1 + labelSize[0], y1 + baseLine), (223, 128, 255),
                        cv2.FILLED)
                    cv2.putText(
                        origin_img, '{}-{} {:.1f}'.format(tp.id, label_name, float(score)),
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return origin_img

    def video(self, path):
        global tmp_img, left_min, right_min
        self.Process.start()
        self.opt.source = path
        dataset = LoadStreams(self.opt.source, img_size=self.opt.img_size)

        # Run inference
        # img = torch.zeros((1, 3, self.opt.img_size, self.opt.img_size), device=self.device)  # init img
        # self.model(img)  # run once
        if 'big' in self.opt.source:
            tmp_img = cv2.imread("./Image/big.jpg")
        elif 'small' in self.opt.source:
            tmp_img = cv2.imread("./Image/small.jpg")
        elif 'mid' in self.opt.source:
            tmp_img = cv2.imread("./Image/mid.jpg")
        elif 'line' in self.opt.source:
            tmp_img = cv2.imread("./Image/line.jpg")
        else:
            tmp_img = ""
        with torch.no_grad():
            first_index = True
            for path, img, im0s, vid_cap in dataset:
                if first_index:
                    first_index = False
                    self.h, self.w = im0s[0].shape[:2]
                    self.objectArrs.height = int(self.w / 4)
                    self.objectArrs.width = int(self.h / 3)
                    self.car_dict = {"left_line": [], "right_line": []}
                    left_min, right_min, self.left_lines, self.right_lines = get_car_lane(tmp_img, self.opt.source)

                    if left_min is None or right_min is None or len(left_min) == 0:
                        print('no line')
                        self.car_dict = loadJson("./Weights/carLine.json")
                        self.left_lines.append(self.car_dict["left_line"][0])
                        self.right_lines.append(self.car_dict["right_line"][0])
                    else:
                        self.car_dict["left_line"].append(left_min)
                        self.car_dict["right_line"].append(right_min)
                        dumpJson(self.car_dict, "./Weights/carLine.json")

                if im0s is None:
                    break

                img = torch.from_numpy(img).to(self.device)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                show_image = self.process(img=img, o_img=im0s)
                self.draw_pre_area(show_image, self.opt.source)
                show_image = cv2.resize(show_image, (int(self.w / 4), int(self.h / 3)))
                cv2.imshow("video", show_image)
                k = cv2.waitKey(1)
                self.share_arr.put(self.objectArrs)
                self.objectArrs.order_pre()
                self.objectArrs.clear()
                if k == ord('q'):
                    break
            cv2.destroyAllWindows()
            self.Process.terminate()

    def draw_pre_area(self, img, path):

        for label, pos in self.hand_area_pos.items():
            for p in pos:
                x1, y1, x2, y2 = p
                ab = abObject()
                ab.add(img[y1:y2, x1:x2].copy(), label, pos)
                self.objectArrs.add(ab)
                if label == 'zebra_crossing' and 'line' in path:
                    continue
                cv2.rectangle(img, (x1, y1),
                              (x2, y2), (179, 255, 179), 2, 1)
                labelSize, baseLine = cv2.getTextSize('{}'.format(
                    label), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(img, (x1, y1 - labelSize[1]), (x1 + labelSize[0], y1 + baseLine), (0, 102, 255),
                              cv2.FILLED)
                cv2.putText(img, '{}'.format(label),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)

        for line in self.left_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
        for line in self.right_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 5)

        # for line in self.left_lines:
        #     x1, y1, x2, y2 = line[0]
        #     if y2 < left_max[3]:
        #         left_max = [x1, y1, x2, y2]
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        #
        # for line in self.right_lines:
        #     x1, y1, x2, y2 = line[0]
        #     if right_max[1] > y1 > left_max[3]:
        #         right_max = [x1, y1, x2, y2]
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # if left_max is not None and right_max is not None:
        #     cv2.line(img, (left_max[2], left_max[3]), (right_max[0], right_max[1]), (0, 255, 0), 2)


if __name__ == '__main__':
    detector = Detect()
    detector.video(r'C:\Users\Administrator\Desktop\jstest\bigroad.mp4')
    # detector.video(r'G:\video-03.avi')
