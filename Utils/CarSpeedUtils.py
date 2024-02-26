import math
import numpy as np
import json


class Speed_Utils:
    def __init__(self, Path='Weights/carLine.json'):
        self.road_width = 3.25  # m

        # with open(Path, encoding='utf-8') as f:
        #     self.CarLine = json.load(f)
        self.CarLine = {"left_line": [[1712, 1143, 1825, 1144]], "right_line": [[1958, 1403, 1963, 1138]]}
        self.left_line = self.CarLine.get('left_line')[0]
        self.right_line = self.CarLine.get('right_line')[0]

        self.lwb, self.rwb = self.car_line_wb()

    def get_function_from_points(self, x1, y1, x2, y2):
        w = (y2 - y1) / max(x2 - x1, 0.000001)
        b = -(x1 * w) + y1
        return [w, b], lambda x: w * x + b

    def get_dis(self, dis1, dis2):
        x1, y1 = dis1
        x2, y2 = dis2
        cd = pow(y2 - y1, 2) + pow(x2 - x1, 2)
        return pow(cd, 0.5)

    def car_line_wb(self):
        lwb, _ = self.get_function_from_points(*self.left_line)
        rwb, _ = self.get_function_from_points(*self.right_line)
        return lwb, rwb

    def get_pivot_dis(self, y):
        pivot_dis = (y - self.rwb[1]) / self.rwb[0] - (y - self.lwb[1]) / max(self.lwb[0], 0.000001)
        pivot_dis = (y - self.rwb[1]) / self.rwb[0] - (y - self.lwb[1]) / max(self.lwb[0], 0.000001)
        return abs(pivot_dis)

    def angle2radian(self, x):
        return x / 180 * math.pi

    def radian2angle(self, x):
        return x * 180 / math.pi

    def int2angle(self, x):
        return np.arctan(x)

    def add_angle(self, x, add_angle):
        return np.tan(self.angle2radian(self.int2angle(x) + add_angle))


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    speed_utils = Speed_Utils('../Weights/carLine.json')
    lfunc = lambda x: speed_utils.lwb[0] * x + speed_utils.lwb[1]
    rfunc = lambda x: speed_utils.rwb[0] * x + speed_utils.rwb[1]

    x = [i for i in range(100)]
    ly = []
    ry = []
    for i in x:
        ly.append(lfunc(i))
        ry.append(rfunc(i))

    plt.plot(x, ly, color='r')
    plt.plot(x, ry, color='g')
    plt.show()
    pass
