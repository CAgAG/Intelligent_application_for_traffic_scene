import math
from Utils import distance
from Utils.idPool import IDPool

"""
整数
计算距离: 大于该距离将视为不是同一物体, 调大将导致性能下降, 调小将导致精度下降
"""
DIS = 90


class abObject:
    def __init__(self):
        # 目标追踪
        self.id = ''
        # 原图像
        self.image = []
        # 标签
        self.label = None
        # 当前位置
        self.pos = []

    def add(self, img, lable: str, pos: list):
        self.label = lable
        self.image = img
        self.pos = pos


class abArrs:
    def __init__(self):
        self.width = 0
        self.height = 0

        self.pre_cars = []
        self.pre_motors = []

        self.cars = []
        self.motors = []
        self.persons = []
        self.lights = []
        self.areas = []

        self.idpool = IDPool(100)
        self.pre_car_ids = set()
        self.pre_motor_ids = set()

    def add(self, ab: abObject):
        if ab.label == 'car':
            min_cd = math.inf
            min_index = -1
            for index, car in enumerate(self.pre_cars):
                cd = distance.cdistance(ab.pos, car.pos).manhattan()
                if cd < min_cd:
                    min_cd = cd
                    min_index = index

            if min_index != -1 and min_cd < DIS:
                ab.id = self.pre_cars[min_index].id
                self.pre_car_ids.add(ab.id)
            else:
                ab.id = self.idpool.getNewID()
            self.cars.append(ab)

        elif ab.label == 'motor':
            min_cd = math.inf
            min_index = -1
            for index, motor in enumerate(self.pre_motors):
                cd = distance.cdistance(ab.pos, motor.pos).manhattan()
                if cd < min_cd:
                    min_cd = cd
                    min_index = index

            if min_index != -1 and min_cd < DIS:
                ab.id = self.pre_motors[min_index].id
                self.pre_motor_ids.add(ab.id)
            else:
                ab.id = self.idpool.getNewID()
            self.motors.append(ab)

        elif ab.label == 'person':
            self.persons.append(ab)
        elif 'light' in ab.label:
            self.lights.append(ab)
        elif 'area' in ab.label:
            self.areas.append(ab)
        else:
            pass

    def clear(self):
        self.pre_cars = self.cars[:]
        self.pre_motors = self.motors[:]

        self.cars.clear()
        self.persons.clear()
        self.motors.clear()
        self.lights.clear()
        self.areas.clear()
        self.pre_car_ids.clear()
        self.pre_motor_ids.clear()

    def order_pre(self):
        for car in self.pre_cars:
            id = car.id
            if id not in self.pre_car_ids:
                self.idpool.releaseID(id)

        for motor in self.pre_motors:
            id = motor.id
            if id not in self.pre_motor_ids:
                self.idpool.releaseID(id)

        pass

    def is_border(self, pos):
        x1, y1, x2, y2 = pos
        if x1 == 0 or y1 == 0:
            return True
        if x2 == self.width or y2 == self.height:
            return True
        return False
