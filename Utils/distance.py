class cdistance:
    def __init__(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        self.b1x_centre = b1_x1 + (b1_x2 - b1_x1) // 2
        self.b1y_centre = b1_y1 + (b1_y2 - b1_y1) // 2

        self.b2x_centre = b2_x1 + (b2_x2 - b2_x1) // 2
        self.b2y_centre = b2_y1 + (b2_y2 - b2_y1) // 2

    def manhattan(self):
        return abs(self.b2x_centre - self.b1x_centre) + \
               abs(self.b2y_centre - self.b1y_centre)

    def chebyshev(self):
        return max(abs(self.b2x_centre - self.b1x_centre),
                   abs(self.b2y_centre - self.b1y_centre))

    def euclidean(self):
        result = (self.b2x_centre - self.b1x_centre) ** 2 + \
                 (self.b2y_centre - self.b1y_centre) ** 2
        return pow(result, 0.5)
