import numpy as np


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.max([b1_x1, b2_x1])
    inter_rect_y1 = np.max([b1_y1, b2_y1])
    inter_rect_x2 = np.min([b1_x2, b2_x2])
    inter_rect_y2 = np.min([b1_y2, b2_y2])

    # Intersection area
    inter_area = (np.max([0, inter_rect_x2-inter_rect_x1])) * \
                 (np.max([0, inter_rect_y2-inter_rect_y1]))
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    inter_area = inter_area.item()
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


if __name__ == '__main__':
    bbox1 = np.array([0, 0, 3, 3])
    bbox2 = np.array([1, 1, 4, 4])

    print(bbox_iou(bbox1, bbox2))
