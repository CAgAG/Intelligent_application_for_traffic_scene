import numpy as np
import cv2
import matplotlib.pyplot as plt

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 70  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 60
max_line_gap = 30



global w, h


def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def hough_lines(img, rho, theta, threshold,
                min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return get_lanes(lines, img)


def get_lanes(lines, img):
    left_lines, right_lines = [], []
    if lines is None:
        return [None] * 4
    for line in lines:
        for x1, y1, x2, y2 in line:
            if getDis(x1, y1, x2, y2) < 10000:
                continue
            if x1 < img.shape[1] / 2 and x2 < img.shape[1] / 2:
                left_lines.append(line)
            else:
                right_lines.append(line)
    if len(left_lines) <= 0 and len(right_lines) <= 0:
        return [None] * 4

    d = []  # left, right, dis

    if len(left_lines) == 0 and right_lines is not None:
        left_lines.append(right_lines[0])

    for rline in right_lines:
        rx1, ry1, rx2, ry2 = rline[0]
        for lline in left_lines:
            lx1, ly1, lx2, ly2 = lline[0]
            dis1 = getDis(lx1, ly1, rx1, ry1)
            dis2 = getDis(lx2, ly2, rx2, ry2)
            dis = min(dis1, dis2)
            d.append([lline[0], rline[0], dis])

    d.sort(key=lambda x: x[2])  # 按照距离排序
    if d is None or len(d) == 0:
        return [None] * 4
    lmin_x1, lmin_y1, lmin_x2, lmin_y2 = d[0][0]
    rmin_x1, rmin_y1, rmin_x2, rmin_y2 = d[0][1]

    return [int(lmin_x1), int(lmin_y1), int(lmin_x2), int(lmin_y2)], [int(rmin_x1), int(rmin_y1), int(rmin_x2),
                                                                      int(rmin_y2)], left_lines, right_lines


def getDis(x1, y1, x2, y2):
    return (y2 - y1) ** 2 + (x2 - x1) ** 2


def get_car_lane(img, path):
    global blur_ksize
    roi_vtx = np.array([[(2151, 2113), (2949, 2148), (2039, 179), (3235, 2137)]])
    if 'big' in path:
        roi_vtx = np.array([[(1294, 1015), (703, 1743), (3140, 1694), (2712, 1026)]])
    elif 'small' in path:
        roi_vtx = np.array([[(1317, 1078), (549, 2145), (3834, 1924), (3209, 1109)]])
    elif 'mid' in path:
        roi_vtx = np.array([[(1377, 1303), (963, 2153), (3834, 2148), (3489, 1443)]])
        blur_ksize = 7
    elif 'line' in path:
        roi_vtx = np.array([[(2070, 350), (2879, 2119), (3252, 2119), (2200, 350)]])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges = roi_mask(edges, roi_vtx)
    # plt.imshow(edges)
    # plt.show()

    return hough_lines(roi_edges, rho, theta, threshold,
                       min_line_length, max_line_gap)


if __name__ == '__main__':
    # video_cap = cv2.VideoCapture(r"C:\Users\Administrator\Desktop\jstest\bigroad.mp4")
    # ret, img = video_cap.read()
    img = cv2.imread("../Image/big.jpg")
    h, w = img.shape[0], img.shape[1]
    # i = 0
    # while True:
    #     ret, img = video_cap.read()
    #     if i % 20 == 0 and i > 80:
    #         cv2.imwrite("./test"+str(i)+".jpg",img)
    #     i = i + 1
    left_min, right_min, left_lines, right_lines = get_car_lane(img, "big")

    if left_lines is None or right_lines is None:
        # print('no line')
        pass
    else:
        for line in left_lines:
            print(line[0])
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
        print('--')
        for line in right_lines:
            print(line[0])
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 5)

    img = cv2.resize(img, (w // 2, h // 2))
    cv2.imshow("line", img)
    cv2.waitKey(0)
