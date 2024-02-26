import os
import sys
from utils.datasets import *
from utils.utils import *

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(os.path.join(curPath, 'theyolo'))

class OPT:
    def __init__(self):
        self.weights = 'Weights/yolov5l.pt'
        self.source = r'E:\comp\smallroad.mp4'
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.device = ''
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.img_size = 640

        self.car_color = (255, 0, 0)
        self.per_color = (0, 255, 0)
        self.driver_color = (0, 0, 255)

        self.Map = {
            0: 0, # person
            1: 1, 3: 1, # driver
            2: 2, 5: 2, 7: 2, # car
            9: 4 # light
        }
        self.names = (
            'person', 'driver', 'car', 'other', 'light'
        )


def detect():
    opt = OPT()

    source, weights, imgsz = \
        opt.source, opt.weights, opt.img_size

    # Initialize
    device = torch_utils.select_device(opt.device)

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()

    # Set Dataloader
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz)

    # Get names and colors
    names = opt.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    model(img) # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p,  im0 = path[i],  im0s[i].copy()

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[opt.Map.get(int(cls), 3)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[opt.Map.get(int(cls), 3)], line_thickness=3, opt=opt)

            # Stream results
            im0 = cv2.resize(im0, (0, 0), fx=0.3, fy=0.3)
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration


if __name__ == '__main__':
    with torch.no_grad():
        detect()
