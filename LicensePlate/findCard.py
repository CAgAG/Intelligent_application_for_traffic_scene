# encoding:utf-8
import torch
from torch.autograd import Variable
from load_data import *
from build_models import wR2, fh02

args = {
    'model': 'Weights/fh02.pth',
    # 车牌图片目录
    'input': 'MakeData/RandCard/',
}

numClasses = 4
numPoints = 4
imgSize = (480, 480)
batchSize = 8
use_gpu = True
resume_file = str(args["model"])


class car_number_detector:
    def __init__(self, score_thred=0.31):
        self.provinces = ["湘", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "皖",
                          "粤", "桂",
                          "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        self.alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                          'V', 'W',
                          'X', 'Y', 'Z', 'O']
        self.ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                    'W', 'X',
                    'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        self.score_thred = score_thred
        self.model_conv = fh02(numPoints, numClasses)
        self.model_conv = torch.nn.DataParallel(self.model_conv, device_ids=range(torch.cuda.device_count()))
        self.model_conv.load_state_dict(torch.load(resume_file))
        # use gpu ?
        if use_gpu:
            self.model_conv = self.model_conv.cuda()
        self.model_conv.eval()

    def getCarId(self):
        result = []
        dst = demoTestDataLoader(args["input"].split(','), imgSize)
        trainloader = DataLoader(dst, batch_size=1, shuffle=False)

        for i, (XI, ims) in enumerate(trainloader):

            if use_gpu:
                x = Variable(XI.cuda())
            else:
                x = Variable(XI)
            # Forward pass: Compute predicted y by passing x to the model

            _, y_pred = self.model_conv(x)

            outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
            labelPred = [t[0].index(max(t[0])) for t in outputY]

            lpn = self.alphabets[labelPred[1]] + self.ads[labelPred[2]] + self.ads[labelPred[3]] + self.ads[
                labelPred[4]] + self.ads[labelPred[5]] + self.ads[
                      labelPred[6]]
            result.append(lpn)
        return result

    def isEqual(self, labelGT, labelP):
        compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
        return sum(compare)

    def detect_result(self, image):
        image = cv2.resize(image, imgSize)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype('float32')
        image /= 255.0

        image = torch.from_numpy(image)
        image = Variable(image.unsqueeze(0))

        if use_gpu:
            x = Variable(image.cuda())
        else:
            x = Variable(image)
        # Forward pass: Compute predicted y by passing x to the model

        scores, y_pred = self.model_conv(x)
        scores = scores.mean().data.cpu().numpy()
        # print(scores)
        if scores < self.score_thred:
            return 0,''

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]

        lpn = self.alphabets[labelPred[1]] + self.ads[labelPred[2]] + self.ads[labelPred[3]] + self.ads[
            labelPred[4]] + self.ads[labelPred[5]] + self.ads[
                  labelPred[6]]
        return scores, lpn


# if __name__ == '__main__':
#     import cv2
#     cdetector = car_number_detector()
#     image = cv2.imread('1.jpg')
#     score,lpn = cdetector.detect_result(image)
#     print('车牌识别结果:' , lpn)
#     # print(cdetector.detect_result(image))
#
#     pass
