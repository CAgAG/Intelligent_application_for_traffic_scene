import torch
from torch import nn
import cv2
from torchvision import transforms, models


class cls_model:
    def __init__(self, path, label_list):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label_list = label_list

        self.tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = self.get_models(path)

    def get_models(self, path):
        model_ft = models.MobileNetV2()
        model_ft.classifier = nn.Linear(model_ft.last_channel, len(self.label_list))
        model_ft.load_state_dict(torch.load(path))

        model_ft = model_ft.to(self.device)
        model_ft.eval()
        return model_ft

    def detect_result(self, img):
        img = cv2.resize(img, (256, 256))
        img = self.tfs(img)
        img = img.unsqueeze(0).to(self.device)
        out = self.model(img)
        _, pred = torch.max(out, 1)
        return self.label_list[pred]


if __name__ == '__main__':
    cls = cls_model()
    img = cv2.imread('./test-g.png')
    out = cls.detect_result(img)
    print(out)
    pass
