import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
import os


class EmotionRecognition(object):
    def __init__(self, device, gpu_id=0,*, model_file):
        assert device == 'cpu' or device == 'gpu'
        if torch.cuda.is_available():
            if device == 'cpu':
                print('[*]Warning: Your device have GPU, for better performance do EmotionRecognition(device=gpu)')
                self.device = torch.device('cpu')
            if device == 'gpu':
                self.device = torch.device(f'cuda:{str(gpu_id)}')
        else:
            if device == 'gpu':
                print('[*]Warning: No GPU is detected, so cpu is selected as device')
                self.device = torch.device('cpu')
            if device == 'cpu':
                self.device = torch.device('cpu')
        # self.network = NetworkV2(in_c=1, nl=32, out_f=7).to(self.device)
        self.model_file = model_file

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # model_dict = torch.load(os.path.join(os.path.dirname(__file__), 'model', 'model.pkl'), map_location=self.device)
        network = torch.load(os.path.join(os.path.dirname(__file__), self.model_file),
                             map_location=self.device)
        # accuracy = torch.load(os.path.join(os.path.dirname(__file__), 'model', 'ResNet18_lr_0.1_accuracy.pt'))
        # print(f'Accuracy - loaded: {accuracy}')

        self.emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        # self.network.load_state_dict(model_dict['network'])

        self.network = network
        self.network.eval()

    def _predict(self, image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.network(tensor)
        ps = torch.exp(output).tolist()
        index = np.argmax(ps)
        return self.emotions[index]

    def recognise_emotion(self, frame, return_type='BGR'):
        f_h, f_w, c = frame.shape
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        boxes, _ = self.mtcnn.detect(frame)
        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = int(round(boxes[i][0])), int(round(boxes[i][1])), int(round(boxes[i][2])), int(
                    round(boxes[i][3]))
                emotion = self._predict(gray[y1:y2, x1:x2])
                frame = cv.rectangle(frame, (x1, y1), (x2, y2), color=[0, 255, 0], thickness=1)
                frame = cv.rectangle(frame, (x1, y1 - int(f_h*0.03125)), (x1 + int(f_w*0.125), y1), color=[0, 255, 0], thickness=-1)
                frame = cv.putText(frame, text=emotion, org=(x1 + 5, y1 - 3), fontFace=cv.FONT_HERSHEY_PLAIN,
                                   color=[0, 0, 0], fontScale=1, thickness=1)

            if return_type == 'BGR':
                return True, frame
            if return_type == 'RGB':
                return True, cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        else:
            print('No face detected')
            if return_type == 'BGR':
                return False, frame
            if return_type == 'RGB':
                return False, cv.cvtColor(frame, cv.COLOR_BGR2RGB)
