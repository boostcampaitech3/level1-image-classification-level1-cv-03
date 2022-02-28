import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class EfficientNetB7Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

        print(self.model)
        # 미리 학습된 weight들은 고정하고 뒷부분 2단계만 학습
        # fc 제외하고 freeze
        # for n, p in self.model.named_parameters():
        #     if '_fc' not in n:
        #         p.requires_grad = False
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        # self.dropout = nn.Dropout(0.75)
        # self.relu = nn.ReLU()
        # self.l1 = nn.Linear(64 , 2560)
        # self.model._fc = nn.Sequential(
        #     nn.Dropout(0.75),
        #     nn.Linear(in_features=self.model._fc.in_features, out_features=num_classes)
        # )
        
        self.mask = nn.Linear(1280, 3, bias=True)
        self.gender = nn.Linear(1280, 2, bias=True)
        self.age = nn.Linear(1280, 3, bias=True)
        

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        # x = self.model(x)
        # print('21: ', x.shape)

        # x = torch.flatten(x, start_dim=1)
        # print('22: ', x.shape)

        x = self.model.extract_features(x)
        x = F.avg_pool2d(x, x.size()[2:]).reshape(-1, 1280)

        return {
            'mask': self.mask(x),
            'gender': self.gender(x),
            'age': self.age(x)
        }