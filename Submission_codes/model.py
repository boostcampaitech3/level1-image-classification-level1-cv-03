from json.tool import main
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch


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
class Res50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pretrain_model = torchvision.models.resnext50_32x4d(pretrained=True)
        self.pretrain_model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True) # resnet18.fc의 in_features의 크기는?

    def forward(self, x):
        x = self.pretrain_model.forward(x)
        return x

# Custom Model Template
class Res18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pretrain_model = torchvision.models.resnet18(pretrained=True)
        self.pretrain_model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True) # resnet18.fc의 in_features의 크기는?

    def forward(self, x):
        x = self.pretrain_model.forward(x)
        return x

# Custom Model Template
class Res183Ways(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.res18 = Res18(num_classes)
        self.res18.load_state_dict(torch.load("./model/res18/best.pth"))
        self.res18 = nn.Sequential(*list(self.res18.pretrain_model.children())[:-1])

        self.mask = nn.Linear(512, 3, bias=True)
        self.age = nn.Linear(512, 3, bias=True)
        self.gender = nn.Linear(512, 3, bias=True)

        def dfs_freeze(model):
            for name, child in model.named_children():
                for param in child.parameters():
                    #print(param)
                    param.requires_grad = False
                    #print(param)
                dfs_freeze(child)
        dfs_freeze(self.res18)
    

    def forward(self, x):
        x = self.res18.forward(x)
        x = torch.flatten(x, start_dim=1)
        m = self.mask(x)
        a = self.age(x)
        s = self.gender(x)
        return {"mask":m, "age":a, "gender":s}

# Custom Model Template
class Res503Ways(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.res50 = Res50(num_classes)
        self.res50.load_state_dict(torch.load("./model/res50/best.pth"))
        self.res50 = nn.Sequential(*list(self.res50.pretrain_model.children())[:-1])

        self.mask = nn.Linear(2048, 3, bias=True)
        self.age = nn.Linear(2048, 3, bias=True)
        self.gender = nn.Linear(2048, 3, bias=True)

        def dfs_freeze(model):
            for name, child in model.named_children():
                for param in child.parameters():
                    #print(param)
                    param.requires_grad = False
                    #print(param)
                dfs_freeze(child)
        dfs_freeze(self.res50)
    

    def forward(self, x):
        x = self.res50.forward(x)
        x = torch.flatten(x, start_dim=1)
        m = self.mask(x)
        a = self.age(x)
        s = self.gender(x)
        return {"mask":m, "age":a, "gender":s}
