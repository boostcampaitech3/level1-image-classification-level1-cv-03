import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# V100서버 터뜨린 모델
class CNN(nn.Module):
    def __init__(self,
                 name='Convolutional Neural Network',
                 xdim=[3,512,384],
                 kernel_size=3,
                 cdims=[36,72],
                 hdims=[1024,128],
                 ydim=18
                ):
        super(CNN,self).__init__()
        self.name=name
        self.xdim=xdim
        self.kernel_size=kernel_size
        self.cdims=cdims
        self.hdims=hdims
        self.ydim=ydim
        
        #Convolutional layers
        self.layers = []
        prev_cdim = self.xdim[0]
        for cdim in self.cdims:
            self.layers.append(
            nn.Conv2d(
                in_channels=prev_cdim,
                out_channels=cdim,
                kernel_size=self.kernel_size,
                stride=(1,1),
                padding=self.kernel_size//2
            ))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            self.layers.append(nn.Dropout2d(p=0.5))
            prev_cdim=cdim
        # Dense layers
        self.layers.append(nn.Flatten())
        prev_hdim = prev_cdim*(self.xdim[1]//2**len(cdims))*(self.xdim[2]//2**len(cdims))
        for hdim in self.hdims:
            self.layers.append(nn.Linear(
            in_features=prev_hdim,
            out_features=hdim,
            bias=True
            ))
            self.layers.append(nn.ReLU(True))
            prev_hdim=hdim
        # Final Layer
        self.layers.append(nn.Linear(prev_hdim,self.ydim,bias=True))
        
        # Concatenate all layers
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.net.add_module(layer_name,layer)

    def forward(self,x):
        return self.net(x)


class MultiMaskModel(nn.Module):
    def __init__(self):
        super().__init__()

        model = resnet18().to(device)
        self.conv_layers = nn.Sequential(*list(model.children())[:-1])
        self.ismask = nn.Sequential(nn.Linear(512,3, bias=True),nn.ReLU())
        self.gender = nn.Sequential(nn.Linear(512,2, bias=True),nn.ReLU())
        self.age = nn.Sequential(nn.Linear(512,1, bias=True))

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        ismask=self.ismask(x)
        gender=self.gender(x)
        age=self.age(x)
        
        return ismask, gender, age

if __name__ == "__main__":
    model = resnet18().to(device)
    print(*list(model.children())[:-1])
    pass