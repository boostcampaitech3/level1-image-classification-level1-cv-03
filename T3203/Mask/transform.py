from torchvision import transforms

class ImageTransform():

    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128,96)),
        ])
    
    def __call__(self,img):
        return self.data_transform(img)

