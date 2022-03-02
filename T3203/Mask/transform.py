from torchvision import transforms

class ImageTransform():

    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3), 
            transforms.CenterCrop((440,290)),
        ])
    
    def __call__(self,img):
        return self.data_transform(img)

