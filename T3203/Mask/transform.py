from torchvision import transforms

class ImageTransform():

    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
<<<<<<< HEAD
            # transforms.Grayscale(num_output_channels=3), 
            transforms.CenterCrop((440,290)),
=======
            # transforms.Grayscale(num_output_channels=1), 
>>>>>>> 01780aff723488d48daace92a4aeb5e3c7053bce
        ])
    
    def __call__(self,img):
        return self.data_transform(img)

