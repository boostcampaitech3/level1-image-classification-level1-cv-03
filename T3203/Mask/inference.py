import os
import torch
import pandas as pd
from dataset import TestDataset
from transform import ImageTransform
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_dir = '/opt/ml/backup/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')


image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
trans = ImageTransform()
dataset = TestDataset(image_paths,trans)
loader = DataLoader(dataset,shuffle=False)


model = resnet18().to(device)
save_folder = "./runs/"
save_path = os.path.join(save_folder, "resnet_centercrop_grayscale_epochs20.pth")   # ./runs/best.pth
model.load_state_dict(torch.load(save_path))
print(f"{save_path} 에서 성공적으로 모델을 load 하였습니다.")
model.eval()

all_predictions = []
for images in tqdm(loader):
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission_resnet_centercrop_grayscale_epochs20.csv'), index=False)
print('test inference is done!')

