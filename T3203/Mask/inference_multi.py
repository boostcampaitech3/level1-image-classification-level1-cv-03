import os
import torch
import pandas as pd
from dataset import TestDataset
from transform import ImageTransform
from torch.utils.data import DataLoader
from model import MultiMaskModel
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_dir = '/opt/ml/backup/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')


image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
trans = ImageTransform()
dataset = TestDataset(image_paths,trans)
loader = DataLoader(dataset,shuffle=False)


model = MultiMaskModel().to(device)
save_folder = "/opt/ml/workspace/T3203/runs/"
save_path = os.path.join(save_folder, "resnet_centercrop_grayscale_multilayer_epochs20.pth")   # ./runs/best.pth
model.load_state_dict(torch.load(save_path))
print(f"{save_path} 에서 성공적으로 모델을 load 하였습니다.")
model.eval()

all_predictions = []
for images in tqdm(loader):
    with torch.no_grad():
        images = images.to(device)
        prev_ismask, prev_gender, prev_age = model(images)
        m = prev_ismask.argmax(dim=-1)
        g = prev_gender.argmax(dim=-1)

        def check_age_class(x):
            answer = []
            x_cpu = x.cpu().squeeze(0)
            for i in x_cpu:
                if i < 30:
                    answer.append(0)
                elif i >= 30 and i < 60:
                    answer.append(1)
                else:
                    answer.append(2)
            return torch.tensor(answer).to(device)

        a = check_age_class(prev_age)
        pred = m*6 + g*3 + a
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission_resnet_centercrop_grayscale_multilayer_epochs20.csv'), index=False)
print('test inference is done!')

if __name__ =="__main__":

    # image = next(iter(loader)).to(device)
    


    pass