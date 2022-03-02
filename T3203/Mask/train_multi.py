import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ready_to_data import Sampling, refine_data
from transform import ImageTransform
from dataset import MultiMaskDataset
from model import MultiMaskModel

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

images_path = '/opt/ml/backup/input/data/train/images'
sampling_size_rate = 0.2

S = Sampling(images_path,sampling_size_rate)
train_image_list,train_mask_class,train_gender_class,_,train_age,_ = refine_data(images_path,S.train_image_directory_names)
test_image_list,test_mask_class,test_gender_class,_,test_age,_ = refine_data(images_path,S.test_image_directory_names)

trans = ImageTransform()

train_dataset = MultiMaskDataset(image_list=train_image_list,ismask=train_mask_class,gender=train_gender_class,age=train_age,transform=trans)
test_dataset = MultiMaskDataset(test_image_list,test_mask_class,test_gender_class,test_age,trans)

BATCH_SIZE = 64
train_iter = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_iter = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

model = MultiMaskModel().to(device)
loss = nn.CrossEntropyLoss()
loss_l1 = nn.L1Loss()
optm = optim.Adam(model.parameters(),lr=3e-4)

save_folder = "/opt/ml/workspace/T3203/runs/"
save_path = os.path.join(save_folder, "resnet_centercrop_grayscale_multilayer_epochs20.pth")   # ./runs/best.pth
model.load_state_dict(torch.load(save_path))
print(f"{save_path} 에서 성공적으로 모델을 load 하였습니다.")


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
    return torch.tensor(answer)


def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        ismask_total,ismask_correct = 0,0
        gender_total,gender_correct = 0,0
        age_total,age_correct = 0,0
        model.eval()
        for image,ismask,gender,age in data_iter:
            ismask, gender, age = ismask.to(device), gender.to(device),age.to(device)
            prev_ismask,prev_gender,prev_age = model(image.view(-1,3,440,290).to(device))
            _,prev_mask = torch.max(prev_ismask,1)
            _,prev_gend = torch.max(prev_gender,1)
            m = (prev_mask==ismask)
            g = (prev_gend==gender)
            a = (check_age_class(prev_age)==check_age_class(age)).to(device)
            n = (a==(m==g))
            ismask_correct += m.sum().item()
            ismask_total += ismask.size(0)
            gender_correct += g.sum().item()
            gender_total += gender.size(0)
            age_correct += a.sum().item()
            age_total += age.size(0)
            n_correct += n.sum().item()
            n_total += age.size(0)
        ismask_accr = (ismask_correct/ismask_total)
        gender_accr = (gender_correct/gender_total)
        age_accr = (age_correct/age_total)
        n_accr = (n_correct/n_total)
        model.train()
    return n_accr,ismask_accr,gender_accr,age_accr


model.train()
EPOCHS,print_every = 20,1
for epoch in tqdm(range(EPOCHS)):
    loss_total_sum = 0
    loss_ismask_sum = 0
    loss_gender_sum = 0
    loss_age_sum = 0
    for image,ismask,gender,age in train_iter:
        ismask, gender, age = ismask.to(device), gender.to(device),age.to(device)
        prev_ismask,prev_gender,prev_age = model(image.view(-1,3,440,290).to(device))
        loss_ismask = loss(prev_ismask,ismask)
        loss_gender = loss(prev_gender,gender)
        loss_age = loss_l1(prev_age,age)
        loss_total = loss_ismask+loss_gender+loss_age
        optm.zero_grad()
        loss_total.backward()
        optm.step()
        loss_ismask_sum += loss_ismask.item()
        loss_gender_sum += loss_gender.item()
        loss_age_sum += loss_age.item()
        loss_total_sum += loss_total.item()
    loss_ismask_avg = loss_ismask_sum/len(train_iter)
    loss_gender_avg = loss_gender_sum/len(train_iter)
    loss_age_avg = loss_age_sum/len(train_iter)
    loss_total_avg = loss_total_sum/len(train_iter)

    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_n_acc,train_ismask_accr,train_gender_accr,train_age_accr = func_eval(model,train_iter,device)
        test_n_acc,test_ismask_accr,test_gender_accr,test_age_accr = func_eval(model,test_iter,device)
        print(f"epoch:{epoch}")
        print(f"total loss:{loss_total_avg}, ismask loss:{loss_ismask_avg}, gender loss:{loss_gender_avg}, age loss:{loss_age_avg}")
        print("train")
        print(f"total accr:{train_n_acc}, ismask accr:{train_ismask_accr}, gender accr:{train_gender_accr}, age accr:{train_age_accr}")
        print("test")
        print(f"total accr:{test_n_acc}, ismask accr:{test_ismask_accr}, gender accr:{test_gender_accr}, age accr:{test_age_accr}")


save_folder = "/opt/ml/workspace/T3203/runs/"
save_path = os.path.join(save_folder, "resnet_centercrop_grayscale_multilayer_epochs20.pth")   # ./runs/best.pth
os.makedirs(save_folder, exist_ok=True)  

torch.save(model.state_dict(), save_path)
print(f"{save_path} 폴더에 모델이 성공적으로 저장되었습니다.")
print(f"해당 폴더의 파일 리스트: {os.listdir(save_folder)}")

# def scaling(x):
#     m = x.mean()
#     s = x.std()
#     return (x-m)/s

if __name__ == "__main__":
    # image,ismask,gender,age = next(iter(train_iter))
    # image,ismask,gender,age = image.to(device),ismask.to(device),gender.to(device),age.to(device)
    # prev_ismask,prev_gender,prev_age = model(image.view(-1,3,440,290).to(device))
    
    # _,prev_mask = torch.max(prev_ismask,1)
    # _,prev_gend = torch.max(prev_gender,1)
    # m = (prev_mask==ismask)
    # g = (prev_gend==gender)
    # a = (check_age_class(prev_age)==check_age_class(age)).to(device)
    # n = (a==(m==g))
    # print(n)
    


    # print(age,len(age),type(age),prev_age,len(age),type(prev_age))

    # prev_age,prev_gender,prev_age = model(next(iter(train_iter))[0].view(-1,3,440,290)).to(device)
    # print(prev_ismask,prev_gender,prev_age)
    pass