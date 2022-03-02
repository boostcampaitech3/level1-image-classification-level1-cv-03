import os
from ready_to_data import Sampling, refine_data
from transform import ImageTransform
from dataset import MaskDataset
from torchvision.models import resnet18

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

images_path = '/opt/ml/backup/input/data/train/images'
sampling_size_rate = 0.2

S = Sampling(images_path,sampling_size_rate)
test_image_list,test_mask_class,test_gender_class,test_age_class,_ = refine_data(images_path,S.test_image_directory_names)
train_image_list,train_mask_class,train_gender_class,train_age_class,_ = refine_data(images_path,S.train_image_directory_names)

trans = ImageTransform()
resnet = resnet18().to(device)
# save_folder = "./runs/"
# save_path = os.path.join(save_folder, "resnet_base_epochs20.pth")   # ./runs/best.pth
# resnet.load_state_dict(torch.load(save_path))
# print(f"{save_path} 에서 성공적으로 모델을 load 하였습니다.")
loss = nn.CrossEntropyLoss()
optm = optim.Adam(resnet.parameters(),lr=1e-3)

train_dataset = MaskDataset(train_image_list,train_mixed_class,trans)
test_dataset = MaskDataset(test_image_list,test_mixed_class,trans)

BATCH_SIZE = 128
train_iter = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_iter = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)



def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.view(-1,3,440,290).to(device))
            _,y_pred = torch.max(model_pred.data,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr

resnet.train() # to train mode 
EPOCHS,print_every = 20,1
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for batch_in,batch_out in train_iter:
        # Forward path
        y_pred = resnet(batch_in.view(-1,3,440,290).to(device))        
        loss_out = loss(y_pred,batch_out.to(device))
        # Update
        # FILL IN HERE      # reset gradient 
        optm.zero_grad()
        # FILL IN HERE      # backpropagate
        loss_out.backward()
        # FILL IN HERE      # optimizer update
        optm.step()
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum/len(train_iter)
    # Print
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_accr = func_eval(resnet,train_iter,device)
        test_accr = func_eval(resnet,test_iter,device)
        print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
               (epoch,loss_val_avg,train_accr,test_accr))


save_folder = "./runs/"
save_path = os.path.join(save_folder, "resnet_base_epochs20.pth")   # ./runs/best.pth
os.makedirs(save_folder, exist_ok=True)  

torch.save(resnet.state_dict(), save_path)
print(f"{save_path} 폴더에 모델이 성공적으로 저장되었습니다.")
print(f"해당 폴더의 파일 리스트: {os.listdir(save_folder)}")