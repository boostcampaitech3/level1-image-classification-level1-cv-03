from ready_to_data import RefineData
from transform import ImageTransform
from dataset import MaskDataset
from torchvision.models import alexnet

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = RefineData()
trans = ImageTransform()
alexnet_model = alexnet().to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(alexnet_model.parameters(),lr=1e-3)

train_dataset = MaskDataset(data.train_image_list,data.train_mixed_class,trans)
test_dataset = MaskDataset(data.test_image_list,data.test_mixed_class,trans)

BATCH_SIZE = 64
train_iter = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_iter = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)



def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.view(-1,3,512,384).to(device))
            _,y_pred = torch.max(model_pred.data,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr

alexnet_model.train() # to train mode 
EPOCHS,print_every = 20,1
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for batch_in,batch_out in train_iter:
        # Forward path
        y_pred = alexnet_model(batch_in.view(-1,3,512,384).to(device))        
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
        train_accr = func_eval(alexnet_model,train_iter,device)
        test_accr = func_eval(alexnet_model,test_iter,device)
        print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
               (epoch,loss_val_avg,train_accr,test_accr))

