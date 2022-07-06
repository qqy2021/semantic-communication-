


import os, platform
import pickle
import torch
from math import log
from torch.utils.data import Dataset
from data_loader import Dataset_sentence, collate_func 
from model import make_model,subsequent_mask,make_std_mask,make_decoder,make_dense
from utils import Normlize_tx, Channel, Crit, clip_gradient
import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.optim as optim
import torch.nn as nn
import numpy as np

_iscomplex = True
batch_size = 64
epochs = 61
learning_rate = 1e-4  
epoch_start = 61  

# set path
save_model_path = "./ckpt/"
if 'Windows' in platform.system():
    data_path = r'C:\Users\10091\Desktop\Py\dataset'
else:
    data_path = '/data/zqy/2022'

if not os.path.exists(save_model_path): os.makedirs(save_model_path)

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:1" if use_cuda else "cpu")
data_parallel = False

train_loader_params = {'batch_size': batch_size,
                       'shuffle': True, 'num_workers':8,
                       'drop_last': True}

crossentropyloss=nn.CrossEntropyLoss()

_path = '/data/zqy/2022'
_path = os.path.join(_path, 'datapartfd++.pkl')
tmp = pickle.load(open(_path, 'rb'))
one1 = torch.ones(tmp.shape[0],1)
tmp= torch.cat((one1,tmp),1)
a=one1
for idx,i in enumerate(tmp):
    if i[33] == 1:
        a[idx] = 0
    elif i[34] ==1:
        a[idx] = 1
    elif i[35] ==1:
        a[idx]= 2
tmp1 = tmp[:,:32]
snr = tmp[:,[32]]
tmp1 = tmp1.type(torch.int)
a=a.long()
deal_dataset = TensorDataset(tmp1,snr,a)
train_data_loader = DataLoader(deal_dataset,**train_loader_params)
vocab_size = 32478

tmp_model = make_model(vocab_size,vocab_size,act1=False,act2=False).to(device)  
tmp_model.load_state_dict(torch.load('./ckpt/TRY1part3_epoch{}.pth'.format(epoch_start-1)))
tmp_dense = make_dense(device,16,31).to(device)
channel = Channel(_iscomplex=_iscomplex)

_params = list(tmp_dense.parameters())
optimizer = torch.optim.Adam(_params, lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,20], gamma = 0.2)
crit = Crit()

for name,param in tmp_model.named_parameters():
    param.requires_grad = False


print(123)


def train(model, model2,device, train_loader, optimizer, epoch):
    model.eval()


    print('--------------------epoch: %d' % epoch)

    for batch_idx, train_sents1 in enumerate(train_loader):
        # distribute data to device
        
        train_sents,snr1,label = train_sents1  # with eos

        label=label.to(device)
        optimizer.zero_grad()

        src = train_sents[:, 1:].to(device)

        src_mask = (src != 0).unsqueeze(-2).to(device)
        output= model.encode(src, src_mask).to(device)

        _snr1= snr1.to(device)

        output =model2.forward(output,_snr1).to(device)

        label=label.squeeze()
        loss = crossentropyloss(output,label)
        loss.backward()
        clip_gradient(optimizer, 0.1) 
        optimizer.step()

        if batch_idx%4000==0:
            print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item())


    if epoch%10==0: 
        torch.save(model2.state_dict(),
                   os.path.join(save_model_path, 'TRY1policy_epoch{}.pth'.format(epoch)))
        print("Epoch {} model2 saved!".format(epoch + 1))


# start training
for epoch in range(1, epochs):
    train(tmp_model, tmp_dense,device, train_data_loader, optimizer, epoch)
    scheduler.step()


