

import torch.nn as nn
import os, platform
import torch
from math import log
from data_loader import Dataset_sentence, collate_func
from model import make_model,subsequent_mask,make_std_mask,make_decoder
from utils import Channel, Crit, clip_gradient
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import pickle


class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

sign = LBSign.apply

class DENSE(nn.Module):
    def __init__(self):
        super(DENSE,self).__init__()
        self.layer1=nn.Linear(16,60)
        self.layer2=nn.Linear(60,16)
        
    def Q(self,x):
        return sign(self.layer1(x))
    
    def dQ(self,x):
        return self.layer2(x)

_iscomplex = True
batch_size = 64
epochs = 2
learning_rate = 3e-4  
epoch_start = 81  
# set path
save_model_path = "./ckpt/"
if 'Windows' in platform.system():
    data_path = r'C:\Users\10091\Desktop\Py\dataset'
else:
    data_path = '/data/zqy/act1/dataset'

if not os.path.exists(save_model_path): os.makedirs(save_model_path)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
data_parallel = False

# data loading parameters
train_loader_params = {'batch_size': batch_size,
                       'shuffle': None, 'num_workers':8,
                       'collate_fn': lambda x: collate_func(x),
                       'drop_last': True}
data_train = Dataset_sentence(_path = data_path)
train_data_loader = data.DataLoader(data_train,**train_loader_params)

vocab_size = data_train.get_dict_len()

tmp_model = make_model(vocab_size,vocab_size,act1=False,act2=False).to(device)  
tmp_decoder = make_decoder(vocab_size,vocab_size,N1=32).to(device)


lianghua=DENSE().to(device)
lianghua=lianghua.eval()
tmp_model = tmp_model.eval()
tmp_model.load_state_dict(torch.load('./ckpt/TRY1part3_epoch{}.pth'.format(epoch_start-1)))####################
lianghua.load_state_dict(torch.load('./ckpt/TRY1densepart3_epoch{}.pth'.format(epoch_start-1)))####################



channel = Channel(_iscomplex=_iscomplex)

_params = list(tmp_model.parameters())
optimizer = torch.optim.Adam(_params, lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [20,40], gamma = 0.5)
crit = Crit()


def train(model, model2,device, train_loader, optimizer, epoch):
    tensor3=torch.zeros(1,35)

    for batch_idx, (train_sents, len_batch) in enumerate(train_loader):

        train_sents = train_sents.to(device)  

        len_batch = len_batch.to(device) 
        optimizer.zero_grad()
        src = train_sents[:, 1:]
        trg = train_sents[:, :-1]
        trg_y = train_sents[:, 1:]
        src_mask = (src != 0).unsqueeze(-2).to(device)
        tgt_mask = make_std_mask(trg).to(device)
        output123= model.encode(src, src_mask)
        for _snr1 in  range(-2,8):
            
            _snr3= _snr1
            output=output123


            tensor1=torch.zeros(64,34)
            tensor2=torch.zeros(64,35)
            snr = torch.tensor(_snr1)
            snr1 = torch.zeros(train_sents.shape[0],1)
            snr=snr+snr1
            snr=snr.to(device)
            train_sents1=torch.cat((train_sents,snr),1)
            tensor1[:,:32] = train_sents1[:,1:].to(device)
            tensor2[:,:32] = train_sents1[:,1:].to(device)
            output= model2.Q(output)
            for _snr2 in range(0,2):
                 if _snr2 == 0:
                    output0 = output
                    output1= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output2= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output3= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output4= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output5= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output6= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output7= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output8= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output9= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output10= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output1= sign(output1)
                    output2= sign(output2)
                    output3= sign(output3)
                    output4= sign(output4)
                    output5= sign(output5)
                    output6= sign(output6)
                    output7= sign(output7)
                    output8= sign(output8)
                    output9= sign(output9)
                    output10= sign(output10)
                    output1[:,:,30:]=0.
                    output2[:,:,30:]=0
                    output3[:,:,30:]=0
                    output4[:,:,30:]=0
                    output5[:,:,30:]=0
                    output6[:,:,30:]=0
                    output7[:,:,30:]=0
                    output8[:,:,30:]=0
                    output9[:,:,30:]=0
                    output10[:,:,30:]=0
                    output1= model2.dQ(output1)
                    output2= model2.dQ(output2)
                    output3= model2.dQ(output3)
                    output4= model2.dQ(output4)
                    output5= model2.dQ(output5)
                    output6= model2.dQ(output6)
                    output7= model2.dQ(output7)
                    output8= model2.dQ(output8)
                    output9= model2.dQ(output9)
                    output10= model2.dQ(output10)

                 if _snr2 == 1:
                    output0 = output
                    output1= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output2= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output3= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output4= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output5= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output6= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output7= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output8= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output9= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output10= channel.agwn_physical_layer(output0, _snr=_snr3)
                    output1= sign(output1)
                    output2= sign(output2)
                    output3= sign(output3)
                    output4= sign(output4)
                    output5= sign(output5)
                    output6= sign(output6)
                    output7= sign(output7)
                    output8= sign(output8)
                    output9= sign(output9)
                    output10= sign(output10)
                    output1[:,:,45:]=0.
                    output2[:,:,45:]=0.
                    output3[:,:,45:]=0.
                    output4[:,:,45:]=0.
                    output5[:,:,45:]=0.
                    output6[:,:,45:]=0.
                    output7[:,:,45:]=0.
                    output8[:,:,45:]=0.
                    output9[:,:,45:]=0.
                    output10[:,:,45:]=0.
                    output1= model2.dQ(output1)
                    output2= model2.dQ(output2)
                    output3= model2.dQ(output3)
                    output4= model2.dQ(output4)
                    output5= model2.dQ(output5)
                    output6= model2.dQ(output6)
                    output7= model2.dQ(output7)
                    output8= model2.dQ(output8)
                    output9= model2.dQ(output9)
                    output10= model2.dQ(output10)



                 output1= model.from_channel_emb(output1)
                 output2= model.from_channel_emb(output2)
                 output3= model.from_channel_emb(output3)
                 output4= model.from_channel_emb(output4)
                 output5= model.from_channel_emb(output5)
                 output6= model.from_channel_emb(output6)
                 output7= model.from_channel_emb(output7)
                 output8= model.from_channel_emb(output8)
                 output9= model.from_channel_emb(output9)
                 output10= model.from_channel_emb(output10)
                 output1= model.decode(output1, src_mask,trg, tgt_mask)
                 output2= model.decode(output2, src_mask,trg, tgt_mask)
                 output3= model.decode(output3, src_mask,trg, tgt_mask)
                 output4= model.decode(output4, src_mask,trg, tgt_mask)
                 output5= model.decode(output5, src_mask,trg, tgt_mask)
                 output6= model.decode(output6, src_mask,trg, tgt_mask)
                 output7= model.decode(output7, src_mask,trg, tgt_mask)
                 output8= model.decode(output8, src_mask,trg, tgt_mask)
                 output9= model.decode(output9, src_mask,trg, tgt_mask)
                 output10= model.decode(output10, src_mask,trg, tgt_mask)

                 output1= model.generator.forward(output1)
                 _, output1 = torch.max(output1, dim=-1)
                 output2= model.generator.forward(output2)
                 _, output2 = torch.max(output2, dim=-1)
                 output3= model.generator.forward(output3)
                 _, output3 = torch.max(output3, dim=-1)
                 output4= model.generator.forward(output4)
                 _, output4 = torch.max(output4, dim=-1)
                 output5= model.generator.forward(output5)
                 _, output5 = torch.max(output5, dim=-1)      
                 output6= model.generator.forward(output6)
                 _, output6 = torch.max(output6, dim=-1)    
                 output7= model.generator.forward(output7)
                 _, output7 = torch.max(output7, dim=-1)  
                 output8= model.generator.forward(output8)
                 _, output8 = torch.max(output8, dim=-1)  
                 output9= model.generator.forward(output9)
                 _, output9 = torch.max(output9, dim=-1)  
                 output10= model.generator.forward(output10)
                 _, output10 = torch.max(output10, dim=-1)  

                 for idx,x in enumerate(output1):
                     a=(output1[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if a:
                         tensor1[idx,32+_snr2]+=1
                 for idx,x in enumerate(output2):
                     b=(output2[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if b:
                         tensor1[idx,32+_snr2]+=1
                 for idx,x in enumerate(output3):
                     c=(output3[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if c:
                         tensor1[idx,32+_snr2]+=1
                 for idx,x in enumerate(output4):
                     d=(output4[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if d:
                         tensor1[idx,32+_snr2]+=1
                 for idx,x in enumerate(output5):
                     e=(output5[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if e:
                         tensor1[idx,32+_snr2]+=1  
                 for idx,x in enumerate(output6):
                     e=(output6[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if e:
                         tensor1[idx,32+_snr2]+=1      
                 for idx,x in enumerate(output7):
                     g=(output7[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if g:
                         tensor1[idx,32+_snr2]+=1     
                 for idx,x in enumerate(output8):
                     h=(output8[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if h:
                         tensor1[idx,32+_snr2]+=1     
                 for idx,x in enumerate(output9):
                     h=(output9[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if h:
                         tensor1[idx,32+_snr2]+=1   
                 for idx,x in enumerate(output10):
                     h=(output10[idx][:len_batch[idx]-1] == trg_y[idx][:len_batch[idx]-1]).all()
                     if h:
                         tensor1[idx,32+_snr2]+=1   
                 for idx,x in enumerate(tensor1):
                     if tensor1[idx,32]>=9:
                         tensor2[idx,32]=1
                         tensor2[idx,33]=0
                         tensor2[idx,34]=0
                     elif tensor1[idx,33]>=9:
                         tensor2[idx,32]=0
                         tensor2[idx,33]=1
                         tensor2[idx,34]=0
                     else:
                         tensor2[idx,32]=0
                         tensor2[idx,33]=0
                         tensor2[idx,34]=1
            tensor3=torch.cat((tensor3,tensor2),0)

        if batch_idx%1000==0:
            print(batch_idx)

    return tensor3




for epoch in range(0, 1):
    tensor3=train(tmp_model, lianghua,device, train_data_loader, optimizer, epoch)
    tensor3=tensor3[1:]
    with open("datapartfd++.pkl", "wb") as f:
        pickle.dump(tensor3, f)


