


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
        self.layer1=nn.Linear(16,30)
        self.layer2=nn.Linear(30,16)
        
    def Q(self,x):
        return sign(self.layer1(x))
    
    def dQ(self,x):
        return self.layer2(x)
        
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
lianghua=DENSE().to(device)

_iscomplex = True
batch_size = 64
epochs = 61
learning_rate = 3e-4  
epoch_start = 61  

# set path
save_model_path = "./ckpt/"
if 'Windows' in platform.system():
    data_path = r'C:\Users\10091\Desktop\Py\dataset'
else:
    data_path = '/data/zqy/act1/dataset'

if not os.path.exists(save_model_path): os.makedirs(save_model_path)

data_parallel = False

# data loading parameters
train_loader_params = {'batch_size': batch_size,
                       'shuffle': True, 'num_workers':4,
                       'collate_fn': lambda x: collate_func(x),
                       'drop_last': True}
data_train = Dataset_sentence(_path = data_path)
train_data_loader = data.DataLoader(data_train,**train_loader_params)

vocab_size = data_train.get_dict_len()

tmp_model = make_model(vocab_size,vocab_size,act1=False,act2=False).to(device)  
tmp_decoder = make_decoder(vocab_size,vocab_size,N1=32).to(device)
tmp_model.load_state_dict(torch.load('./ckpt/TRY1_epoch{}.pth'.format(epoch_start-1)))
lianghua.load_state_dict(torch.load('./ckpt/TRY1dense_epoch{}.pth'.format(60)))
tmp_decoder.load_state_dict(torch.load('./ckpt/TRY1decoder1_epoch{}.pth'.format(60)))


for name,param in tmp_model.named_parameters():
    param.requires_grad = False
for name,param in lianghua.named_parameters():
    param.requires_grad = False


channel = Channel(_iscomplex=_iscomplex)


_params = list(tmp_decoder.parameters())
optimizer = torch.optim.Adam(_params, lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,30], gamma = 0.2)
crit = Crit()



def train(model, model2,model3,device, train_loader, optimizer, epoch):

    model.eval()
    model2.eval()
    model3.train()
    if data_parallel: torch.cuda.synchronize()

    print('--------------------epoch: %d' % epoch)

    for batch_idx, (train_sents, len_batch) in enumerate(train_loader):
        train_sents = train_sents.to(device)  
        len_batch = len_batch.to(device) 

        optimizer.zero_grad()
        src = train_sents[:, 1:]
        trg = train_sents[:, :-1]
        trg_y = train_sents[:, 1:]
        src_mask = (src != 0).unsqueeze(-2).to(device)
        tgt_mask = make_std_mask(trg).to(device)
        output= model.encode(src, src_mask)
        out= model2.Q(output)

        snr = np.random.randint(-2,5)
        _snr2= np.random.randint(0,2)

        output2= channel.agwn_physical_layer(out, _snr=_snr1)
        output1= channel.agwn_physical_layer(out, _snr=_snr1)

        output2= sign(output2)
        output2= model2.dQ(output2)
        output1= sign(output1)
        output1= model2.dQ(output1)

        zero=torch.zeros_like(output1)
        if _snr2 == 0:
            output=torch.dstack((output1,output2))
        else:
            output=torch.dstack((output1,zero))


        output= model3.from_chanenl_embedding(output)
        output= model3.decode(output, src_mask,trg, tgt_mask)
        output= model3.generator.forward(output)

        loss = crit('xe', output, trg_y, len_batch)
        loss.backward()
        clip_gradient(optimizer, 0.1) 
        optimizer.step()

        if batch_idx%4000==0:
            print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item(),'snr={}'.format(snr))


    if epoch%10==0:
        torch.save(model3.state_dict(),
                   os.path.join(save_model_path, 'TRY1decoder2_epoch{}.pth'.format(epoch)))

        print("Epoch {} model saved!".format(epoch + 1))


# start training
for epoch in range(1, epochs):
    train(tmp_model, lianghua,tmp_decoder,device, train_data_loader, optimizer, epoch)
    scheduler.step()



