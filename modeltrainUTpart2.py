



import os, platform
import torch
import torch.nn as nn
from math import log
from data_loader import Dataset_sentence, collate_func
from model import make_model,subsequent_mask,make_std_mask,make_decoder
from utils import Normlize_tx, Channel, Crit, clip_gradient
import torch.utils.data as data
import torch.optim as optim
import numpy as np

_iscomplex = True
batch_size = 64
epochs = 91
learning_rate = 3e-4 
epoch_start = 79  
time_penalty = 0.0001

save_model_path = "./ckpt/"
if 'Windows' in platform.system():
    data_path = r'C:\Users\10091\Desktop\Py\dataset'
else:
    data_path = '/data/zqy/act1/dataset'

if not os.path.exists(save_model_path): os.makedirs(save_model_path)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
data_parallel = False

train_loader_params = {'batch_size': batch_size,
                       'shuffle': True, 'num_workers':8,
                       'collate_fn': lambda x: collate_func(x),
                       'drop_last': True}
data_train = Dataset_sentence(_path = data_path)
train_data_loader = data.DataLoader(data_train,**train_loader_params)

vocab_size = data_train.get_dict_len()

tmp_model = make_model(vocab_size,vocab_size,act1=True,act2=True).to(device)
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

lianghua=DENSE().to(device)
tmp_model.load_state_dict(torch.load('./ckpt/BaseUT_epoch{}.pth'.format(50)))
lianghua.load_state_dict(torch.load('./ckpt/BaseUTpart1_epoch{}.pth'.format(50)))

channel = Channel(_iscomplex=_iscomplex)

optimizer = torch.optim.Adam([{'params': lianghua.parameters(), 'lr': 1e-5}, {'params': tmp_model.parameters(), 'lr': 1e-4}])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,20,30], gamma = 0.3)
crit = Crit()

act1=True
act2=True

def train(model, model2,device, train_loader, optimizer, epoch):
    model.train()
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
        if act1 == True:
            output,remainders1,n_updates1 = model.encode(src, src_mask)
            remainders1=torch.sum(remainders1)
            n_updates1=torch.sum(n_updates1)
            ponder_cost1=remainders1+n_updates1
        else:
            output= model.encode(src, src_mask)
        out= model2.Q(output)
        snr = np.random.randint(-2,5)
        out= channel.agwn_physical_layer(out, _snr=snr)
        out= sign(out)
        out= model2.dQ(out)

        output= model.from_channel_emb(out)
        if act2 == True:
            output,remainders2,n_updates2= model.decode(output, src_mask,trg, tgt_mask)
            remainders2=torch.sum(remainders2)
            n_updates2=torch.sum(n_updates2)
            ponder_cost2=remainders2+n_updates2
        else:
            output = model.decode(output, src_mask,trg, tgt_mask)
        output= model.generator.forward(output)

        loss = crit('xe', output, trg_y, len_batch)
        if act1 ==True:
            loss = loss + ponder_cost1*time_penalty*(1e-4)
        if act2 ==True:
            loss = loss + ponder_cost2*time_penalty*(3e-2)
        loss.backward()
        clip_gradient(optimizer, 0.1) 
        optimizer.step()

        if batch_idx%4000==0:
            print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item())

    if epoch%10==0: #== 0:

        torch.save(model.module.state_dict() if data_parallel else model.state_dict(),
                   os.path.join(save_model_path, 'BaseUTpart2_epoch{}.pth'.format(epoch)))
        torch.save(model2.module.state_dict() if data_parallel else model2.state_dict(),
                   os.path.join(save_model_path, 'BaseUTdensepart2_epoch{}.pth'.format(epoch)))
        print("Epoch {} model saved!".format(epoch + 1))


# start training
for epoch in range(1, epochs):
    train(tmp_model, lianghua,device, train_data_loader, optimizer, epoch)
    scheduler.step()


