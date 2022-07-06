#第二重要的部分，用于训练使用,这个是act =False，使用act版本



import os, platform
import torch
from math import log
import torch.nn as nn
import torch.nn.functional as F
from data_loader import Dataset_sentence, collate_func
from model import make_model,subsequent_mask,make_std_mask,make_decoder
from utils import Channel, Crit, clip_gradient
import torch.utils.data as data
import torch.optim as optim
import numpy as np

_iscomplex = True
batch_size = 64
epochs = 61
learning_rate = 1e-5  
epoch_start = 61  

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

tmp_model = make_model(vocab_size,vocab_size,act1=False,act2=False).to(device)  
tmp_model.load_state_dict(torch.load('./ckpt/TRY1part2_epoch{}.pth'.format(epoch_start-1)))

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

lianghua=DENSE().to(device)
lianghua.load_state_dict(torch.load('./ckpt/TRY1densepart2_epoch{}.pth'.format(60)))
tmp_model=tmp_model.train()
criterion = nn.MSELoss()

channel = Channel(_iscomplex=_iscomplex)

optimizer = torch.optim.Adam([{'params': lianghua.parameters(), 'lr': 3e-5}, {'params': tmp_model.parameters(), 'lr': 3e-5}])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,20,30], gamma = 0.3)
crit = Crit()



def train(model,model2, device, train_loader, optimizer, epoch):

    # set model as training mode
    model2.train()
    if data_parallel: torch.cuda.synchronize()

    this_batch_right, total_batch = 0, 0

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
        snr = np.random.randint(-2,5)
        snr2= np.random.randint(0,3)

        if snr2 == 0:
            out= model2.Q(output)
            out[:,:,30:]=0.
            out= channel.agwn_physical_layer(out, _snr=snr)
            out= sign(out)
            out[:,:,30:]=0.
        elif snr2 == 1:
            out= model2.Q(output)
            out[:,:,45:]=0.
            out= channel.agwn_physical_layer(out, _snr=snr)
            out= sign(out)
            out[:,:,45:]=0.
        else:
            out= model2.Q(output)
            out= channel.agwn_physical_layer(out, _snr=snr)
            out= sign(out)

        out= model2.dQ(out)
        output= model.from_channel_emb(out)
        output= model.decode(output, src_mask,trg, tgt_mask)
        output= model.generator.forward(output)
        loss = crit('xe', output, trg_y, len_batch)


        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()

        if batch_idx%4000==0:
            print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item(),'SNR2{}'.format(snr2),'SNR1{}dB'.format(snr))


    if epoch%10==0: 
        torch.save(model.module.state_dict() if data_parallel else model.state_dict(),
                   os.path.join(save_model_path, 'TRY1part3_epoch{}.pth'.format(epoch)))
        torch.save(model2.module.state_dict() if data_parallel else model2.state_dict(),
                   os.path.join(save_model_path, 'TRY1densepart3_epoch{}.pth'.format(epoch)))
        print("Epoch {} model saved!".format(epoch + 1))


# start training
for epoch in range(1, epochs):
    train(tmp_model,lianghua, device, train_data_loader, optimizer, epoch)
    scheduler.step()


