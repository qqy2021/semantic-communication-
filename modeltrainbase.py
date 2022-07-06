#第二重要的部分，用于训练使用,这个是act =False，使用act版本

import os, platform
import torch
from math import log
from data_loader import Dataset_sentence, collate_func
from model import make_model,subsequent_mask,make_std_mask,make_decoder
from utils import Normlize_tx, Channel, Crit, clip_gradient
import torch.utils.data as data
import torch.optim as optim
import numpy as np

#_snr = 12
_iscomplex = True
batch_size = 64
epochs = 91
learning_rate = 3e-4  
epoch_start = 79  # only used when loading ckpt

# set path
save_model_path = "./ckpt/"
if 'Windows' in platform.system():
    data_path = r'C:\Users\10091\Desktop\Py\dataset'
else:
    data_path = '/data/zqy/act1/dataset'

if not os.path.exists(save_model_path): os.makedirs(save_model_path)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
data_parallel = False

# data loading parameters
train_loader_params = {'batch_size': batch_size,
                       'shuffle': True, 'num_workers':8,
                       'collate_fn': lambda x: collate_func(x),
                       'drop_last': True}
data_train = Dataset_sentence(_path = data_path)
train_data_loader = data.DataLoader(data_train,**train_loader_params)

vocab_size = data_train.get_dict_len()

tmp_model = make_model(vocab_size,vocab_size,act1=False,act2=False).to(device)  #此处修改act决定是否使用，此处有修改如果不行，请删除，用act时候加act=ture
tmp_decoder = make_decoder(vocab_size,vocab_size,N1=32).to(device)

#tmp_model.load_state_dict(torch.load('./ckpt/fadenew7_shared_epoch{}.pth'.format(epoch_start-1)))####################
#print('loaded ckpt at ./ckpt/fadenew7_shared_epoch{}.pth'.format(epoch_start-1))             #####################



channel = Channel(_iscomplex=_iscomplex)
_params = list(tmp_model.parameters())
optimizer = torch.optim.Adam(_params, lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,20,40], gamma = 0.5)
crit = Crit()



def train(model, device, train_loader, optimizer, epoch):

    # set model as training mode
    model.train()
    if data_parallel: torch.cuda.synchronize()


    print('--------------------epoch: %d' % epoch)

    for batch_idx, (train_sents, len_batch) in enumerate(train_loader):
        # distribute data to device
        train_sents = train_sents.to(device)  # with eos
        #print(train_sents)############################
        len_batch = len_batch.to(device) #cpu()
#感觉修改此处即可
        optimizer.zero_grad()
        src = train_sents[:, 1:]
        trg = train_sents[:, :-1]
        trg_y = train_sents[:, 1:]
        src_mask = (src != 0).unsqueeze(-2).to(device)
        tgt_mask = make_std_mask(trg).to(device)
        ##output= model.forward(src,trg,src_mask, tgt_mask,len_batch)##改了
        output= model.encode(src, src_mask)
        _snr1= np.random.randint(-2,5)
        output= channel.agwn(output, _snr=_snr1)
        output= model.from_channel_emb(output)
        output= model.decode(output, src_mask,trg, tgt_mask)
        output= model.generator.forward(output)

        loss = crit('xe', output, trg_y, len_batch)
        loss.backward()
        clip_gradient(optimizer, 0.1) 
        optimizer.step()

        if batch_idx%4000==0:
            print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item())


    if epoch%10==0: #== 0:
        torch.save(model.module.state_dict() if data_parallel else model.state_dict(),
                   os.path.join(save_model_path, 'TRY1_epoch{}.pth'.format(epoch)))
        print("Epoch {} model saved!".format(epoch + 1))


# start training
for epoch in range(1, epochs):
    train(tmp_model, device, train_data_loader, optimizer, epoch)
    scheduler.step()


