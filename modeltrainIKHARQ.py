



import os, platform
import torch
from math import log
from data_loader import Dataset_sentence, collate_func
from model import make_model,subsequent_mask,make_std_mask,make_decoder
from utils import Channel, Crit, clip_gradient
import torch.utils.data as data
import torch.optim as optim
import numpy as np


_iscomplex = True
batch_size = 64
epochs = 91
learning_rate = 3e-4  
epoch_start = 51  

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
                       'shuffle': True, 'num_workers':4,
                       'collate_fn': lambda x: collate_func(x),
                       'drop_last': True}
data_train = Dataset_sentence(_path = data_path)
train_data_loader = data.DataLoader(data_train,**train_loader_params)

vocab_size = data_train.get_dict_len()

tmp_model = make_model(vocab_size,vocab_size,act1=False,act2=False).to(device)  
tmp_decoder = make_decoder(vocab_size,vocab_size,N1=32).to(device)
tmp_model.load_state_dict(torch.load('./ckpt/TRY1_epoch{}.pth'.format(epoch_start-1)))

for name,param in tmp_model.named_parameters():
    param.requires_grad = False

channel = Channel(_iscomplex=_iscomplex)

_params = list(tmp_decoder.parameters())
optimizer = torch.optim.Adam(_params, lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [20,40], gamma = 0.5)
crit = Crit()



def train(model, model2,device, train_loader, optimizer, epoch):

    # set model as training mode
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

        output= model.encode(src, src_mask)
        _snr1= np.random.randint(-2,6)
        _snr2= np.random.randint(0,2)
        output2= channel.agwn(output, _snr=_snr1)
        output1= channel.agwn(output, _snr=_snr1)

        zero=torch.zeros_like(output1)
        if _snr2 == 0:
            output=torch.dstack((output1,output2))
        else:
            output=torch.dstack((output1,zero))

        output= model2.from_chanenl_embedding(output)
        output= model2.decode(output, src_mask,trg, tgt_mask)
        output= model2.generator.forward(output)

        loss = crit('xe', output, trg_y, len_batch)
        loss.backward()
        clip_gradient(optimizer, 0.1) 
        optimizer.step()

        if batch_idx%4000==0:
            print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item())

    if epoch%10==0: 
        torch.save(model2.state_dict(),
                   os.path.join(save_model_path, 'TRYdecoder1_epoch{}.pth'.format(epoch)))
        print("Epoch {} model saved!".format(epoch + 1))


for epoch in range(1, epochs):
    train(tmp_model, tmp_decoder,device, train_data_loader, optimizer, epoch)
    scheduler.step()


