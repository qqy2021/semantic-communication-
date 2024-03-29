

import os, pickle, torch
from torch.utils.data import Dataset

class Dataset_sentence(Dataset):
    def __init__(self, _path):
        # E:\DOC\RESEARCH\MASTER\Dataset\Europarl
        if not _path: _path = r'C:\Users\10091\Desktop\Py\dataset'
        self._path = os.path.join(_path, 'english_vocab.pkl')
        self.dict = {}
        tmp = pickle.load(open(self._path, 'rb'))
        for kk,vv in tmp['voc'].items(): self.dict[kk] = vv+3
        # add sos, eos, and pad.
        self.dict['PAD'], self.dict['SOS'], self.dict['EOS'] = 0, 1, 2
        self.len_range = tmp['len_range']
        self.rev_dict = {vv: kk for kk, vv in self.dict.items()}
        self.data_num = [[1]+list(map(lambda t:self.dict[t], x.split(' '))) + [2]
                         + (self.len_range[1]-len(x.split(' ')))*[0]
                         for idx, x in enumerate(tmp['sent_str']) if idx%5!=0] 
        print('[*]------------vocabulary size is:----', self.get_dict_len())
    

    def __getitem__(self, index):
        return torch.tensor(self.data_num[index])

    def __len__(self):
        return len(self.data_num)

    def get_dict_len(self):
        return len(self.dict)


class Dataset_sentence_test(Dataset):
    def __init__(self, _path):
        # E:\DOC\RESEARCH\MASTER\Dataset\Europarl
        if not _path: _path = r'C:\Users\10091\Desktop\Py\dataset'
        self._path = os.path.join(_path, 'english_vocab.pkl')
        self.dict = {}
        tmp = pickle.load(open(self._path, 'rb'))
        for kk,vv in tmp['voc'].items(): self.dict[kk] = vv+3
        # add sos, eos, and pad.
        self.dict['PAD'], self.dict['SOS'], self.dict['EOS'] = 0, 1, 2
        self.len_range = tmp['len_range']
        self.rev_dict = {vv: kk for kk, vv in self.dict.items()}
        self.data_num = [list(map(lambda t:self.dict[t], x.split(' '))) + [2]
                         + (self.len_range[1]-len(x.split(' ')))*[0]
                         for idx, x in enumerate(tmp['sent_str']) if idx%5==0][:5000]    
        print('[*]------------vocabulary size is:----', self.get_dict_len())

    def __getitem__(self, index):
        return torch.tensor(self.data_num[index])

    def __len__(self):
        return len(self.data_num)

    def get_dict_len(self):
        return len(self.dict)


def collate_func(batch_tensor):
    # orig_len_batch = list(map(lambda s: sum(s != 0), batch_tensor))
    batch_tensor = sorted(batch_tensor, key=lambda s: -sum(s != 0))#此处按照pad数目进行排列
    batch_len = list(map(lambda s: sum(s != 0), batch_tensor))  # eos counted as well.
    #assert len_batch == sorted(len_batch, reverse=True), 'seq should be sorted before pack pad.'
    return torch.stack(batch_tensor, dim=0), torch.stack(batch_len, dim=0)

if __name__ == '__main__':
    import torch
    dataset = Dataset_sentence(r'C:\Users\10091\Desktop\Py\dataset')
    xx = torch.tensor([[1,4,2,5],[3,2,0,0]])
    print('done')
