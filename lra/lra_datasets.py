import numpy as np
import pandas as pd
import pickle
from functools import reduce
import torch
from glob import glob
from itertools import cycle


class ImdbDataset:
    def __init__(self, config, split='train'):       
        data_paths = {'train': "datasets/aclImdb/train", 'eval': "datasets/aclImdb/test"}
        split_path = data_paths[split]
        neg_path = split_path + "/neg"
        pos_path = split_path + "/pos"
        neg_inputs = zip(glob(neg_path+"/*.txt"), cycle([0]))
        pos_inputs = zip(glob(pos_path+"/*.txt"), cycle([1]))
        self.data = np.random.permutation(list(neg_inputs) + list(pos_inputs))
        
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data[i]
        with open(data[0], 'r') as fo:
            source = fo.read()
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = int(data[1])
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)


class ListOpsDataset:
    def __init__(self, config, split='train'):
        
        data_paths = {'train': "datasets/lra_release/listops-1000/basic_train.tsv",
                      'eval': "datasets/lra_release/listops-1000/basic_val.tsv"}
        self.data = pd.read_csv(data_paths[split], delimiter='\t')
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        source = data.Source
        inputs = self.tokenizer(source, max_length=self.max_length) #return_tensors='pt', truncation=True, padding='max_length'
        target = data.Target
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)


class Cifar10Dataset:
    def __init__(self, config, split='train'):
        data_paths = {'train': [f"datasets/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)],
                      'eval': ["datasets/cifar-10-batches-py/test_batch"]
                     }
        print("loading cifar-10 data...")
        data_dicts = [Cifar10Dataset.unpickle(path) for path in data_paths[split]]
        print("assembling cifar-10 files..")
        self.data = reduce((lambda x, y: {b'data': np.concatenate([x[b'data'], y[b'data']], axis=0), 
                                         b'labels': np.concatenate([x[b'labels'], y[b'labels']], axis=0)}), 
                           data_dicts)
        # TODO CHECK: i think this is the right shape 
        # see: https://www.cs.toronto.edu/~kriz/cifar.html 
        #      section "Dataset layouts" discusses the memory layout of the array
        self.data[b'data'] = self.data[b'data'].reshape((-1, 3, 1024)) 
       
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
    
    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d
    
    def __getitem__(self, i):
        r, g, b = self.data[b'data'][i]
        # grayscale image (assume pixels in [0, 255])
        source = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(int)
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = self.data[b'labels'][i]
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data[b'data'])
