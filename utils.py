import re
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from collections import Counter

def create_data_loader(texts, numerical_features, labels, tokenizer, max_length, batch_size, device):
    texts = [str(text) for text in texts]
    
    inputs = tokenizer(texts, 
                       max_length=max_length, 
                       padding='max_length', 
                       truncation=True, 
                       return_tensors='pt')
    
    numerical_features = torch.tensor(np.array(numerical_features), dtype=torch.float32).to(device)
    labels = torch.tensor(labels).to(device)
    # import pdb;pdb.set_trace()
    dataset = TensorDataset(inputs['input_ids'].to(device), 
                            inputs['attention_mask'].to(device), 
                            numerical_features, 
                            labels)
    # import pdb;pdb.set_trace()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def create_data_loader_early(texts, labels, tokenizer, max_length, batch_size,device):
    texts = [str(text) for text in texts]
    
    inputs = tokenizer(texts, 
                       max_length=max_length, 
                       padding='max_length', 
                       truncation=True, 
                       return_tensors='pt')
    
    labels = torch.tensor(labels).to(device)
    # import pdb;pdb.set_trace()
    dataset = TensorDataset(inputs['input_ids'].to(device), 
                            inputs['attention_mask'].to(device),  
                            labels)
    # import pdb;pdb.set_trace()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)