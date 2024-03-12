# -*- coding: utf-8 -*-
"""
Created on Thu Mar  12 13:00:00 2024

@author: Mels
"""
import torch

from Dictionary import  load_dict
from AttentionModel import load_model

#%% load the data
if __name__ ==  "__main__":
        ## Load the list back from the Pickle file
    #with open('Dataset.pickle', 'rb') as f:
    #    Dataset = pickle.load(f)
        
    model = load_model()
    dictionary = load_dict()
    
    
    
    #%% test it 
    from Dataset import print_basstab
    
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context = ["GDAE","||||","---2","--5-","----","----","----"]
    context = torch.reshape(torch.LongTensor(dictionary.encode(context)), shape=(len(context),1))
    print_basstab(dictionary.decode(model.generate(idx = context, max_new_tokens=100)[0].tolist()))


